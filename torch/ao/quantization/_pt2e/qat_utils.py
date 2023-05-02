import copy
import operator
from typing import Any, Callable, Tuple

import torch
from torch.fx import GraphModule, Node
from torch.fx.subgraph_rewriter import _replace_pattern
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from .utils import _fold_bn_weights_into_conv_node
from torch.ao.quantization.fx.utils import create_getattr_from_value

# Example inputs for both `_conv2d_bn_pattern` and `_qat_conv2d_bn_pattern`
_conv2d_bn_pattern_example_inputs = (
    torch.randn(1, 1, 3, 3),  # x
    torch.randn(1, 1, 1, 1),  # conv_weight
    torch.randn(1),           # conv_bias
    torch.randn(1),           # bn_weight
    torch.randn(1),           # bn_bias
    torch.randn(1),           # bn_running_mean
    torch.randn(1),           # bn_running_var
)

# Example inputs for both `_quantized_qat_conv2d_bn_pattern` and `_folded_quantized_qat_conv2d_bn_pattern`
_quantized_conv2d_bn_pattern_example_inputs = (
    torch.randn(1, 1, 3, 3).to(torch.int8),  # x
    torch.randn(1, 1, 1, 1),  # conv_weight
    torch.randn(1),           # conv_bias
    torch.randn(1),           # bn_weight
    torch.randn(1),           # bn_bias
    torch.randn(1),           # bn_running_mean
    torch.randn(1),           # bn_running_var
)

def _conv2d_bn_pattern(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True)
    return x

def _qat_conv2d_bn_pattern(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """
    Approximated method to fuse conv and bn. It requires only one forward pass.
    conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std.
    This is based on `nniqat.ConvBn2d._forward_approximate`.
    """
    # TODO: allow setting eps
    bn_eps = 1e-5
    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
    x = F.conv2d(x, scaled_weight, zero_bias)
    x = x / scale_factor.reshape(bias_shape)
    x = x + conv_bias.reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    return x

def _quantized_qat_conv2d_bn_pattern(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """
    Quantized version of qat conv bn pattern,
    This is based on `nniqat.ConvBn2d._forward_approximate`.
    used in qat convert, we first match this pattern and then replace it with
    normal conv - bn pattern and then fold the weights of bn into conv
    """
    # TODO: allow setting eps
    bn_eps = 1e-5
    # TODO: make scale/zero_point arguments after we update them to
    # Tensor
    weight_scale = 1
    weight_zero_point = 0
    weight_quant_min = -127
    weight_quant_max = 127
    input_scale = 1
    input_zero_point = 0
    input_quant_min = -128
    input_quant_max = 127
    output_scale = 1
    output_zero_point = 0
    output_quant_min = -128
    output_quant_max = 127

    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, input_scale, input_zero_point, input_quant_min, input_quant_max, torch.int8)
    zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
    scaled_weight = torch.ops.quantized_decomposed.quantize_per_tensor(scaled_weight, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    scaled_weight = torch.ops.quantized_decomposed.dequantize_per_tensor(scaled_weight, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    x = F.conv2d(x, scaled_weight, zero_bias)
    x = x / scale_factor.reshape(bias_shape)
    x = x + conv_bias.reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    x = torch.ops.quantized_decomposed.quantize_per_tensor(x, output_scale, output_zero_point, output_quant_min, output_quant_max, torch.int8)
    return x

def _folded_quantized_qat_conv2d_bn_pattern(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """ Quantized QAT conv - bn pattern with bn weights being folded into conv
    """
    # TODO: allow setting eps
    bn_eps = 1e-5
    # TODO: make scale/zero_point arguments after we update them to
    # Tensor
    weight_scale = 1
    weight_zero_point = 0
    weight_quant_min = -127
    weight_quant_max = 127
    input_scale = 1
    input_zero_point = 0
    input_quant_min = -128
    input_quant_max = 127
    output_scale = 1
    output_zero_point = 0
    output_quant_min = -128
    output_quant_max = 127

    x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, input_scale, input_zero_point, input_quant_min, input_quant_max, torch.int8)
    conv_weight = torch.ops.quantized_decomposed.quantize_per_tensor(conv_weight, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    conv_weight = torch.ops.quantized_decomposed.dequantize_per_tensor(conv_weight, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    x = torch.ops.quantized_decomposed.quantize_per_tensor(x, output_scale, output_zero_point, output_quant_min, output_quant_max, torch.int8)
    return x

def _get_aten_graph_module(
    pattern: Callable,
    example_inputs: Tuple[Any, ...],
) -> GraphModule:
    """
    Convert the pattern to an FX graph with decomposed aten ops.
    """
    # Avoid circular imports
    import torch._dynamo
    aten_pattern, _ = torch._dynamo.export(
        pattern,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )
    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern

def _fuse_conv_bn_qat(m: GraphModule) -> GraphModule:
    """
    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with
    the fused QAT subgraph equivalent. The input graph should already be annotated.
    The annotations in the original nodes will be preserved in the corresponding
    nodes in the new subgraph.
    """
    m.graph.eliminate_dead_code()
    m.recompile()
    example_inputs = _conv2d_bn_pattern_example_inputs
    match_pattern = _get_aten_graph_module(_conv2d_bn_pattern, example_inputs)
    replacement_pattern = _get_aten_graph_module(_qat_conv2d_bn_pattern, example_inputs)
    # TODO: use the public replace_pattern API once it also returns replacement nodes
    match_and_replacement = _replace_pattern(m, match_pattern, replacement_pattern)
    m.recompile()

    # Copy over metadata from original subgraph
    # This ensures the stack traces and annotations are preserved in the new subgraph
    # TODO: handle this in replace_pattern
    for mr in match_and_replacement:
        # Find replacement conv and bn nodes by climbing upwards from anchor node
        assert len(mr.replacements) == 1, "expected only one replacement node"
        replacement_conv_node = None
        replacement_bn_node = None
        replacement_getitem_node = mr.replacements[0]
        assert replacement_getitem_node.target == operator.getitem
        n = replacement_getitem_node
        while replacement_conv_node is None or replacement_bn_node is None:
            if n.target == torch.ops.aten.convolution.default:
                replacement_conv_node = n
            if n.target == torch.ops.aten._native_batch_norm_legit.default:
                replacement_bn_node = n
            assert isinstance(n.args[0], Node)
            n = n.args[0]

        # Copy over metadata for all three nodes in [conv - bn - getitem]
        for match_pattern_node, original_node in mr.nodes_map.items():
            if original_node.target == torch.ops.aten.convolution.default:
                replacement_conv_node.meta = original_node.meta
            if original_node.target == torch.ops.aten._native_batch_norm_legit.default:
                replacement_bn_node.meta = original_node.meta
            if original_node.target == operator.getitem:
                replacement_getitem_node.meta = original_node.meta
    return m

def _fold_conv_bn_qat(m: GraphModule) -> GraphModule:
    """
    Replace the quantized (conv + bn) pattern with conv with bn weights folded into the weights of conv.
    """
    # Workaround: current convert does not produce q/dq ops with a specific overload
    # we'll add the overload here as a workaround since we do not want to break
    # BC for now
    for n in m.graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.quantized_decomposed.quantize_per_tensor:
            n.target = torch.ops.quantized_decomposed.quantize_per_tensor.default
        if n.op == "call_function" and n.target == torch.ops.quantized_decomposed.dequantize_per_tensor:
            n.target = torch.ops.quantized_decomposed.dequantize_per_tensor.default

    m.graph.eliminate_dead_code()
    m.recompile()
    example_inputs = _quantized_conv2d_bn_pattern_example_inputs
    match_pattern = _get_aten_graph_module(_quantized_qat_conv2d_bn_pattern, example_inputs)
    replacement_pattern = _get_aten_graph_module(_folded_quantized_qat_conv2d_bn_pattern, example_inputs)

    # Workaround: match first to get the original graph and extra the scale/zero_point for
    # weight, since it got replaced with constant right now
    # if we just did replace_pattern, then the scale and zero_point nodes will be erased in
    # the original graph and we won't be able to access them anymore.
    # In the future, we should change the type for scale/zero_point to be Tensor so
    # that we can use them as arguments for pattern and replacement graphs
    matcher = SubgraphMatcher(match_pattern.graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True, ignore_literals=True)
    _matches: List[InternalMatch] = matcher.match(m.graph)
    _match_map = {match.anchors[0]: match for match in _matches}
    anchor_node_to_weight_scale_and_zp = {}
    for match in _matches:
        anchor_node = match.anchors[0]
        assert anchor_node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        conv_node = None
        n = match.nodes_map[anchor_node]
        while conv_node is None:
            if n.target == torch.ops.aten.convolution.default:
                conv_node = n
            n = n.args[0]

        conv_weight_dq = conv_node.args[1]
        assert conv_weight_dq.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
        weight_scale_name = conv_weight_dq.args[1].target
        weight_zp_name = conv_weight_dq.args[2].target
        weight_scale = getattr(m, weight_scale_name)
        weight_zp = getattr(m, weight_zp_name)
        anchor_node_to_weight_scale_and_zp[anchor_node] = (weight_scale, weight_zp)
    # END of Workaround

    # TODO: use the public replace_pattern API once it also returns replacement nodes
    match_and_replacement = _replace_pattern(m, match_pattern, replacement_pattern, ignore_literals=True)
    m.recompile()

    def _copy_scale_and_zp_arg_for_q_dq_node(node: Node, scale: Any, zp: Any):
        args = list(node.args)
        args[1] = scale
        args[2] = zp
        node.args = tuple(args)

    for mr in match_and_replacement:
        # Find replacement conv and bn nodes by climbing upwards from anchor node
        assert len(mr.replacements) == 1, "expected only one replacement node"

        # find conv, bn, weight, bias nodes in the graph
        replacement_quantize_node = mr.replacements[0]
        assert replacement_quantize_node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        n = replacement_quantize_node
        conv_node = None
        bn_node = None
        while conv_node is None or bn_node is None:
            if n.target == torch.ops.aten.convolution.default:
                conv_node = n
            if n.target == torch.ops.aten._native_batch_norm_legit.default:
                bn_node = n
            assert isinstance(n.args[0], Node)
            n = n.args[0]
        assert conv_node is not None and bn_node is not None

        conv_weight_dq = conv_node.args[1]
        assert conv_weight_dq.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
        conv_weight_q = conv_weight_dq.args[0]
        assert conv_weight_q.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        conv_weight = conv_weight_q.args[0]
        assert conv_weight.op == "get_attr"
        conv_bias = conv_node.args[2]

        # Workaround: Restore the scale/zero_point arguments for the ops in the pattern
        # since we are using constant scale/zero_point right now
        # future fix: use the Tensor variant of quantize/dequantize ops everywhere
        # and remove the hack
        # ops that needs fix: input of conv (dq), output of bn (q), weight of conv (q/dq)
        # replacement subgraph:
        #            dq -> conv -> bn -> q
        # weight -> q -> dq /
        # for input of conv and output of bn, since we only match one dq for conv and one q for bn, we can copy over the qparam from the other non matched q/dq node
        # for weight, both q/dq are matched, so the param is lost, that's why we stored
        # them in `anchor_node_to_weight_scale_and_zp`, indexed by anchor node for the
        # pattern
        conv_input_dq = conv_node.args[0]
        assert conv_input_dq.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
        conv_input_q = conv_input_dq.args[0]
        assert conv_input_q.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        _copy_scale_and_zp_arg_for_q_dq_node(conv_input_dq, conv_input_q.args[1], conv_input_q.args[2])

        # output of bn
        bn_output_q = replacement_quantize_node
        bn_output_dq = list(bn_output_q.users)[0]
        assert bn_output_dq.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
        _copy_scale_and_zp_arg_for_q_dq_node(bn_output_q, bn_output_dq.args[1], bn_output_dq.args[2])

        # END of Workaround

        # q/dq for conv_weight
        weight_scale, weight_zp = anchor_node_to_weight_scale_and_zp[mr.anchor]
        with m.graph.inserting_before(conv_weight_q):
            conv_input_scale_name = conv_input_q.args[1].target
            conv_input_zp_name = conv_input_q.args[2].target
            weight_scale_node = create_getattr_from_value(m, m.graph, conv_input_scale_name + "_weight", weight_scale)
            weight_zp_node = create_getattr_from_value(m, m.graph, conv_input_zp_name + "_weight", weight_zp)
            _copy_scale_and_zp_arg_for_q_dq_node(conv_weight_q, weight_scale_node, weight_zp_node)
            _copy_scale_and_zp_arg_for_q_dq_node(conv_weight_dq, weight_scale_node, weight_zp_node)

        # fold bn weights into conv
        _fold_bn_weights_into_conv_node(conv_node, conv_weight, conv_bias, bn_node, m)

    m.graph.eliminate_dead_code()
    m.recompile()
    return m
