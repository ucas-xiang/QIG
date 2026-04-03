import gc
import os
import copy
import functools
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .qmodule import ScaledActivation
from .quantizer import get_module_by_name_suffix
from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear

__all__ = ["auto_scale_block_wa_distort"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    """Compute per-channel weight scale statistics."""
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    """Compute per-channel activation scale: [B, T, C] -> [C]."""
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """
    Apply channel-wise scaling across (Norm -> Linear(s)):
    - Divide LN/RMSNorm weights (and bias) by scales
    - Multiply following Linear weights by scales
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """
    Apply channel-wise scaling across (Linear -> Linear).
    Note: keeps the original behavior of only scaling the last `scales.size(0)` rows of fc1.
    """
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    # Keep original behavior: only scale the tail rows
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    """Apply channel-wise scaling across (GELU-like activation -> Linear)."""
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block_wa_distort(
    module,
    module_kwargs,
    w_bit,
    a_bit,
    q_config,
    input_feat,
    ans_mask,
    vis_mask,
    reweight_ratio_dict,
    q_input,
    loss_mode="mae",
):
    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def _search_module_scale_wa_distort(
        block,
        linears2scale: list,
        layers_name,
        x,
        x_q,
        reweight_ratio,
        kwargs={},
        ans_mask=ans_mask,
        vis_mask=vis_mask,
        compute_token_importance=False,
    ):
        """
        Search best per-channel scales for the provided `linears2scale` under WA quantization.

        Steps:
        1) Compute baseline output using quantized input x_q (with FP block weights).
        2) Build a quantized version of the block (`block_q`) using WALinear.from_float.
        3) Compute token importance weights using a QIG-style integrated gradients
           between FP and quantized blocks on the interpolated inputs.
        4) Grid search scales; for each scale:
           - scale weights
           - replace target linear modules with WALinear
           - scale input by 1/scales
           - forward and compute token-weighted loss vs baseline
           - restore original modules / state
        """
        x = x.to(next(block.parameters()).device)
        x_q = x_q.to(next(block.parameters()).device)

        # Baseline output computed on x_q using the original (float) block
        with torch.no_grad():
            org_out = block(x_q, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        # Build a quantized copy of the block (WALinear replacement)
        block_q = copy.deepcopy(block)

        # Special case: if the "block" itself is a Linear (e.g., InternLM2 w2)
        if isinstance(block_q, nn.Linear):
            block_q = WALinear.from_float(
                block_q,
                weight_quant="per_channel",
                act_quant="per_token",
                w_bit=w_bit,
                a_bit=a_bit,
            )
        else:
            # Only traverse/replace for composite blocks that contain submodules
            for name, m in block_q.named_modules():
                if isinstance(m, nn.Linear):
                    father = get_module_by_name_suffix(block_q, ".".join(name.split(".")[:-1]))
                    setattr(
                        father,
                        name.split(".")[-1],
                        WALinear.from_float(
                            m,
                            weight_quant="per_channel",
                            act_quant="per_token",
                            w_bit=w_bit,
                            a_bit=a_bit,
                        ),
                    )

        def _compute_token_importance_weights(
            block_fp,
            block_q,
            x,
            x_q,
            kwargs,
            loss_mode="mae",  # kept for compatibility with callers
        ):
            """
            Compute token importance weights (QIG style) by integrating gradients
            of the per-token discrepancy between FP and quantized blocks.

            We compute IG on the per-token magnitude of (f_fp(x) - f_q(x)) rather than
            collapsing to a single scalar too early, so we get true token-wise weights.
            """
            device = next(block_fp.parameters()).device
            param_dtype = next(block_fp.parameters()).dtype

            x = x.to(device, dtype=param_dtype)
            x_q = x_q.to(device, dtype=param_dtype)
            B, T, H = x.shape

            # Ensure use_cache does not interfere
            kwargs = dict(kwargs)
            if "use_cache" in kwargs:
                kwargs["use_cache"] = False

            # Preserve and restore training/eval modes
            was_training_fp = block_fp.training
            was_training_q = block_q.training
            block_fp.eval()
            block_q.eval()

            steps = 32
            alphas = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)
            weights = torch.ones_like(alphas) / steps

            # Baseline for IG is x_q
            x_baseline = x_q.detach().clone().to(device=device, dtype=param_dtype)

            total_grad = torch.zeros_like(x, dtype=torch.float32, device=device)

            for i, alpha in enumerate(alphas):
                # Interpolate in float, then cast back to param dtype, then enable grad
                x_interp = (x_baseline + alpha.to(param_dtype) * (x - x_baseline))
                x_interp = x_interp.detach().clone().requires_grad_(True)

                with torch.enable_grad():
                    y_fp = block_fp(x_interp, **kwargs)
                    y_q = block_q(x_interp, **kwargs)

                    if isinstance(y_fp, tuple):
                        y_fp = y_fp[0]
                    if isinstance(y_q, tuple):
                        y_q = y_q[0]

                    # Per-token discrepancy signal
                    diff = (y_fp - y_q) * 1000  # keep original scaling
                    F_tok = diff.abs().mean(dim=-1)  # [B, T]

                    # Vectorized gradient for per-token loss
                    grad_outputs = torch.ones_like(F_tok, device=F_tok.device, dtype=F_tok.dtype)
                    grad = torch.autograd.grad(
                        outputs=F_tok,
                        inputs=x_interp,
                        grad_outputs=grad_outputs,
                        retain_graph=False,
                        create_graph=False,
                    )[0]  # [B, T, H]

                if grad is None:
                    grad = torch.zeros_like(x_interp, dtype=param_dtype, device=device)

                total_grad += grad.detach().to(torch.float32) * weights[i].to(torch.float32)

                del grad, y_fp, y_q, diff, F_tok, x_interp

            integrated_grads = (x - x_baseline).to(torch.float32) * total_grad  # [B, T, H]
            token_importance = integrated_grads.abs().mean(dim=-1)  # [B, T]

            # Normalize to sum=1 per sample
            eps = torch.finfo(torch.float32).tiny
            token_importance = token_importance / token_importance.sum(dim=1, keepdim=True).clamp_min(eps)

            # IQR bounds (kept for debugging/compat), but original code ultimately used raw token_importance
            q1 = torch.quantile(token_importance, 0.25, dim=1, keepdim=True)
            q3 = torch.quantile(token_importance, 0.75, dim=1, keepdim=True)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            # Keep original behavior: do NOT clip; use raw normalized importance
            weights_tok = token_importance

            if was_training_fp:
                block_fp.train()
            if was_training_q:
                block_q.train()

            iqr_vis = {
                "weights_raw": token_importance.mean(0).detach(),
                "weights_clipped": weights_tok.mean(0).detach(),
                "lower_bound": lower.mean(0).detach(),
                "upper_bound": upper.mean(0).detach(),
                "iqr_alpha": 1.5,
            }

            return weights_tok.detach(), iqr_vis

        # Token weights: always compute with QIG to avoid any call-order / layer gating
        token_w, _ = _compute_token_importance_weights(block, block_q, x, x_q, kwargs, loss_mode=loss_mode)

        # block_q is no longer needed after token weights are computed
        del block_q
        torch.cuda.empty_cache()

        x_max = get_act_scale(x_q)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}

        for r in range(n_grid):
            ratio = r * 1.0 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            # If block is a single Linear, we may need a temporary quantized module
            new_block = None
            if isinstance(block, nn.Linear):
                new_block = block

            # Replace each target linear module with WALinear under scaled weights
            for fc, fc_name in zip(linears2scale, layers_name):
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                new_fc = WALinear.from_float(
                    fc,
                    weight_quant="per_channel",
                    act_quant="per_token",
                    w_bit=w_bit,
                    a_bit=a_bit,
                )

                if isinstance(block, nn.Linear):
                    new_block = copy.deepcopy(new_fc)
                else:
                    setattr(block, fc_name, new_fc)

                del new_fc
                torch.cuda.empty_cache()

            # Input scaling (distort-style)
            x_scale = x_q / (scales.view(1, 1, -1))

            if isinstance(block, nn.Linear):
                out = new_block(x_scale, **kwargs)
            else:
                out = block(x_scale, **kwargs)

            if isinstance(out, tuple):
                out = out[0]

            eps = 1e-8
            if loss_mode == "mse":
                pos_change = (org_out - out).float().pow(2).mean(dim=-1)  # [B, T]
            else:
                pos_change = (org_out - out).float().abs().mean(dim=-1)  # [B, T]

            # Keep original structure: sample_mag computed but not used in loss (as in your code)
            sample_mag = pos_change.detach().mean(dim=1, keepdim=True).clamp_min(eps)
            _ = token_w * sample_mag  # computed but intentionally unused to preserve original behavior

            loss = (pos_change * token_w).sum() / token_w.sum().clamp_min(eps)

            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales

            # Restore original modules
            for fc, fc_name in zip(linears2scale, layers_name):
                if isinstance(block, nn.Linear):
                    continue
                setattr(block, fc_name, fc)

            if isinstance(block, nn.Linear):
                del new_block

            torch.cuda.empty_cache()
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            raise RuntimeError("Failed to find best ratio/scales in grid search.")

        best_scales = best_scales.view(-1)
        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale_wa_distort(
        prev_op,
        layers,
        layers_name,
        inp,
        inp_q,
        reweight_ratio,
        module2inspect=None,
        kwargs={},
        compute_token_importance=False,
    ):
        """
        Wrapper producing (prev_op_name, layer_names, scales_cpu) compatible with apply_scale().
        """
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale_wa_distort(
            module2inspect,
            layers,
            layers_name,
            inp,
            inp_q,
            reweight_ratio,
            kwargs,
            compute_token_importance=compute_token_importance,
        ).detach().cpu()

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []

    def _auto_get_input_feat_distort(inps_q, scales_list=None):
        """
        Collect input features for (WALinear) modules:
        - Work on a deepcopy of `module` to avoid mutating the original module
        - Optionally apply already-found scales
        - Replace nn.Linear with WALinear
        - Run forward once and cache inputs via hooks
        """
        new_module = copy.deepcopy(module)

        named_linears = {
            name: m for name, m in new_module.named_modules() if isinstance(m, nn.Linear)
        }

        if scales_list is not None:
            apply_scale(new_module, scales_list)
            new_module.cuda()

            for n, m in named_linears.items():
                new_linear = WALinear.from_float(
                    m,
                    weight_quant="per_channel",
                    act_quant="per_token",
                    w_bit=w_bit,
                    a_bit=a_bit,
                )
                father_module = get_module_by_name_suffix(new_module, ".".join(n.split(".")[:-1]))
                setattr(father_module, n.split(".")[-1], new_linear)
                del new_linear, m
                torch.cuda.empty_cache()

            named_linears = {
                name: m for name, m in new_module.named_modules() if isinstance(m, WALinear)
            }

        def cache_input_hook(m, x, y, name, feat_dict):
            x0 = x[0].detach().cpu()
            feat_dict[name].append(x0)

        input_feat_q = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat_q)
                )
            )

        inps_q = inps_q.to(next(new_module.parameters()).device)
        new_module(inps_q, **module_kwargs)

        for h in handles:
            h.remove()

        input_feat_q = {k: torch.cat(v, dim=0) for k, v in input_feat_q.items()}

        del new_module
        torch.cuda.empty_cache()
        return input_feat_q

    # ===================== Architecture branches =====================

    if isinstance(module, OPTDecoderLayer):
        # NOTE: This branch uses _auto_get_scale which is not defined in this snippet.
        # Kept as-is to match your original structure.
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
                module2inspect=module.self_attn.out_proj,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
                module2inspect=module.fc1,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
                module2inspect=module.fc2,
            )
        )

    elif isinstance(module, LlamaDecoderLayer):
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                layers_name=["q_proj", "k_proj", "v_proj"],
                inp=input_feat["self_attn.q_proj"],
                inp_q=input_feat_q["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale_wa_distort(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    layers_name=["o_proj"],
                    inp=input_feat["self_attn.o_proj"],
                    inp_q=input_feat_q["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                    module2inspect=module.self_attn.o_proj,
                )
            )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                layers_name=["gate_proj", "up_proj"],
                inp=input_feat["mlp.gate_proj"],
                inp_q=input_feat_q["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                layers_name=["down_proj"],
                inp=input_feat["mlp.down_proj"],
                inp_q=input_feat_q["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp.down_proj,
            )
        )

    elif isinstance(module, BloomBlock):
        # NOTE: This branch uses _auto_get_scale which is not defined in this snippet.
        # Kept as-is to match your original structure.
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
                module2inspect=module.mlp.dense_4h_to_h,
            )
        )

    elif "mpt" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
                module2inspect=module.attn.out_proj,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.act,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
                module2inspect=module.ffn.down_proj,
            )
        )

    elif "falcon" in str(module.__class__).lower():
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.input_layernorm,
                    layers=[module.mlp.dense_h_to_4h, module.self_attention.query_key_value],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_attn,
                    layers=[module.self_attention.query_key_value],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_mlp,
                    layers=[module.mlp.dense_h_to_4h],
                    inp=input_feat["mlp.dense_h_to_4h"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        else:
            raise NotImplementedError("Unknown Falcon variant")

        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
                module2inspect=module.mlp.dense_4h_to_h,
            )
        )

    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc],
                inp=input_feat["mlp.c_fc"],
                module2inspect=module.mlp,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
                module2inspect=module.mlp.c_proj,
            )
        )

    elif "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_4h"] if "mlp.dense_4h_to_4h" in input_feat else input_feat["mlp.dense_4h_to_h"],
                module2inspect=module.mlp.dense_4h_to_h,
            )
        )

    elif module.__class__.__name__ in ("Qwen2DecoderLayer", "Qwen2VLDecoderLayer"):
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                layers_name=["q_proj", "k_proj", "v_proj"],
                inp=input_feat["self_attn.q_proj"],
                inp_q=input_feat_q["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
                compute_token_importance=True,
            )
        )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale_wa_distort(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    layers_name=["o_proj"],
                    inp=input_feat["self_attn.o_proj"],
                    inp_q=input_feat_q["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                    module2inspect=module.self_attn.o_proj,
                    compute_token_importance=True,
                )
            )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                layers_name=["gate_proj", "up_proj"],
                inp=input_feat["mlp.gate_proj"],
                inp_q=input_feat_q["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
                compute_token_importance=True,
            )
        )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                layers_name=["down_proj"],
                inp=input_feat["mlp.down_proj"],
                inp_q=input_feat_q["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp.down_proj,
                compute_token_importance=True,
            )
        )

    elif module.__class__.__name__ == "InternLM2DecoderLayer":
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.attention_norm,
                layers=[module.attention.wqkv],
                layers_name=["wqkv"],
                inp=input_feat["attention.wqkv"],
                inp_q=input_feat_q["attention.wqkv"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
                compute_token_importance=True,
            )
        )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        if module.attention.wqkv.weight.shape == module.attention.wo.weight.shape:
            scales_list.append(
                _auto_get_scale_wa_distort(
                    prev_op=module.attention.wqkv,
                    layers=[module.attention.wo],
                    layers_name=["wo"],
                    inp=input_feat["attention.wo"],
                    inp_q=input_feat_q["attention.wo"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                    module2inspect=module.attention.wo,
                    compute_token_importance=True,
                )
            )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.ffn_norm,
                layers=[module.feed_forward.w1, module.feed_forward.w3],
                layers_name=["w1", "w3"],
                inp=input_feat["feed_forward.w1"],
                inp_q=input_feat_q["feed_forward.w1"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.feed_forward,
                compute_token_importance=True,
            )
        )

        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.feed_forward.w3,
                layers=[module.feed_forward.w2],
                layers_name=["w2"],
                inp=input_feat["feed_forward.w2"],
                inp_q=input_feat_q["feed_forward.w2"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.feed_forward.w2,
                compute_token_importance=True,
            )
        )

    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None):
    """
    Apply computed scales to `module` in-place.
    This matches the interface/behavior you used elsewhere.
    """
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ in (
            "InternLM2RMSNorm",
            "Qwen2RMSNorm",
        ):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # Optionally apply the scaling to cached input features
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()