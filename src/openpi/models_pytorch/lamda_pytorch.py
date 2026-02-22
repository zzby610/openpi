"""LaMDA model wrapper for OpenPI framework.

Wraps LaMDAModelLM (Diffusion-VLM with und/gen dual branches) for action
prediction. Only the understanding (und) branch is used; the generation
branch is disabled at construction time via ``visual_gen=False``.
"""

from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

# --- 导入你的原厂模型组件 ---
from models.diff_transformer import LLaDAConfig, LLaDAModelLM
from models.vision_transformer import SiglipVisionConfig, SiglipVisionModel
from diffusion.lamda import LaMDAConfig, LaMDAModelLM
from utils.data_utils import patchify

import openpi.models.gemma as _gemma


class LaMDAPytorch(nn.Module):
    """Thin wrapper around LaMDAModelLM with gen branch disabled."""

    def __init__(self, lamda_model_path: str, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        super().__init__()
        
        llm_config = LLaDAConfig.from_pretrained(lamda_model_path)
        
        try:
            vit_config = SiglipVisionConfig.from_pretrained(lamda_model_path)
        except Exception:
            vit_config = SiglipVisionConfig(
                hidden_size=1152, image_size=518, patch_size=14,
                num_hidden_layers=27, num_attention_heads=16,
            )
            
        if not hasattr(vit_config, "rope"):
            vit_config.rope = False
            
        language_model_backbone = LLaDAModelLM(config=llm_config)
        vit_model_backbone = SiglipVisionModel(config=vit_config)
        
        lamda_config = LaMDAConfig(
            visual_gen=False, visual_und=True,      
            llm_config=llm_config, vit_config=vit_config,
            vae_config=None, latent_patch_size=2,
            max_latent_size=32, vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            interpolate_pos=False, timestep_shift=1.0,
        )
        
        self.lamda = LaMDAModelLM(
            language_model=language_model_backbone, 
            vision_model=vit_model_backbone, 
            config=lamda_config
        )
        
        self.lamda.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        self._apply_precision(precision)

    def _apply_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    @property
    def vit_model(self): return self.lamda.vit_model

    @property
    def connector(self): return self.lamda.connector

    @property
    def language_model(self): return self.lamda.language_model

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        p = self.lamda.config.vit_config.patch_size
        device = image.device
        dtype = torch.bfloat16

        all_patches = []
        for i in range(B):
            p_img = patchify(image[i].to(dtype=dtype), p)
            all_patches.append(p_img.to(device=device, dtype=dtype).contiguous())
        
        packed_vit_tokens = torch.cat(all_patches, dim=0).contiguous()
        V_per_img = all_patches[0].shape[0]

        max_patches_in_config = (self.lamda.config.vit_config.image_size // p) ** 2
        raw_ids = torch.arange(V_per_img, device=device, dtype=torch.long)
        safe_ids = raw_ids % max_patches_in_config 
        packed_vit_position_ids = safe_ids.repeat(B).contiguous()

        cu_seqlens = torch.arange(0, (B + 1) * V_per_img, V_per_img, device=device, dtype=torch.int32).contiguous()

        vit_out = self.vit_model.vision_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=V_per_img
        )

        if vit_out.ndim == 2:
            vit_out = vit_out.view(B, V_per_img, -1).contiguous()
        
        return self.connector(vit_out)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_tokens(tokens)


class LaMDAWithExpertModel(nn.Module):
    def __init__(
        self, lamda_model_path: str, action_expert_config,
        use_adarms=None, precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        self.lamda_model = LaMDAPytorch(lamda_model_path, precision=precision)

        # =================================================================
        # 👑 核心黑客科技：动态窃取 LLaDA 的注意力配置，强行覆盖 Gemma 的图纸
        # =================================================================
        llm_cfg = self.lamda_model.language_model.config
        real_n_heads = llm_cfg.n_heads
        real_head_dim = llm_cfg.d_model // real_n_heads
        
        # 兼容你们内部的各种 KV heads 命名 (MQA/GQA)
        if hasattr(llm_cfg, "effective_n_kv_heads"):
            real_kv_heads = llm_cfg.effective_n_kv_heads
        elif hasattr(llm_cfg, "n_kv_heads"):
            real_kv_heads = llm_cfg.n_kv_heads
        else:
            real_kv_heads = real_n_heads

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=real_head_dim,                # 强行对齐 LLaDA 的 128
            hidden_size=llm_cfg.d_model,           # 强行对齐 LLaDA 的 4096
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=real_n_heads,      # 强行对齐 LLaDA 的 32
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=real_kv_heads,     # 强行对齐 LLaDA 的 KV 数量
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=llm_cfg.d_model if use_adarms[1] else None,
        )

        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self._apply_precision(precision)

    def _apply_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "bfloat16": self.to(dtype=torch.bfloat16)
        elif precision == "float32": self.to(dtype=torch.float32)
        else: raise ValueError(f"Unsupported precision: {precision}")

    @property
    def vlm(self): return self.lamda_model

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.lamda_model.embed_image(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lamda_model.embed_language_tokens(tokens)

    def forward(
        self, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, adarms_cond=None,
    ):
        if adarms_cond is None: adarms_cond = [None, None]

        if inputs_embeds[1] is None:
            prefix_output = self._run_language_model(
                inputs_embeds[0], attention_mask, position_ids,
                past_key_values, use_cache, adarms_cond[0],
            )
            return [prefix_output.last_hidden_state, None], prefix_output.past_key_values

        if inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1], attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, adarms_cond=adarms_cond[1],
            )
            return [None, suffix_output.last_hidden_state], None

        return self._joint_forward(inputs_embeds, attention_mask, position_ids, adarms_cond)

    def _run_language_model(self, embeds, mask, pos_ids, past_kv, use_cache, cond):
        lm = self.lamda_model.language_model
        return lm(inputs_embeds=embeds, attention_mask=mask, position_ids=pos_ids,
                  past_key_values=past_kv, use_cache=use_cache, adarms_cond=cond)

    def _joint_forward(self, inputs_embeds, attention_mask, position_ids, adarms_cond):
        lm_backbone = self.lamda_model.language_model.model 
        expert_backbone = self.gemma_expert.model

        # Gemma rotary_emb 需要非 None 的 position_ids；调用方未传时从序列长度构造
        if position_ids is None:
            B, L = inputs_embeds[0].shape[0], inputs_embeds[0].shape[1]
            device = inputs_embeds[0].device
            position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        
        llada_layers = lm_backbone.transformer["blocks"]
        gemma_layers = expert_backbone.layers
        num_layers = min(len(llada_layers), len(gemma_layers))

        use_gradient_checkpointing = (hasattr(expert_backbone, "gradient_checkpointing") and expert_backbone.gradient_checkpointing and self.training)
        if self.training and hasattr(expert_backbone, "gradient_checkpointing"):
            if not expert_backbone.gradient_checkpointing:
                expert_backbone.gradient_checkpointing = True
            use_gradient_checkpointing = True

        def compute_layer(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
            hidden_llada = inputs_embeds[0]
            hidden_gemma = inputs_embeds[1]
            
            layer_llada = llada_layers[layer_idx]
            layer_gemma = gemma_layers[layer_idx]
            b_sz = hidden_gemma.shape[0]

            # ===== 1. Prefix (LLaDA) 的 QKV 提取 =====
            llada_cfg = layer_llada.config
            n_heads_l = llada_cfg.n_heads
            head_dim_l = llada_cfg.d_model // n_heads_l
            n_kv_heads_l = getattr(llada_cfg, "effective_n_kv_heads", getattr(llada_cfg, "n_kv_heads", n_heads_l))

            norm_llada = layer_llada.attn_norm(hidden_llada)
            q_llada = layer_llada.q_proj(norm_llada)
            k_llada = layer_llada.k_proj(norm_llada)
            v_llada = layer_llada.v_proj(norm_llada)
            
            seq_len_llada = hidden_llada.shape[1]
            q_llada = q_llada.view(b_sz, seq_len_llada, n_heads_l, head_dim_l).transpose(1, 2)
            k_llada = k_llada.view(b_sz, seq_len_llada, n_kv_heads_l, head_dim_l).transpose(1, 2)
            v_llada = v_llada.view(b_sz, seq_len_llada, n_kv_heads_l, head_dim_l).transpose(1, 2)
            
            if hasattr(layer_llada, "rotary_emb"):
                q_llada, k_llada = layer_llada.rotary_emb(q_llada, k_llada)

            # ===== 2. Suffix (Gemma Expert) 的 QKV 提取 =====
            n_heads_g = expert_backbone.config.num_attention_heads
            n_kv_heads_g = expert_backbone.config.num_key_value_heads
            head_dim_g = expert_backbone.config.head_dim

            # 👑 兼容官方版 Gemma 与 OpenPI 魔改版
            try:
                norm_gemma, gate_gemma = layer_gemma.input_layernorm(hidden_gemma, cond=adarms_cond[1])
            except TypeError:
                norm_gemma = layer_gemma.input_layernorm(hidden_gemma)
                gate_gemma = None

            q_gemma = layer_gemma.self_attn.q_proj(norm_gemma)
            k_gemma = layer_gemma.self_attn.k_proj(norm_gemma)
            v_gemma = layer_gemma.self_attn.v_proj(norm_gemma)
            
            seq_len_gemma = hidden_gemma.shape[1]
            q_gemma = q_gemma.view(b_sz, seq_len_gemma, n_heads_g, head_dim_g).transpose(1, 2)
            k_gemma = k_gemma.view(b_sz, seq_len_gemma, n_kv_heads_g, head_dim_g).transpose(1, 2)
            v_gemma = v_gemma.view(b_sz, seq_len_gemma, n_kv_heads_g, head_dim_g).transpose(1, 2)
            
            # Suffix (Gemma) 序列长度可能小于 prefix，需用与 suffix 对应的 position_ids 算 RoPE，否则 cos/sin 与 q 的 seq 维不一致
            position_ids_gemma = position_ids[:, -seq_len_gemma:].contiguous()
            dummy = torch.zeros(b_sz, seq_len_gemma, head_dim_g, device=q_gemma.device, dtype=q_gemma.dtype)
            cos, sin = expert_backbone.rotary_emb(dummy, position_ids_gemma)
            q_gemma, k_gemma = modeling_gemma.apply_rotary_pos_emb(q_gemma, k_gemma, cos, sin, unsqueeze_dim=1)

            # ===== 3. 无缝联合注意力计算 =====
            q_joint = torch.cat([q_llada, q_gemma], dim=2)
            k_joint = torch.cat([k_llada, k_gemma], dim=2)
            v_joint = torch.cat([v_llada, v_gemma], dim=2)

            scaling = layer_gemma.self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                layer_gemma.self_attn, q_joint, k_joint, v_joint, attention_mask, scaling
            )
            att_output = att_output.reshape(b_sz, -1, n_heads_g * head_dim_g)

            # ===== 4. 分离并完成各自的 MLP =====
            # -- LLaDA 半段 --
            out_llada = att_output[:, :seq_len_llada]
            if out_llada.dtype != layer_llada.attn_out.weight.dtype:
                out_llada = out_llada.to(layer_llada.attn_out.weight.dtype)
            out_llada = layer_llada.attn_out(out_llada)
            
            out_llada = hidden_llada + out_llada
            after_res1_llada = out_llada.clone()
            
            norm2_llada = layer_llada.ff_norm(out_llada)
            gate_out = layer_llada.act(layer_llada.ff_proj(norm2_llada))
            up_out = layer_llada.up_proj(norm2_llada)
            mlp_llada = layer_llada.ff_out(gate_out * up_out)
            out_llada = after_res1_llada + mlp_llada

            # -- Gemma 半段 (兼容官方) --
            out_gemma = att_output[:, seq_len_llada:]
            if out_gemma.dtype != layer_gemma.self_attn.o_proj.weight.dtype:
                out_gemma = out_gemma.to(layer_gemma.self_attn.o_proj.weight.dtype)
            out_gemma = layer_gemma.self_attn.o_proj(out_gemma)
            
            if gate_gemma is not None:
                out_gemma = modeling_gemma._gated_residual(hidden_gemma, out_gemma, gate_gemma)
            else:
                out_gemma = hidden_gemma + out_gemma  # 官方纯净残差
                
            after_res1_gemma = out_gemma.clone()
            
            try:
                norm2_gemma, gate2_gemma = layer_gemma.post_attention_layernorm(out_gemma, cond=adarms_cond[1])
            except TypeError:
                norm2_gemma = layer_gemma.post_attention_layernorm(out_gemma)
                gate2_gemma = None
                
            if layer_gemma.mlp.up_proj.weight.dtype == torch.bfloat16:
                norm2_gemma = norm2_gemma.to(dtype=torch.bfloat16)
            mlp_gemma = layer_gemma.mlp(norm2_gemma)
            
            if gate2_gemma is not None:
                out_gemma = modeling_gemma._gated_residual(after_res1_gemma, mlp_gemma, gate2_gemma)
            else:
                out_gemma = after_res1_gemma + mlp_gemma

            return [out_llada, out_gemma]

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer, layer_idx, inputs_embeds, attention_mask,
                    position_ids, adarms_cond, use_reentrant=False, preserve_rng_state=False,
                )
            else:
                inputs_embeds = compute_layer(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond)

        if "ln_f" in lm_backbone.transformer:
            out_llada = lm_backbone.transformer["ln_f"](inputs_embeds[0])
        else:
            out_llada = inputs_embeds[0]
            
        # 👑 最后一层兼容
        try:
            out_gemma, _ = expert_backbone.norm(inputs_embeds[1], cond=adarms_cond[1])
        except TypeError:
            out_gemma = expert_backbone.norm(inputs_embeds[1])

        return [out_llada, out_gemma], None