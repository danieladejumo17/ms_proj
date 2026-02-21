import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# ===========================================================================
# 1. CONFIGURATION (Based on Qwen2.5-VL-7B / Cosmos Reason1)
# ===========================================================================

class CosmosConfig:
    def __init__(self):
        # LLM Config
        self.vocab_size = 152064
        self.hidden_size = 3584
        self.intermediate_size = 18944  # SwiGLU dimension
        self.num_hidden_layers = 28
        self.num_attention_heads = 28
        self.num_key_value_heads = 4    # Grouped Query Attention (GQA)
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-6
        self.rope_theta = 1000000.0
        self.max_position_embeddings = 32768
        
        # Vision Config
        self.vision_hidden_size = 1280
        self.vision_intermediate_size = 3420  # Approx 4 * hidden / 1.5 usually
        self.vision_num_layers = 32
        self.vision_num_heads = 16
        self.patch_size = 14
        self.temporal_patch_size = 2
        self.spatial_merge_size = 2
        self.merge_hidden_size = self.vision_hidden_size * (self.spatial_merge_size ** 2)

# ===========================================================================
# 2. HELPER LAYERS (RMSNorm, SwiGLU, RoPE)
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, mrope_section=None):
    """
    Apply mRoPE (Multimodal RoPE).
    mRoPE splits the head dimension into 3 parts: [time, height, width].
    """
    # q, k: [batch, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] (pre-computed based on position_ids)
    
    # Simple RoPE implementation for demonstration. 
    # In full mRoPE, 'cos' and 'sin' are constructed by concatenating T/H/W frequencies
    # corresponding to the specific mrope_section splits.
    
    # We assume cos/sin are already gathered correctly for the sequence positions
    # and unsqueezed to [1, seq_len, 1, head_dim] or similar.
    
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# ===========================================================================
# 3. VISION ENCODER (ViT with 3D Patching)
# ===========================================================================

class VisionAttention(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.hidden_size = config.vision_hidden_size
        self.num_heads = config.vision_num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard Scaled Dot Product Attention (FlashAttention usually used here)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class VisionMLP(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.act = nn.SiLU() # Or GELU
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class VisionEncoderLayer(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.vision_hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.norm2 = RMSNorm(config.vision_hidden_size, eps=1e-6)
        self.mlp = VisionMLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        # 3D Patch Embedding: (Temporal, Height, Width)
        # Kernel: (2, 14, 14), Stride: (2, 14, 14)
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=config.vision_hidden_size,
            kernel_size=(config.temporal_patch_size, config.patch_size, config.patch_size),
            stride=(config.temporal_patch_size, config.patch_size, config.patch_size),
            bias=False
        )
        
        self.blocks = nn.ModuleList([VisionEncoderLayer(config) for _ in range(config.vision_num_layers)])
        self.rotary_pos_emb = nn.Sequential() # Placeholder for Vision RoPE logic

    def forward(self, video_pixels):
        # video_pixels: [Batch, Channels, Time, Height, Width]
        x = self.patch_embed(video_pixels) # -> [B, Embed, T_patch, H_patch, W_patch]
        B, E, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # -> [B, N_tokens, Embed]
        
        # Apply Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        return x, (T, H, W)

# ===========================================================================
# 4. PROJECTOR (Spatial Merger + MLP)
# ===========================================================================

class Projector(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        # Merges 2x2 spatial neighbors -> dimension increases by 4x
        self.linear = nn.Linear(config.vision_hidden_size * 4, config.hidden_size)
        self.act = nn.SiLU()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, grid_shape):
        # x: [B, T*H*W, Vision_Dim]
        T, H, W = grid_shape
        B, N, C = x.shape
        
        # Reshape to grid
        x = x.view(B, T, H, W, C)
        
        # Patch Merging (2x2 spatial pooling via reshaping)
        # Logic: We take blocks of 2x2 in H,W and concat them in feature dim
        x = x.view(B, T, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, T, H // 2, W // 2, 4 * C)
        
        x = x.flatten(1, 3) # [B, New_Seq_Len, 4*Vision_Dim]
        
        # Projection MLP
        x = self.linear(x)
        x = self.act(x)
        x = self.proj(x)
        return x

# ===========================================================================
# 5. LLM (Qwen2.5 Structure)
# ===========================================================================

class Qwen2MLP(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2Attention(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, freqs_cis=None):
        B, L, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim)

        q = q.transpose(1, 2) # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE (Simplified)
        # In production, apply_rotary_pos_emb would use the 'freqs_cis' (mRoPE frequencies)
        if freqs_cis is not None:
            # Placeholder for complex mRoPE application
            pass

        # GQA: Repeat KV heads
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(attn_output)

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, freqs_cis=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, freqs_cis=freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class CosmosReasonModel(nn.Module):
    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.config = config
        
        # 1. Vision Components
        self.visual = VisionTransformer(config)
        self.projector = Projector(config)
        
        # 2. Language Components
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor, video_pixels: torch.Tensor, video_mask: Optional[torch.Tensor] = None):
        """
        Custom inference forward pass handling multimodal fusion.
        
        Args:
            input_ids: [Batch, Seq_Len] - Text tokens.
            video_pixels: [Batch, Channels, Time, Height, Width] - Raw video frames.
            video_mask: Boolean mask indicating where <video> placeholders are in input_ids.
        """
        
        # 1. Encode Video
        # video_pixels shape: [B, 3, T, H, W]
        vision_features, grid_shape = self.visual(video_pixels)
        
        # 2. Project Video to LLM Space
        # This reduces tokens by 4x via spatial merging
        video_embeds = self.projector(vision_features, grid_shape) # [B, Video_Tokens, LLM_Dim]
        
        # 3. Embed Text
        inputs_embeds = self.embed_tokens(input_ids) # [B, Text_Len, LLM_Dim]
        
        # 4. Multimodal Fusion
        # In a real scenario, you splice 'video_embeds' into 'inputs_embeds' where <video> tokens exist.
        # For simplicity here, we assume the prompt is prepended with video.
        final_embeds = torch.cat([video_embeds, inputs_embeds], dim=1)
        
        # 5. LLM Forward
        hidden_states = final_embeds
        for layer in self.layers:
            # Note: mRoPE frequencies calculation would happen here based on T,H,W
            hidden_states = layer(hidden_states)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

# ===========================================================================
# 6. WEIGHT LOADING UTILITY
# ===========================================================================

def load_weights(model: CosmosReasonModel, weight_path: str):
    """
    Loads weights from the official safetensors files.
    Note: You must map official key names to this custom model's names.
    """
    from safetensors.torch import load_file
    
    print(f"Loading weights from {weight_path}...")
    state_dict = load_file(weight_path)
    
    # Example Mapping (This needs to be adjusted based on exact inspection of the checkpoint)
    # Official keys often look like: "model.layers.0.self_attn.q_proj.weight"
    # Our keys look like: "layers.0.self_attn.q_proj.weight"
    
    new_state_dict = {}
    for key, value in state_dict.items():
        # Clean prefix "model." if it exists in official weights but not in ours
        # (Qwen usually has 'model.' prefix for the LLM part)
        new_key = key
        
        if key.startswith("model.embed_tokens"): new_key = "embed_tokens" + key[18:]
        elif key.startswith("model.layers"): new_key = "layers" + key[12:]
        elif key.startswith("model.norm"): new_key = "norm" + key[10:]
        elif key.startswith("visual."): new_key = "visual." + key[7:] # Our visual encoder name matches
        
        # Handle specific layer naming mismatches
        # e.g. Qwen might use 'mlp.gate_up_proj' (merged) vs our split gate/up
        # This requires manual splitting if the weights are merged.
        
        new_state_dict[new_key] = value

    # Strict=False allows partial loading if we miss some buffers, 
    # but we should aim for matching keys.
    model.load_state_dict(new_state_dict, strict=False)
    print("Weights loaded successfully.")

# ===========================================================================
# 7. INFERENCE SCRIPT
# ===========================================================================

def run_inference():
    # 1. Initialize Model
    config = CosmosConfig()
    model = CosmosReasonModel(config).cuda().bfloat16()
    model.eval()
    
    # 2. Load Weights (Dummy path here)
    # load_weights(model, "path/to/cosmos-reason1-7b/model.safetensors")
    
    # 3. Prepare Dummy Inputs (Since we don't have the tokenizer loaded)
    # Real inputs: Video (Time, H, W) + Text Prompt
    
    # Dummy Video: 8 frames, 224x224
    T, H, W = 8, 224, 224
    video_input = torch.randn(1, 3, T, H, W).cuda().bfloat16()
    
    # Dummy Text: "Describe this video." (Tokenized IDs)
    text_input = torch.randint(0, config.vocab_size, (1, 10)).cuda()
    
    # 4. Inference
    with torch.no_grad():
        print("Running forward pass...")
        logits = model(text_input, video_input)
        
        # Simple Greedy Decoding for next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        print(f"Next token ID: {next_token.item()}")
        
if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"Note: This script defines the architecture. To run, download weights and adjust paths.\nError: {e}")


# No Toeknnizer
# Simplified sampling of the LM head? Should that be? What else was simplified????