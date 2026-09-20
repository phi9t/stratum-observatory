"""Actual local vision modules with deterministic CPU teaching inputs and weights."""
from dataclasses import replace
from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F
from .config import V41_CONFIG
from .vision_inputs import VisionInputs, vision_positions


def inputs_example() -> dict[str, Any]:
    """Small deterministic CPU trace through actual local modules, with injected weights."""
    from .worked_examples import tensor_row
    config = replace(V41_CONFIG,hidden_dim=3,vision_dim=4,vision_heads=1,
                     vision_hidden_dim=3,vision_layers=1,vision_patch_size=2)
    # Preserve the caller's RNG state. All trainable parameters are then explicitly set.
    with torch.random.fork_rng(), torch.no_grad():
        model = VisionInputs(config)
        for name, parameter in model.named_parameters():
            if 'norm' in name and name.endswith('weight'):
                parameter.fill_(1)
            else:
                parameter.copy_(((torch.arange(parameter.numel()).reshape(parameter.shape)%11)-5)/20)
        # Already-prepared RGB image; extraction alone defines channel/patch ordering.
        pixels = torch.arange(48,dtype=torch.float32).reshape(3,4,4)/48
        patches = pixels.unfold(1,2,2).unfold(2,2,2).permute(1,2,0,3,4).reshape(4,3,2,2)
        height,width = 2,2
        tower,aligner = model.tower,model.aligner
        embedded = tower.patch_embed(patches)
        cos,sin = vision_positions(config,height,width,embedded.device)
        block = tower.blocks[0]
        normalized = block.norm1(embedded)
        attended = block.attention(normalized,cos,sin)
        residual = embedded+attended
        ffn = block.mlp(block.norm2(residual))
        features = tower(patches,height,width)
        grouped = aligner.group_patches(features,height,width)
        projected = aligner.input(grouped)
        def row(identifier: str, title: str, operation: str, tensor: Tensor, axes: list[str]) -> dict[str, Any]:
            return tensor_row(identifier,title,operation,tensor,axes)
        return dict(title='Prepared RGB patches to visual token features',
            scope='CPU teaching fixture with injected weights; not publisher checkpoint weights. Image preprocessing policy and text placeholder insertion are excluded.',
            settings=dict(grid_height=height,grid_width=width,patch_size=config.vision_patch_size,
                          vision_dim=config.vision_dim,heads=config.vision_heads,layers=config.vision_layers,
                          intermediate=config.vision_hidden_dim,downsample_ratio=config.vision_downsample_ratio,
                          text_dim=config.hidden_dim,rope_theta=config.vision_rope_theta),
            inputs=[row('pixels','Prepared image','Illustrative normalized RGB tensor',pixels,['channels','pixel_rows','pixel_columns']),
                    row('patches','Row-major image patches','Extract each channel-first spatial patch',patches,['patches','channels','patch_rows','patch_columns'])],
            steps=[row('patch_embeddings','Patch embeddings','Linear(flatten RGB patch)',embedded,['patches','vision_dim']),
                   row('rope_cos','Grid rotary cosines','cos(height frequencies; width frequencies)',cos[:,0],['patches','half_head_dim']),
                   row('rope_sin','Grid rotary sines','sin(height frequencies; width frequencies)',sin[:,0],['patches','half_head_dim']),
                   row('attention_norm','Attention input','RMSNorm(patch embeddings)',normalized,['patches','vision_dim']),
                   row('attention_output','Bidirectional attention','Output projection of softmax(QKᵀ / √K)V with 2D RoPE',attended,['patches','vision_dim']),
                   row('attention_residual','Attention residual','patch embeddings + attention output',residual,['patches','vision_dim']),
                   row('ffn_output','SwiGLU output','Down(SiLU(gate) × up) after RMSNorm',ffn,['patches','vision_dim']),
                   row('block_output','Block output','attention residual + FFN output',block(embedded,cos,sin),['patches','vision_dim']),
                   row('vision_features','Final normalized tower features','VisionTransformer: final RMSNorm after blocks',features,['patches','vision_dim']),
                   row('grouped_patches','Padded spatial groups','Channel-first concatenation of spatial neighborhoods',grouped,['visual_tokens','grouped_channels']),
                   row('projector_hidden','First projector linear','Linear(grouped patches)',projected,['visual_tokens','text_dim']),
                   row('projector_gelu','Projector activation','GELU(first projector linear)',F.gelu(projected),['visual_tokens','text_dim']),
                   row('visual_tokens','Visual token features','VisionInputs: tower then aligner',model(patches,height,width),['visual_tokens','text_dim'])])
