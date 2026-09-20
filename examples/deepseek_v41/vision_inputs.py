"""V4.1 image features: channel-first patches → RoPE ViT → spatial aligner.

Readable CPU teaching implementation of dsv41-vision at revision
 dba1be0a40aa45a94ad051997016db3960a90277. This is one already-prepared image,
not a preprocessing/tokenizer policy, checkpoint loader or serving program.
Image resize/crop/normalization, multiple images and placeholder insertion are
outside this excerpt. There is no learned positional table, convolutional patch
module, CLS token or average pooling in this published path.

Axes: N = grid_height × grid_width patches, C = RGB channels, P = patch size,
D = vision width, A = attention heads, K = D/A, R = spatial grouping ratio,
M = ceil(grid_height/R) × ceil(grid_width/R) visual tokens, T = text width.
"""
from typing import Annotated
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .config import V41Config

PatchTensor = Annotated[Tensor, "patches, RGB channels, patch rows, patch columns"]
FeatureTensor = Annotated[Tensor, "patches or visual tokens, feature width"]
RotaryTensor = Annotated[Tensor, "patches, singleton head, half head width"]
HeadTensor = Annotated[Tensor, "patches, attention heads, head width"]


class VisionRMSNorm(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.epsilon = config.vision_norm_epsilon
        self.weight = nn.Parameter(torch.ones(config.vision_dim))

    def forward(self, x: FeatureTensor) -> FeatureTensor:
        # [N,D] → [N,D]; accumulate mean square and learned scale in float32.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.epsilon)
        return (normalized * self.weight).to(x.dtype)


def vision_positions(config: V41Config, height: int, width: int, device: torch.device) -> tuple[RotaryTensor, RotaryTensor]:
    """[N,1,K/2] cos/sin: height frequencies followed by width frequencies."""
    half = config.vision_dim // config.vision_heads // 2
    inverse = config.vision_rope_theta ** (-torch.arange(0, half, 2, device=device, dtype=torch.float32) / half)
    positions = torch.stack(torch.meshgrid(torch.arange(height, device=device),
                                         torch.arange(width, device=device), indexing='ij'), dim=-1)
    angles = (positions.reshape(-1, 2, 1).float() * inverse).flatten(1).unsqueeze(1)
    return angles.cos(), angles.sin()


def vision_rotate(x: HeadTensor, cos: RotaryTensor, sin: RotaryTensor) -> HeadTensor:
    # [N,A,K]: rotate the two feature halves; frequencies encode both grid axes.
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first*cos-second*sin, second*cos+first*sin), dim=-1).to(x.dtype)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.projection = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, patches: PatchTensor) -> FeatureTensor:
        # [N,C,P,P] → [N,C×P×P] → [N,D]; each RGB channel's pixels stay together.
        return self.projection(patches.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.vision_dim, 3*config.vision_dim)
        self.output = nn.Linear(config.vision_dim, config.vision_dim)

    def forward(self, x: FeatureTensor, cos: RotaryTensor, sin: RotaryTensor) -> FeatureTensor:
        # [N,D] → three [N,A,K] tensors; RoPE affects Q and K, never V.
        heads = self.config.vision_heads
        q, k, v = [part.reshape(x.shape[0], heads, -1) for part in self.qkv(x).chunk(3,-1)]
        q, k = vision_rotate(q,cos,sin), vision_rotate(k,cos,sin)
        # [A,N,K]: every patch attends to every patch; no causal mask or dropout.
        context = F.scaled_dot_product_attention(q.transpose(0,1),k.transpose(0,1),v.transpose(0,1))
        return self.output(context.transpose(0,1).reshape(x.shape[0],-1))


class VisionMLP(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.gate_up = nn.Linear(config.vision_dim, 2*config.vision_hidden_dim, bias=False)
        self.down = nn.Linear(config.vision_hidden_dim, config.vision_dim, bias=False)

    def forward(self, x: FeatureTensor) -> FeatureTensor:
        # [N,D] → paired [N,intermediate] → [N,D], bias-free SwiGLU.
        gate, up = self.gate_up(x).chunk(2,-1)
        return self.down(F.silu(gate)*up)


class VisionBlock(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.norm1 = VisionRMSNorm(config)
        self.attention = VisionAttention(config)
        self.norm2 = VisionRMSNorm(config)
        self.mlp = VisionMLP(config)

    def forward(self, x: FeatureTensor, cos: RotaryTensor, sin: RotaryTensor) -> FeatureTensor:
        # Both residuals retain [N,D]; the FFN reads the updated attention residual.
        after_attention = x + self.attention(self.norm1(x),cos,sin)
        return after_attention + self.mlp(self.norm2(after_attention))


class VisionTransformer(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.vision_layers)])
        self.norm = VisionRMSNorm(config)

    def forward(self, patches: PatchTensor, height: int, width: int) -> FeatureTensor:
        # One image in row-major patch order; grid sizes describe patches, not pixels.
        if height <= 0 or width <= 0 or patches.shape[0] != height*width:
            raise ValueError('Patch count must equal the positive grid height × width')
        x = self.patch_embed(patches)
        cos,sin = vision_positions(self.config,height,width,x.device)
        for block in self.blocks:
            x = block(x,cos,sin)
        return self.norm(x)  # [N,D], including the final tower RMSNorm.


class VisionAligner(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.ratio = config.vision_downsample_ratio
        self.input = nn.Linear(config.vision_dim*self.ratio**2,config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim,config.hidden_dim)

    def group_patches(self, x: FeatureTensor, height: int, width: int) -> FeatureTensor:
        # [N,D] → [1,D,H,W]; pad bottom/right so complete R×R groups fit.
        grid = x.reshape(height,width,-1).permute(2,0,1).unsqueeze(0)
        padded = F.pad(grid,(0,-width%self.ratio,0,-height%self.ratio))
        # [M,D×R×R]: channel first, then local row and column. This is concatenation.
        return F.unfold(padded,self.ratio,stride=self.ratio).squeeze(0).transpose(0,1)

    def forward(self, x: FeatureTensor, height: int, width: int) -> FeatureTensor:
        # [M,D×R×R] → [M,T] → exact GELU → [M,T], with both linear biases.
        grouped = self.group_patches(x,height,width)
        return self.output(F.gelu(self.input(grouped)))


class VisionInputs(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.tower = VisionTransformer(config)
        self.aligner = VisionAligner(config)

    def forward(self, patches: PatchTensor, height: int, width: int) -> FeatureTensor:
        # [N,3,P,P] → [M,T]; these features can replace image placeholder embeddings.
        return self.aligner(self.tower(patches,height,width),height,width)

