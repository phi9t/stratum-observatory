/** Educational PyTorch adaptations of DeepSeek-AI's MIT-licensed publisher reference. */
export const BACKBONE_EXAMPLES=[{
 id:'backbone-mhc',title:'Single-pass mHC and the flat encoder/decoder stack',stages:['encoder','decoder'],
 sources:['dsv41-code','dsv41-kernel'],
 symbols:['RMSNorm','Block.hc_mixes','Block.hc_pre','Block.hc_post','Block.forward','make_identity_pre_mix','Transformer.forward','hc_split_sinkhorn','hc_split_sinkhorn_kernel'],
 summary:'Track residual copies and previous-sublayer premix explicitly. The encoder boundary is a point in a flat stack, with no inserted cross-attention.',
 limitations:'Educational adaptation of the DeepSeek-AI publisher reference (MIT). Small random weights, CPU float arithmetic; normalization epsilons are explicit caller inputs; no checkpoint loading, fused kernels, quantization, distribution, vision, Engram insertion or vocabulary head. Attention and FFN are injected operators: this demonstrates orchestration, not a complete V4.1 model or the bounded-replay serving optimization. The boundary residual is captured for inspection; actual global-cache ownership lives inside the injected attention implementation.',
 requires:[],code:`import torch
from torch import nn
from torch.nn import functional as F


class BB_RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


def bb_split_sinkhorn(mixes, scale, base, streams, eps, iterations=20):
    # Readable transcription of the publisher kernel; retain epsilon placement.
    if streams < 1 or iterations < 1 or eps <= 0:
        raise ValueError("positive streams, iterations and epsilon required")
    size = streams * (streams + 2)
    if mixes.shape[-1] != size or base.shape != (size,) or scale.shape != (3,):
        raise ValueError("coefficient dimensions disagree")
    pre = torch.sigmoid(mixes[..., :streams] * scale[0] + base[:streams]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., streams:2*streams] * scale[1] + base[streams:2*streams])
    logits = mixes[..., 2*streams:] * scale[2] + base[2*streams:]
    comb = logits.unflatten(-1, (streams, streams)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def bb_pre(residual, premix):
    # [B,T,input-copy,D] -> [B,T,D]
    return (premix.unsqueeze(-1) * residual.float()).sum(-2).to(residual.dtype)


def bb_post(output, residual, post, comb):
    # comb[..., input-copy, output-copy]; do not transpose its meaning.
    carried = torch.einsum("btij,btid->btjd", comb.float(), residual.float())
    return (post.unsqueeze(-1) * output.unsqueeze(-2) + carried).to(output.dtype)


class BB_Mixer(nn.Module):
    def __init__(self, width, streams, norm_eps, hc_eps):
        super().__init__()
        size = streams * (streams + 2)
        self.projection = nn.Linear(width * streams, size, bias=False)
        self.scale = nn.Parameter(torch.ones(3))
        self.base = nn.Parameter(torch.zeros(size))
        self.streams, self.norm_eps, self.hc_eps = streams, norm_eps, hc_eps

    def forward(self, residual):
        flat = residual.flatten(-2).float()
        # One RMS statistic across ALL copies and width, not one per copy.
        inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.projection.weight.float()) * inverse_rms
        return bb_split_sinkhorn(mixes, self.scale, self.base,
                                 self.streams, eps=self.hc_eps)


class BB_Block(nn.Module):
    def __init__(self, width, streams, attention, ffn, norm_eps, hc_eps):
        super().__init__()
        self.attention, self.ffn = attention, ffn
        self.attn_norm, self.ffn_norm = BB_RMSNorm(width, norm_eps), BB_RMSNorm(width, norm_eps)
        self.attn_mix, self.ffn_mix = BB_Mixer(width, streams, norm_eps, hc_eps), BB_Mixer(width, streams, norm_eps, hc_eps)

    def forward(self, residual, previous_pre):
        attn_pre, attn_post, attn_comb = self.attn_mix(residual)
        # Attention consumes the PREVIOUS block's FFN premix (initially one-hot).
        attention_input = self.attn_norm(bb_pre(residual, previous_pre))
        residual = bb_post(self.attention(attention_input), residual, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.ffn_mix(residual)
        # The current attention's predicted premix is used only now, by FFN.
        ffn_input = self.ffn_norm(bb_pre(residual, attn_pre))
        residual = bb_post(self.ffn(ffn_input), residual, ffn_post, ffn_comb)
        return residual, ffn_pre


class BB_FlatCED(nn.Module):
    def __init__(self, blocks, encoder_layers, width, streams, norm_eps):
        super().__init__()
        if not 0 < encoder_layers < len(blocks) or streams < 1:
            raise ValueError("require nonempty encoder and decoder with positive streams")
        self.blocks = nn.ModuleList(blocks)
        self.encoder_layers, self.streams = encoder_layers, streams
        self.norm = BB_RMSNorm(width, norm_eps)

    def forward(self, embedded):
        # Caller supplies [B,T,D] embeddings and causal attention implementations.
        residual = embedded.unsqueeze(-2).repeat(1, 1, self.streams, 1)
        premix = embedded.new_zeros(*embedded.shape[:2], self.streams, dtype=torch.float32)
        premix[..., 0] = 1  # identity initialization is one-hot, not uniform averaging
        boundary = None
        for index, block in enumerate(self.blocks):
            residual, premix = block(residual, premix)
            if index + 1 == self.encoder_layers:
                boundary = residual.clone()  # observation; no extra cross-attention
        # The last FFN's premix collapses the residual before final normalization.
        return self.norm(bb_pre(residual, premix)), boundary
`
},{
 id:'backbone-moe',title:'Routed experts plus a separate shared expert',stages:['encoder','decoder'],
 sources:['dsv41-code'],symbols:['Gate.forward','Expert.forward','MoE.forward'],
 summary:'Correction bias chooses routed experts; unbiased scores weight their outputs. The shared expert runs independently of routed top-k.',
 limitations:'Educational adaptation of the DeepSeek-AI publisher reference (MIT). Text-only sqrt-softplus routing, normalized routed top-k greater than one, and one shared expert. Dimensions, normalization temperature, routing scale and positive SwiGLU clamp are explicit caller inputs; weights are random. Omits image-specific routing bias, quantization, expert parallelism, checkpoint loading and distributed reductions. This is a local mathematical operator, not a serving benchmark.',
 requires:[],code:`import torch
from torch import nn
from torch.nn import functional as F


class BB_Expert(nn.Module):
    def __init__(self, width, hidden, limit):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        if limit <= 0:
            raise ValueError("positive SwiGLU clamp required")
        self.limit = limit

    def forward(self, x, weights):
        gate, up = self.gate(x).float(), self.up(x).float()
        gate = gate.clamp(max=self.limit)
        up = up.clamp(-self.limit, self.limit)
        activated = F.silu(gate) * up
        activated = activated * weights
        return self.down(activated.to(x.dtype))


class BB_MoE(nn.Module):
    def __init__(self, width, hidden, experts, topk, temperature, route_scale, limit):
        super().__init__()
        if not 1 < topk <= experts or temperature <= 0:
            raise ValueError("invalid routed expert count, top-k or temperature")
        self.router = nn.Linear(width, experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(experts))
        self.experts = nn.ModuleList([BB_Expert(width, hidden, limit) for _ in range(experts)])
        self.shared = BB_Expert(width, hidden, limit)
        self.topk, self.temperature = topk, temperature
        self.route_scale = route_scale

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = F.softplus(F.linear(flat.float(), self.router.weight.float())
                            / self.temperature).sqrt()
        selected = (scores + self.bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(-1, selected)  # correction bias NEVER enters weights
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        routed = torch.zeros_like(flat, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token, slot = torch.where(selected == expert_id)
            # As in publisher Expert.forward, apply route weights before down projection.
            contribution = expert(flat[token], weights[token, slot, None]).float()
            routed.index_add_(0, token, contribution)
        return (routed + self.shared(flat, 1.0).float()).to(x.dtype).reshape(shape)
`
}];
