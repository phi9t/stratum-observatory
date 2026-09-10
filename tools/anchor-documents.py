"""Propose text_quote anchors for paper and vendor-document citations.

Quotes are authored by hand in QUOTES, because a quote is an assertion about what a document
says. This tool's job is to prove each authored quote resolves uniquely against the cached
bytes; one that does not is reported as residue and never written.

Every entry below was authored against bytes that were actually read (see
.superpowers/sdd/2026-09-09-evidence-anchors-and-dependency-provenance/task-3-report.md for the
full table and the per-entry justification). Two collision hazards shaped what is and is not
here, because QUOTES is keyed only by (source, field), not by (source, model, field):

1. RULING (L/N), carried from Task 2: a field whose value asserts the ABSENCE of a mechanism
   (numeric 0, empty list, false, or a "no special mechanism" sentinel like residual:'standard')
   never gets an anchor. A paper does not document an absence by failing to mention it.
2. Shared-source collision: several vendor documents and the Qwen3 report are cited as evidence
   for MULTIPLE different models/checkpoints at the SAME field name (e.g. blackwell-tuning is
   evidence for both b200 and gb300; q3-paper is evidence for q38, q330 and q3235). Because one
   QUOTES entry is applied to every claim that shares its (source, field) key regardless of that
   claim's own expected value, an entry is only added for such a shared key when the underlying
   fact and value are genuinely identical across every model that cites it (e.g. q3-paper's GQA
   and QK-Norm sentences apply equally to q38/q330/q3235). Where the value differs per model
   (layers, d, heads, expertFF, ...) the shared key is left entirely unmapped -- residue -- rather
   than risk one quote silently backing a different model's wrong number. blackwell-tuning is
   dropped entirely for the same reason: its clean sentences name "B200" specifically, and using
   them for gb300 (Blackwell Ultra / B300) would misattribute the variant.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.evidence import loads, safe_path, select_anchor

# (source id, claim field) -> (exact quote, required semantic tokens)
# Start empty: every row is authored against bytes that were read first.
QUOTES: dict[tuple[str, str], tuple[str, list[str]]] = {}

QUOTES[('b200', 'arch')] = ('Equipped with eight NVIDIA Blackwell GPUs interconnected with fifth-generation', ['Blackwell'])
QUOTES[('b200', 'gpus')] = ('Equipped with eight NVIDIA Blackwell GPUs interconnected with fifth-generation', ['eight'])
QUOTES[('b200', 'nvGen')] = ('Equipped with eight NVIDIA Blackwell GPUs interconnected with fifth-generation', ['fifth-generation'])
QUOTES[('b200-system', 'cpu')] = ('2 x Intel Xeon 8570 PCIe Gen5 CPUs with 56 cores each 2.1/4 GHz (Base/Max boost)', ['Xeon 8570'])
QUOTES[('b200-system', 'cpus')] = ('2 x Intel Xeon 8570 PCIe Gen5 CPUs with 56 cores each 2.1/4 GHz (Base/Max boost)', ['2 x Intel Xeon'])
QUOTES[('b200-system', 'gpus')] = ('8 x NVIDIA B200 GPUs that provide 1,440 GB total GPU memory', ['8 x NVIDIA B200'])
QUOTES[('b200-system', 'nic')] = ('8 x NVIDIA® ConnectX®-7 Single Port Cards', ['ConnectX'])
QUOTES[('b200-system', 'nicGbps')] = ('InfiniBand (default): Up to 400Gbps', ['400Gbps'])
QUOTES[('b200-system', 'nvGen')] = ('2 x 5th generation NVLink switches that provide 14.4 TB/s aggregate bandwidth', ['5th generation'])
QUOTES[('b200-system', 'switches')] = ('2 x 5th generation NVLink switches that provide 14.4 TB/s aggregate bandwidth', ['2 x'])
QUOTES[('ds2-paper', 'active')] = ('It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens.', ['21B'])
QUOTES[('ds2-paper', 'architecture')] = ('DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE.', ['Multi-head Latent Attention'])
QUOTES[('ds2-paper', 'configMax')] = ('and the target maximum context length to 160K.', ['160K'])
QUOTES[('ds2-paper', 'd')] = ('We set the number of Transformer layers to 60 and the hidden dimension to 5120.', ['5120'])
QUOTES[('ds2-paper', 'denseLayers')] = ('we substitute all FFNs except for the first layer with MoE layers', ['first layer'])
QUOTES[('ds2-paper', 'expertFF')] = ('Each MoE layer consists of 2 shared experts and 160 routed experts, where the intermediate hidden dimension of each expert is 1536.\nAmong the routed experts, 6 experts will be activated for each token.', ['1536'])
QUOTES[('ds2-paper', 'experts')] = ('Each MoE layer consists of 2 shared experts and 160 routed experts, where the intermediate hidden dimension of each expert is 1536.\nAmong the routed experts, 6 experts will be activated for each token.', ['160 routed experts'])
QUOTES[('ds2-paper', 'heads')] = ('In MLA, we set the number of attention heads <math id="S3.SS1.SSS2.Px1.p1.m1" class="ltx_Math" alttext="n_{h}" display="inline" intent=":literal"><semantics><msub><mi>n</mi><mi>h</mi></msub><annotation encoding="application/x-tex">n_{h}</annotation></semantics></math> to 128 and the per-head dimension <math id="S3.SS1.SSS2.Px1.p1.m2" class="ltx_Math" alttext="d_{h}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>h</mi></msub><annotation encoding="application/x-tex">d_{h}</annotation></semantics></math> to 128.', ['128'])
QUOTES[('ds2-paper', 'kvRank')] = ('The KV compression dimension <math id="S3.SS1.SSS2.Px1.p1.m3" class="ltx_Math" alttext="d_{c}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>c</mi></msub><annotation encoding="application/x-tex">d_{c}</annotation></semantics></math> is set to 512, and the query compression dimension <math id="S3.SS1.SSS2.Px1.p1.m4" class="ltx_Math" alttext="d_{c}^{\\prime}" display="inline" intent=":literal"><semantics><msubsup><mi>d</mi><mi>c</mi><mo>′</mo></msubsup><annotation encoding="application/x-tex">d_{c}^{\\prime}</annotation></semantics></math> is set to 1536.', ['512'])
QUOTES[('ds2-paper', 'layers')] = ('We set the number of Transformer layers to 60 and the hidden dimension to 5120.', ['60'])
QUOTES[('ds2-paper', 'maxContext')] = ('It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens.', ['128K'])
QUOTES[('ds2-paper', 'nopeDim')] = ('In MLA, we set the number of attention heads <math id="S3.SS1.SSS2.Px1.p1.m1" class="ltx_Math" alttext="n_{h}" display="inline" intent=":literal"><semantics><msub><mi>n</mi><mi>h</mi></msub><annotation encoding="application/x-tex">n_{h}</annotation></semantics></math> to 128 and the per-head dimension <math id="S3.SS1.SSS2.Px1.p1.m2" class="ltx_Math" alttext="d_{h}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>h</mi></msub><annotation encoding="application/x-tex">d_{h}</annotation></semantics></math> to 128.', ['128'])
QUOTES[('ds2-paper', 'qRank')] = ('The KV compression dimension <math id="S3.SS1.SSS2.Px1.p1.m3" class="ltx_Math" alttext="d_{c}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>c</mi></msub><annotation encoding="application/x-tex">d_{c}</annotation></semantics></math> is set to 512, and the query compression dimension <math id="S3.SS1.SSS2.Px1.p1.m4" class="ltx_Math" alttext="d_{c}^{\\prime}" display="inline" intent=":literal"><semantics><msubsup><mi>d</mi><mi>c</mi><mo>′</mo></msubsup><annotation encoding="application/x-tex">d_{c}^{\\prime}</annotation></semantics></math> is set to 1536.', ['1536'])
QUOTES[('ds2-paper', 'ropeDim')] = ('For the decoupled queries and key, we set the per-head dimension <math id="S3.SS1.SSS2.Px1.p1.m5" class="ltx_Math" alttext="d_{h}^{R}" display="inline" intent=":literal"><semantics><msubsup><mi>d</mi><mi>h</mi><mi>R</mi></msubsup><annotation encoding="application/x-tex">d_{h}^{R}</annotation></semantics></math> to 64.', ['64'])
QUOTES[('ds2-paper', 'shared')] = ('Each MoE layer consists of 2 shared experts and 160 routed experts, where the intermediate hidden dimension of each expert is 1536.\nAmong the routed experts, 6 experts will be activated for each token.', ['2 shared experts'])
QUOTES[('ds2-paper', 'topK')] = ('Each MoE layer consists of 2 shared experts and 160 routed experts, where the intermediate hidden dimension of each expert is 1536.\nAmong the routed experts, 6 experts will be activated for each token.', ['6 experts'])
QUOTES[('ds2-paper', 'total')] = ('It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens.', ['236B'])
QUOTES[('ds2-paper', 'vocab')] = ('which is built based on the Byte-level Byte-Pair Encoding (BBPE) algorithm and has a vocabulary size of 100K.', ['100K'])
QUOTES[('ds3-paper', 'active')] = ('Under this configuration, DeepSeek-V3 comprises 671B total parameters, of which 37B are activated for each token.', ['37B'])
QUOTES[('ds3-paper', 'architecture')] = ('To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE', ['Multi-head Latent Attention'])
QUOTES[('ds3-paper', 'd')] = ('We set the number of Transformer layers to 61 and the hidden dimension to 7168.', ['7168'])
QUOTES[('ds3-paper', 'denseLayers')] = ('We substitute all FFNs except for the first three layers with MoE layers.', ['first three layers'])
QUOTES[('ds3-paper', 'expertFF')] = ('Each MoE layer consists of 1 shared expert and 256 routed experts, where the intermediate hidden dimension of each expert is 2048.\nAmong the routed experts, 8 experts will be activated for each token, and each token will be ensured to be sent to at most 4 nodes.', ['2048'])
QUOTES[('ds3-paper', 'experts')] = ('Each MoE layer consists of 1 shared expert and 256 routed experts, where the intermediate hidden dimension of each expert is 2048.\nAmong the routed experts, 8 experts will be activated for each token, and each token will be ensured to be sent to at most 4 nodes.', ['256 routed experts'])
QUOTES[('ds3-paper', 'heads')] = ('In MLA, we set the number of attention heads <math id="S4.SS2.SSS0.Px1.p1.m1" class="ltx_Math" alttext="n_{h}" display="inline" intent=":literal"><semantics><msub><mi>n</mi><mi>h</mi></msub><annotation encoding="application/x-tex">n_{h}</annotation></semantics></math> to 128 and the per-head dimension <math id="S4.SS2.SSS0.Px1.p1.m2" class="ltx_Math" alttext="d_{h}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>h</mi></msub><annotation encoding="application/x-tex">d_{h}</annotation></semantics></math> to 128.', ['128'])
QUOTES[('ds3-paper', 'kvRank')] = ('The KV compression dimension <math id="S4.SS2.SSS0.Px1.p1.m3" class="ltx_Math" alttext="d_{c}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>c</mi></msub><annotation encoding="application/x-tex">d_{c}</annotation></semantics></math> is set to 512, and the query compression dimension <math id="S4.SS2.SSS0.Px1.p1.m4" class="ltx_Math" alttext="d_{c}^{\\prime}" display="inline" intent=":literal"><semantics><msubsup><mi>d</mi><mi>c</mi><mo>′</mo></msubsup><annotation encoding="application/x-tex">d_{c}^{\\prime}</annotation></semantics></math> is set to 1536.', ['512'])
QUOTES[('ds3-paper', 'layers')] = ('We set the number of Transformer layers to 61 and the hidden dimension to 7168.', ['61'])
QUOTES[('ds3-paper', 'maxContext')] = ('Through this two-phase extension training, DeepSeek-V3 is capable of handling inputs up to 128K in length while maintaining strong performance.', ['128K'])
QUOTES[('ds3-paper', 'mtp')] = ('The multi-token prediction depth <math id="S4.SS2.SSS0.Px1.p1.m6" class="ltx_Math" alttext="D" display="inline" intent=":literal"><semantics><mi>D</mi><annotation encoding="application/x-tex">D</annotation></semantics></math> is set to 1, i.e., besides the exact next token, each token will predict one additional token.', ['predict one additional token'])
QUOTES[('ds3-paper', 'nopeDim')] = ('In MLA, we set the number of attention heads <math id="S4.SS2.SSS0.Px1.p1.m1" class="ltx_Math" alttext="n_{h}" display="inline" intent=":literal"><semantics><msub><mi>n</mi><mi>h</mi></msub><annotation encoding="application/x-tex">n_{h}</annotation></semantics></math> to 128 and the per-head dimension <math id="S4.SS2.SSS0.Px1.p1.m2" class="ltx_Math" alttext="d_{h}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>h</mi></msub><annotation encoding="application/x-tex">d_{h}</annotation></semantics></math> to 128.', ['128'])
QUOTES[('ds3-paper', 'qRank')] = ('The KV compression dimension <math id="S4.SS2.SSS0.Px1.p1.m3" class="ltx_Math" alttext="d_{c}" display="inline" intent=":literal"><semantics><msub><mi>d</mi><mi>c</mi></msub><annotation encoding="application/x-tex">d_{c}</annotation></semantics></math> is set to 512, and the query compression dimension <math id="S4.SS2.SSS0.Px1.p1.m4" class="ltx_Math" alttext="d_{c}^{\\prime}" display="inline" intent=":literal"><semantics><msubsup><mi>d</mi><mi>c</mi><mo>′</mo></msubsup><annotation encoding="application/x-tex">d_{c}^{\\prime}</annotation></semantics></math> is set to 1536.', ['1536'])
QUOTES[('ds3-paper', 'ropeDim')] = ('For the decoupled queries and key, we set the per-head dimension <math id="S4.SS2.SSS0.Px1.p1.m5" class="ltx_Math" alttext="d_{h}^{R}" display="inline" intent=":literal"><semantics><msubsup><mi>d</mi><mi>h</mi><mi>R</mi></msubsup><annotation encoding="application/x-tex">d_{h}^{R}</annotation></semantics></math> to 64.', ['64'])
QUOTES[('ds3-paper', 'scoring')] = ('Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values.', ['sigmoid'])
QUOTES[('ds3-paper', 'shared')] = ('Each MoE layer consists of 1 shared expert and 256 routed experts, where the intermediate hidden dimension of each expert is 2048.\nAmong the routed experts, 8 experts will be activated for each token, and each token will be ensured to be sent to at most 4 nodes.', ['1 shared expert'])
QUOTES[('ds3-paper', 'topK')] = ('Each MoE layer consists of 1 shared expert and 256 routed experts, where the intermediate hidden dimension of each expert is 2048.\nAmong the routed experts, 8 experts will be activated for each token, and each token will be ensured to be sent to at most 4 nodes.', ['8 experts'])
QUOTES[('ds3-paper', 'total')] = ('Under this configuration, DeepSeek-V3 comprises 671B total parameters, of which 37B are activated for each token.', ['671B'])
QUOTES[('gb300', 'arch')] = ('Configuration</td> \n    <td>72 NVIDIA Blackwell Ultra GPUs, 36 NVIDIA Grace CPUs</td>', ['Blackwell Ultra'])
QUOTES[('gb300', 'cpu')] = ('Configuration</td> \n    <td>72 NVIDIA Blackwell Ultra GPUs, 36 NVIDIA Grace CPUs</td>', ['Grace'])
QUOTES[('gb300', 'cpus')] = ('Configuration</td> \n    <td>72 NVIDIA Blackwell Ultra GPUs, 36 NVIDIA Grace CPUs</td>', ['36'])
QUOTES[('gb300', 'gpus')] = ('Configuration</td> \n    <td>72 NVIDIA Blackwell Ultra GPUs, 36 NVIDIA Grace CPUs</td>', ['72'])
QUOTES[('gb300', 'nic')] = ('input/output (IO) module hosts two ConnectX-8 devices, providing 800 gigabits per second (Gb/s) of network connectivity for each GPU in the NVIDIA GB300 NVL72.', ['ConnectX-8'])
QUOTES[('gb300', 'nicGbps')] = ('input/output (IO) module hosts two ConnectX-8 devices, providing 800 gigabits per second (Gb/s) of network connectivity for each GPU in the NVIDIA GB300 NVL72.', ['800 gigabits per second'])
QUOTES[('h100', 'hbmGB')] = ('<td> GPU Memory </td> \n    <td> 80GB </td>', ['80GB'])
QUOTES[('h100', 'hbmTBps')] = ('<td> GPU Memory Bandwidth </td> \n    <td> 3.35TB/s </td>', ['3.35TB/s'])
QUOTES[('h100', 'name')] = ('H100 SXM', ['H100 SXM'])
QUOTES[('h100', 'nvGBps')] = ('NVIDIA NVLink™: 900GB/s <br> PCIe Gen5: 128GB/s', ['900GB/s'])
QUOTES[('h200', 'hbmGB')] = ('<td>GPU Memory</td> \n    <td>141GB</td> \n   </tr> \n   <tr> \n    <td>GPU Memory Bandwidth</td> \n    <td>4.8TB/s</td> \n   </tr>', ['141GB'])
QUOTES[('h200', 'hbmTBps')] = ('<td>GPU Memory</td> \n    <td>141GB</td> \n   </tr> \n   <tr> \n    <td>GPU Memory Bandwidth</td> \n    <td>4.8TB/s</td> \n   </tr>', ['4.8TB/s'])
QUOTES[('h200', 'nvGBps')] = ('NVIDIA NVLink™: 900GB/s<br>PCIe Gen5: 128GB/s', ['900GB/s'])
QUOTES[('hopper-system', 'cpu')] = ('2 x Intel Xeon 8480C PCIe Gen5 CPUs with 56 cores each 2.0/2.9/3.8 GHz (base/all core turbo/Max turbo)', ['Xeon'])
QUOTES[('hopper-system', 'cpus')] = ('2 x Intel Xeon 8480C PCIe Gen5 CPUs with 56 cores each 2.0/2.9/3.8 GHz (base/all core turbo/Max turbo)', ['2 x Intel Xeon'])
QUOTES[('hopper-system', 'nic')] = ('4 x OSFP ports for 8 x NVIDIA® ConnectX®-7 Single Port InfiniBand Cards', ['ConnectX'])
QUOTES[('hopper-system', 'nicGbps')] = ('InfiniBand (default): Up to 400Gbps', ['400Gbps'])
QUOTES[('hopper-system', 'nvGBps')] = ('4 x 4th generation NVLinks that provide 900 GB/s GPU-to-GPU bandwidth', ['900 GB/s'])
QUOTES[('hopper-system', 'nvGen')] = ('4 x 4th generation NVLinks that provide 900 GB/s GPU-to-GPU bandwidth', ['4th generation'])
QUOTES[('hopper-system', 'switches')] = ('4 x 4th generation NVLinks that provide 900 GB/s GPU-to-GPU bandwidth', ['4 x'])
QUOTES[('k3-paper', 'active')] = ('We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, native vision capabilities, and a 1-million-token context window.', ['104 billion'])
QUOTES[('k3-paper', 'architecture')] = ('Kimi K3 is built on Kimi Delta Attention\xa0<cite class="ltx_cite ltx_citemacro_cite">[<a href="#bib.bib115" title="" class="ltx_ref">56</a>]</cite> and Attention Residuals\xa0<cite class="ltx_cite ltx_citemacro_cite">[<a href="#bib.bib56" title="" class="ltx_ref">59</a>]</cite>, which improve information flow across sequence length and model depth.', ['Kimi Delta Attention'])
QUOTES[('k3-paper', 'd')] = ('Hidden Dimension</span>\n</span></td>\n<td id="S3.T1.2.6.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.6.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.6.2.1.1" class="ltx_p ltx_align_center">7,168</span>\n</span></td>\n<td id="S3.T1.2.6.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.6.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.6.3.1.1" class="ltx_p ltx_align_center">7,168</span>\n</span></td>', ['7,168'])
QUOTES[('k3-paper', 'denseLayers')] = ('Number of Dense Layers</span>\n</span></td>\n<td id="S3.T1.2.13.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.13.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.13.2.1.1" class="ltx_p ltx_align_center">1</span>\n</span></td>\n<td id="S3.T1.2.13.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.13.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.13.3.1.1" class="ltx_p ltx_align_center">1</span>\n</span></td>', ['Number of Dense Layers'])
QUOTES[('k3-paper', 'expertFF')] = ('per Expert</span>\n</span></td>\n<td id="S3.T1.2.8.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.8.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.8.2.1.1" class="ltx_p ltx_align_center">2,048</span>\n</span></td>\n<td id="S3.T1.2.8.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.8.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.8.3.1.1" class="ltx_p ltx_align_center">3,072</span>\n</span></td>', ['3,072'])
QUOTES[('k3-paper', 'experts')] = ('Stable LatentMoE expands the routed expert space to 896 experts, with 16 activated per token, while normalization, SiTU-GLU, and Quantile Balancing stabilize optimization at extreme sparsity.', ['896 experts'])
QUOTES[('k3-paper', 'gate')] = ('In addition, Kimi K3 augments MLA with an input-dependent, channel-wise full-rank output gate.', ['output gate'])
QUOTES[('k3-paper', 'heads')] = ('Attention Heads</span>\n</span></td>\n<td id="S3.T1.2.12.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.12.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.12.2.1.1" class="ltx_p ltx_align_center">64</span>\n</span></td>\n<td id="S3.T1.2.12.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.12.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.12.3.1.1" class="ltx_p ltx_align_center">96</span>\n</span></td>', ['96'])
QUOTES[('k3-paper', 'latent')] = ('Latent MoE Dimension</span>\n</span></td>\n<td id="S3.T1.2.7.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.7.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.7.2.1.1" class="ltx_p ltx_align_center">–</span>\n</span></td>\n<td id="S3.T1.2.7.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.7.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.7.3.1.1" class="ltx_p ltx_align_center">3584 (0.5×)</span>\n</span></td>', ['3584'])
QUOTES[('k3-paper', 'layers')] = ('Layers</span>\n</span></td>\n<td id="S3.T1.2.3.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.3.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.3.2.1.1" class="ltx_p ltx_align_center">61</span>\n</span></td>\n<td id="S3.T1.2.3.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.3.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.3.3.1.1" class="ltx_p ltx_align_center">93</span>\n</span></td>', ['93'])
QUOTES[('k3-paper', 'maxContext')] = ('We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, native vision capabilities, and a 1-million-token context window.', ['1-million-token'])
QUOTES[('k3-paper', 'residual')] = ('Kimi K3 is built on Kimi Delta Attention\xa0<cite class="ltx_cite ltx_citemacro_cite">[<a href="#bib.bib115" title="" class="ltx_ref">56</a>]</cite> and Attention Residuals\xa0<cite class="ltx_cite ltx_citemacro_cite">[<a href="#bib.bib56" title="" class="ltx_ref">59</a>]</cite>, which improve information flow across sequence length and model depth.', ['Attention Residuals'])
QUOTES[('k3-paper', 'shared')] = ('Shared Experts</span>\n</span></td>\n<td id="S3.T1.2.11.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.11.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.11.2.1.1" class="ltx_p ltx_align_center">1</span>\n</span></td>\n<td id="S3.T1.2.11.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.11.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.11.3.1.1" class="ltx_p ltx_align_center">2</span>\n</span></td>', ['2'])
QUOTES[('k3-paper', 'topK')] = ('Stable LatentMoE expands the routed expert space to 896 experts, with 16 activated per token, while normalization, SiTU-GLU, and Quantile Balancing stabilize optimization at extreme sparsity.', ['16 activated'])
QUOTES[('k3-paper', 'total')] = ('We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, native vision capabilities, and a 1-million-token context window.', ['2.8T'])
QUOTES[('k3-paper', 'vision')] = ('We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, native vision capabilities, and a 1-million-token context window.', ['native vision capabilities'])
QUOTES[('k3-paper', 'vocab')] = ('Vocabulary Size</span>\n</span></td>\n<td id="S3.T1.2.14.2" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.14.2.1" class="ltx_inline-block ltx_align_top" style="width:55.2pt;">\n<span id="S3.T1.2.14.2.1.1" class="ltx_p ltx_align_center">160K</span>\n</span></td>\n<td id="S3.T1.2.14.3" class="ltx_td ltx_align_left ltx_align_top" style="padding:0.4pt 7.0pt;">\n<span id="S3.T1.2.14.3.1" class="ltx_inline-block ltx_align_top" style="width:75.9pt;">\n<span id="S3.T1.2.14.3.1.1" class="ltx_p ltx_align_center">160K</span>\n</span></td>', ['160K'])
QUOTES[('q3-paper', 'architecture')] = ('The architecture of the Qwen3 dense models is similar to Qwen2.5 <cite class="ltx_cite ltx_citemacro_citep">(<a href="#bib.bib70" title="" class="ltx_ref">Yang et al., 2024b</a>)</cite>, including using Grouped Query Attention (GQA, <cite class="ltx_cite ltx_citemacro_citep"><a href="#bib.bib3" title="" class="ltx_ref">Ainslie et al., 2023</a></cite>)', ['Grouped Query Attention'])
QUOTES[('q3-paper', 'qkNorm')] = ('Besides, we remove QKV-bias used in Qwen2 <cite class="ltx_cite ltx_citemacro_citep">(<a href="#bib.bib69" title="" class="ltx_ref">Yang et al., 2024a</a>)</cite> and introduce QK-Norm <cite class="ltx_cite ltx_citemacro_citep">(<a href="#bib.bib17" title="" class="ltx_ref">Dehghani et al., 2023</a>)</cite> to the attention mechanism to ensure stable training for Qwen3.', ['QK-Norm'])


def propose(target: dict, sources: dict, root: Path) -> tuple[dict, list[dict]]:
    out = copy.deepcopy(target)
    residue: list[dict] = []
    for claim in out.get('claims', []):
        cid = claim.get('id', '')
        field = cid.split('.')[-1]
        for entry in claim.get('evidence', []):
            if entry.get('anchor'):
                continue
            source = sources.get(entry.get('source'))
            if not source or source.get('kind') != 'document':
                continue
            row = {'claim': cid, 'source': source['id'], 'expected': claim.get('expected'),
                   'quote': None, 'upstream': None, 'reason': ''}
            authored = QUOTES.get((source['id'], field))
            if not authored:
                row['reason'] = 'no authored quote'
                residue.append(row)
                continue
            quote, contains = authored
            row['quote'] = quote
            anchor = {'kind': 'text_quote', 'quote': quote, 'contains': list(contains)}
            try:
                selected, span = select_anchor(safe_path(root, source['cache']).read_bytes(), anchor)
            except (ValueError, KeyError, OSError):
                row['reason'] = 'quote does not resolve uniquely'
                residue.append(row)
                continue
            row['upstream'] = selected
            anchor['sha256'] = hashlib.sha256(selected.encode()).hexdigest()
            anchor['resolved_span'] = span
            entry['anchor'] = anchor
    return out, residue


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--residue', required=True)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args(argv)
    root = Path(a.root).resolve()
    target = loads((root/'verification/manifests/observatory.json').read_text())
    catalog = loads((root/'verification/manifests/sources.json').read_text())
    new, residue = propose(target, {s['id']: s for s in catalog['sources']}, root)
    candidate = safe_path(root, a.candidate)
    if candidate.exists():
        raise SystemExit('Candidate already exists; use a fresh candidate path')
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(
        {'schema': 'stratum-acquisition-candidate/v1', 'catalog': catalog, 'target': new,
         'results': [], 'review_policy': 'Candidate is NOT an accepted lock. Read every quote in '
                                         'its surrounding context before promoting.'}, indent=2) + '\n')
    safe_path(root, a.residue).write_text(json.dumps(residue, indent=2) + '\n')
    print(f'{a.candidate}: proposed quotes; {len(residue)} residue rows -> {a.residue}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
