# ML model architecture verifier

## Contract

Every scientific input consumed by `src/core.mjs` has an exact consumer binding. New exported constants, fields, functions, protected files or scene semantics must invalidate coverage or their approved digest. Display-only names and deliberate user scenario settings are separate classes; a model dimension cannot be reclassified as presentation to escape verification.

For a model-architecture fact, require **publisher implementation code on Hugging Face plus implementation code in an independent training/serving repository**. Config JSON is additional evidence, not a substitute for either code role. Paper prose and an implementation README cannot prove the executed graph.

The initial target pins `moonshotai/Kimi-K3` at `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721` and SGLang at `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110`. Exact file URLs are in `sources.json`. Their presence is not an acquired-source or resolved-symbol certificate.

## Per-input evidence

Bind the local consumer/value or function hash; upstream commit and file; JSON pointer or qualified Python symbol; selected-code SHA-256; relation (`equal`, `consumes`, `implements`, `derives`); execution guards; dimensions, dtype and ownership; supported phase; and claim dependencies. Record inherited configuration/defaults and weight-loader renaming. Never cite a similarly named field that an active branch does not consume.

Example: `MODELS.kimi3.d` maps to `/text_config/hidden_size` in the publisher config. It also needs a publisher projection constructor/forward and the SGLang constructor/forward that consume that width. Selecting an entire wrapper file is only a starting locator. The current manifest deliberately leaves unresolved implementation anchors null, which blocks publication.

For derived values, independently reconcile terms. Parameter count requires tensor shapes and tie/alias policy; rounded report totals are not checkpoint-manifest sums. Activated parameters, resident parameters, optimizer state and temporary materialization are different quantities.

## Structural checks

The executable contract checks exact one-based layer partition, positive tensor dimensions, routed top-k versus expert population, shared-expert separation, grouped query/KV divisibility, latent-width convention, linear contractions and equal-shape residual adds. It does **not** impose `model_width = head_count × head_dim` on architectures with separate projection dimensions.

K3 review obligations additionally cover: KDA versus MLA schedule and dense first-layer exception; depth versus token axes of AttnRes; the retained shared-key branch under NoPE; recurrent update/read order; independent convolution state; SiTU placement; full-width shared experts; weighted routed aggregation before normalization/lift; vision/projector/token interleaving; MTP exclusion; quantization and weight aliases. The current atlas tests and scene contracts cover these representations; upstream semantic corroboration still requires concrete code anchors and review.

## Runtime acceptance protocol

Use reduced-shape, deterministic fixtures through the **actual pinned implementations**, preserving the relevant branches and head/group ratios. Compare publisher reference and framework outputs using identical weights/inputs, explicit tolerances, dtype, masks and seeds. Exercise full versus chunked prefill, one-token decode, padding/ragged lengths, reset/restore, nonzero initial recurrent state, multiple requests and the first/last/boundary layers. Compare intermediate Q/K/V, routing indices/weights, recurrent state, expert aggregation and final outputs—not only end logits. Add backward/update checks for training claims.

Upstream imports, compilation and weight loading run only in an explicitly approved isolated environment. Offline CI parses source; it never executes `trust_remote_code`, pickle or unpinned installation scripts. Missing framework support becomes `BLOCKED`/`NOT_RUN`, never a simulated successful run.

## Mutation tests and acceptance

Delete a framework anchor, alter a core dimension, change a source pin, cite a comment instead of an AST symbol, duplicate a consumer, introduce a cycle, omit an expert or change a cache boundary. The appropriate source/coverage/shape gate must fail. Accept only when all required inputs and graph branches are source-bound, reviewed and—where claimed—differentially executed. There is no “mostly sourced” publication threshold.

### DSpark reference boundary

The dSpark chapter expands the pinned V4-Pro inference graph and labels it as Pro.
Its `DSPARK_PRO` facts bind to the release's `inference/config.json`; the root
Transformers config disagrees about draft depth and is cited separately. Flash
feature timing, residual read/collapse and draft dimensions remain explicit mappings.
CPU `ProDraftBlock` covers an unquantized complete block from fused context and draft
residual inputs. The attention/head walkthrough has separately supplied boundaries;
it does not imply an end-to-end model run. Conditional confidence, prefix survival,
proposal likelihood and observed acceptance must stay distinct.
