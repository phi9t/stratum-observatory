# Observatory engineering spec

## Objective
A single portable HTML scientific instrument connecting eight exact model variants through seven coordinated lenses. Shared immutable model definitions drive diagrams and calculations. The first-principles design is in `design/OBSERVATORY.md`; numeric assumptions and exclusions are in the README and inspectors.

## Contracts
1. A variant ID identifies one checkpoint, never an entire changing family.
2. A model field maps to publisher configuration and concrete publisher/reference plus independent serving implementation. Preserve pending/raw/review statuses separately.
3. Model differences change tensor paths, not just headings: MLA, GQA, vector-decay KDA, scalar GatedDeltaNet, QSA, PLE, ordinary/depth/four-stream residuals and latent/shared MoE are separate motifs.
4. Main dependency direction and state/depth axes must be explicit. Repeated layers compress in the overview; every decoder layer is selectable.
5. Orthogonal edges, stable semantic rows, reserved label/rail/header bands. No core text clip, ellipsis or accidental wrap.
6. GPU, superchip, node and rack are different scopes. Tensor peak has dtype/sparsity; bandwidth has direction and units. Selected hardware is hypothetical placement, not a publisher claim.
7. Logical state/weight/traffic accounting names excluded buffers. Unknown or unsupported quantities remain unknown; roofline is a ceiling, not a measurement.
8. Search, selection, trace, keyboard, URL, theme, zoom, scenario controls, mobile scrolling and exports must operate on the real state.
9. Single deterministic HTML; no runtime external assets or fonts.
10. Engineering results do not approve source semantics or certify hardware execution.

## Extension
For numeric changes update `models/core` and tests. For graph changes update the domain template and semantic/geometry tests. For UI changes update `app/styles` and interaction tests. Update the live inventory adapter when adding public inputs or modules. Run all gates, with missing evidence expected to block publication. Never regenerate accepted evidence in place.
