import { Section, Prose, Sub, Takeaways, Check, n, shape } from "../../../kit";
import * as f from "../facts";
import { Graph, type Step } from "./graph";
export function Kda() {
  const stepsFor = (letter: string): Step[] => [
    {
      id: `${letter}-proj`,
      label: `${letter}_proj`,
      width: f.wide.value,
      what: `Build ${letter} channels from the normalized stream.`,
      formula: `${letter}_raw = x W_${letter}ᵀ`,
      params: [["weight", shape(f.wide.value, f.hidden.value)]],
    },
    {
      id: `${letter}-conv`,
      label: "causal depthwise conv",
      width: f.wide.value,
      what: "Mix recent positions independently in each channel, with no bias.",
      formula: `z_t = Σ_{j} weight_j × input_(t+j−${n(f.convKernel.value - 1)})`,
      params: [["conv weight", shape(f.wide.value, 1, f.convKernel.value)]],
      note: "Read and update a fixed-size convolution history; zero history at prefill start. The checkpoint kernel size comes from config.",
    },
    {
      id: `${letter}-silu`,
      label: "SiLU",
      width: f.wide.value,
      what: "Apply the short convolution activation.",
      formula: "z ← z × sigmoid(z)",
    },
    {
      id: `${letter}-heads`,
      label: "reshape to heads",
      width: f.dim.value,
      what: "Expose head-local channel vectors.",
      formula: "[B,S,H×D] → [B,S,H,D]",
      output: shape("B", "S", f.heads.value, f.dim.value),
    },
    ...(letter === "v"
      ? []
      : [
          {
            id: `${letter}-l2`,
            label: "L2 normalize",
            width: f.dim.value,
            what: "Normalize each head vector inside the recurrent kernel.",
            formula: "z ← z / sqrt(Σ z² + ε_L2)",
            output: shape("B", "S", f.heads.value, f.dim.value),
            note: "This is unit-length normalization, not learned RMSNorm.",
          },
        ]),
  ];
  const gate: Step[] = [
    {
      id: "fa",
      label: "f_a_proj",
      width: f.dim.value,
      what: "Compress the stream for the per-channel decay path.",
      formula: "f_a = x W_faᵀ",
      params: [["weight", shape(f.dim.value, f.hidden.value)]],
    },
    {
      id: "fb",
      label: "f_b_proj",
      width: f.wide.value,
      what: "Expand into head/channel decay logits.",
      formula: "f = f_a W_fbᵀ",
      params: [["weight", shape(f.wide.value, f.dim.value)]],
    },
    {
      id: "dt",
      label: "bias + rate scale",
      width: f.dim.value,
      what: "Add the channel time-step bias and scale by the learned exponential head rate.",
      formula: "a = exp(A_log) × (reshape(f) + dt_bias)",
      params: [
        ["A_log", shape(f.heads.value)],
        ["dt_bias", shape(f.wide.value)],
      ],
      output: shape("B", "S", f.heads.value, f.dim.value),
    },
    {
      id: "bounded",
      label: "bounded log decay",
      width: f.dim.value,
      what: "Produce a bounded negative log decay, independently for key channels.",
      formula: `g = ${n(f.lowerBound.value)} × sigmoid(a)`,
      output: shape("B", "S", f.heads.value, f.dim.value),
      refs: [
        { source: "kda-gate", symbol: "naive_kda_lowerbound_gate" },
        { source: "reference", symbol: "KimiDeltaAttention.forward" },
        { source: "serving", symbol: "KimiK3DeltaAttention.__init__" },
      ],
    },
  ];
  const beta: Step[] = [
    {
      id: "bproj",
      label: "b_proj",
      width: f.heads.value,
      what: "Produce one update logit per token and head.",
      formula: "b = x W_bᵀ",
      params: [["weight", shape(f.heads.value, f.hidden.value)]],
    },
    {
      id: "bsig",
      label: "sigmoid update",
      width: f.heads.value,
      what: "Bound the strength of the delta correction.",
      formula: "β = sigmoid(b)",
      note: "Publisher chunk and decode modes sigmoid inside the kernel. SGLang prefill can sigmoid before it.",
    },
  ];
  const recurrentSteps: Step[] = [
    {
      id: "state",
      inputs: [["previous state", shape("B", f.heads.value, f.dim.value, f.dim.value)]],
      label: "read recurrent state",
      width: f.dim.value,
      kind: "state",
      lane: "instance",
      what: "Read head-local key/value memory, initially zero.",
      formula: "M ← previous state",
      output: shape("B", f.heads.value, f.dim.value, f.dim.value),
    },
    {
      id: "forget",
      label: "decay key rows",
      width: f.dim.value,
      what: "Apply the per-key-channel forgetting gate to memory.",
      formula: "M′ = diag(exp(g_t)) M",
      output: shape("B", f.heads.value, f.dim.value, f.dim.value),
    },
    {
      id: "predict",
      label: "predict value",
      width: f.dim.value,
      what: "Read what the current key already predicts from decayed memory.",
      formula: "v_hat = k_tᵀ M′",
      output: shape("B", f.heads.value, f.dim.value),
    },
    {
      id: "error",
      inputs: [
        ["v_t, v_hat", shape("B", f.heads.value, f.dim.value)],
        ["β_t", shape("B", f.heads.value)],
      ],
      label: "delta correction",
      width: f.dim.value,
      what: "Compute the value error and bound the update strength.",
      formula: "δ = β_t × (v_t − v_hat)",
      output: shape("B", f.heads.value, f.dim.value),
    },
    {
      id: "write",
      inputs: [
        ["M′", shape("B", f.heads.value, f.dim.value, f.dim.value)],
        ["k_t, δ", shape("B", f.heads.value, f.dim.value)],
      ],
      label: "outer-product write",
      width: f.dim.value,
      what: "Correct memory with the current key and value error.",
      formula: "M_t = M′ + k_t δᵀ",
      output: shape("B", f.heads.value, f.dim.value, f.dim.value),
    },
    {
      id: "read",
      inputs: [
        ["M_t", shape("B", f.heads.value, f.dim.value, f.dim.value)],
        ["q_t", shape("B", f.heads.value, f.dim.value)],
      ],
      label: "query read",
      width: f.dim.value,
      what: "Read the newly updated memory using the current normalized query.",
      formula: `o_t = (q_t / sqrt(${n(f.dim.value)}))ᵀ M_t`,
      output: shape("B", f.heads.value, f.dim.value),
    },
    {
      id: "store",
      inputs: [["M_final", shape("B", f.heads.value, f.dim.value, f.dim.value)]],
      label: "store state",
      width: f.dim.value,
      kind: "state",
      lane: "instance",
      what: "Keep final memory for the next decode token; prefill returns the last state.",
      formula: "cache.recurrent_state ← M_final",
      output: shape("B", f.heads.value, f.dim.value, f.dim.value),
    },
  ];
  const recurrence = recurrentSteps.map((s) => ({
    ...s,
    refs: [
      { source: "kda-math", symbol: "naive_recurrent_kda" },
      { source: "reference", symbol: "KimiDeltaAttention.forward" },
      { source: "serving", symbol: "KimiK3DeltaAttention.forward" },
    ],
  }));
  const output: Step[] = [
    {
      id: "gproj",
      label: "g_proj",
      width: f.wide.value,
      what: "Compute the full-rank output-gate logits from the input stream.",
      formula: "gate = x W_gᵀ",
      params: [["weight", shape(f.wide.value, f.hidden.value)]],
    },
    {
      id: "onorm",
      inputs: [
        ["recurrence output", shape("B", "S", f.heads.value, f.dim.value)],
        ["gate", shape("B", "S", f.wide.value)],
      ],
      label: "gated RMSNorm",
      width: f.dim.value,
      what: "Normalize the recurrence output per head and multiply by sigmoid of the separate gate.",
      formula: "o ← RMSNorm_head(o) × sigmoid(reshape(gate))",
      output: shape("B", "S", f.heads.value, f.dim.value),
      params: [["o_norm.weight", shape(f.dim.value)]],
      note: "Norm gain is shared across heads, not a separate gain for every head. Recurrence output is the other input to this op.",
    },
    {
      id: "flatten",
      label: "merge heads",
      width: f.wide.value,
      what: "Flatten head outputs into a per-token vector.",
      formula: "[B,S,H,D] → [B,S,H×D]",
    },
    {
      id: "oproj",
      label: "o_proj",
      width: f.hidden.value,
      what: "Project gated head output back to the stream.",
      formula: "y = o W_oᵀ",
      params: [["weight", shape(f.hidden.value, f.wide.value)]],
    },
  ];
  const topology: Step[] = [
    {
      id: "x",
      label: "normed stream",
      width: f.hidden.value,
      col: 0,
      row: 1,
      kind: "tensor",
      what: "All input branches consume the same normalized stream.",
      formula: "x = input_layernorm(depth_mix)",
      output: shape("B", "S", f.hidden.value),
    },
    {
      id: "qkv",
      label: "q / k / v paths",
      width: f.wide.value,
      col: 1,
      row: 0,
      what: "Independent projections, causal convolutions and SiLU; queries and keys also receive L2 normalization.",
      formula: "q,k,v = independent_short_conv_paths(x)",
      output: shape("B", "S", f.heads.value, f.dim.value),
    },
    {
      id: "gates",
      label: "decay + update",
      width: f.dim.value,
      col: 1,
      row: 1,
      what: "Separate channel-decay and head-update projections consume x.",
      formula: "g = lower_bound × sigmoid(exp(A_log) × (f_b(f_a(x)) + dt_bias)); β = sigmoid(b_proj(x))",
      output: `g ${shape("B", "S", f.heads.value, f.dim.value)}; β ${shape("B", "S", f.heads.value)}`,
    },
    {
      id: "outgate",
      label: "output gate",
      width: f.wide.value,
      col: 1,
      row: 2,
      what: "Full-rank output gate consumes x directly.",
      formula: "o_gate = g_proj(x)",
    },
    {
      id: "memory",
      label: "token memory",
      width: f.dim.value,
      col: 2,
      row: 1,
      kind: "state",
      lane: "instance",
      what: "Persistent fixed-size recurrent matrix per head; initially zero.",
      formula: "M_previous → M_final",
      output: shape("B", f.heads.value, f.dim.value, f.dim.value),
    },
    {
      id: "recur",
      label: "delta recurrence",
      width: f.dim.value,
      col: 2,
      row: 0,
      what: "Decay key rows, correct value prediction and read the updated matrix with the query.",
      formula: "M′ = diag(exp(g))M; M = M′ + βk(v−kᵀM′)ᵀ; o = qᵀM / sqrt(D)",
      output: shape("B", "S", f.heads.value, f.dim.value),
    },
    {
      id: "gate",
      label: "gated RMSNorm",
      width: f.wide.value,
      col: 3,
      row: 0,
      what: "Normalize each head output and multiply by the sigmoid output gate.",
      formula: "o = RMSNorm_head(o) × sigmoid(o_gate)",
    },
    {
      id: "out",
      label: "o_proj",
      width: f.hidden.value,
      col: 4,
      row: 0,
      what: "Project joined heads back into the residual stream.",
      formula: "y = o W_oᵀ",
      params: [["weight", shape(f.hidden.value, f.wide.value)]],
    },
  ];
  const topologyInputs: Record<string, readonly (readonly [string, string])[]> = {
    x: [["depth mixture", shape("B", "S", f.hidden.value)]],
    qkv: [["x", shape("B", "S", f.hidden.value)]],
    gates: [["x", shape("B", "S", f.hidden.value)]],
    outgate: [["x", shape("B", "S", f.hidden.value)]],
    memory: [["previous state", shape("B", f.heads.value, f.dim.value, f.dim.value)]],
    recur: [
      ["q,k,v,g", shape("B", "S", f.heads.value, f.dim.value)],
      ["β", shape("B", "S", f.heads.value)],
      ["M", shape("B", f.heads.value, f.dim.value, f.dim.value)],
    ],
    gate: [["o,o_gate", shape("B", "S", f.heads.value, f.dim.value)]],
    out: [["gated heads", shape("B", "S", f.wide.value)]],
  };
  return (
    <Section
      id="kda"
      title="A recurrent memory for tokens"
      source={[
        { source: "reference", symbol: "KimiDeltaAttention.forward" },
        { source: "kda-math", symbol: "naive_recurrent_kda" },
      ]}
      lede="KDA updates a fixed-size key/value matrix at each token. A per-channel decay forgets old state; a delta correction writes only the error in the current value prediction."
    >
      <Graph
        id="kda-topology"
        title="Parallel KDA paths"
        heading="Topology: parallel inputs, then recurrence"
        steps={topology.map((s) => ({ ...s, inputs: topologyInputs[s.id] }))}
        source="KimiDeltaAttention.forward"
        serving="KimiK3DeltaAttention.forward"
        edges={[
          { from: "x", to: "qkv" },
          { from: "x", to: "gates" },
          { from: "x", to: "outgate" },
          { from: "qkv", to: "recur" },
          { from: "gates", to: "recur" },
          { from: "memory", to: "recur", kind: "state" },
          { from: "recur", to: "memory", kind: "state" },
          { from: "recur", to: "gate" },
          { from: "outgate", to: "gate" },
          { from: "gate", to: "out" },
        ]}
      />
      <Prose>
        Queries, keys and values each have their own projection and short convolution. The forget and update paths read
        the same normalized stream in parallel. All meet in the recurrence; a separate full-rank gate controls its
        output.
      </Prose>
      {["q", "k", "v"].map((letter) => (
        <div key={letter}>
          <Graph
            id={`kda-${letter}`}
            title={`${letter} construction`}
            heading={`Build ${letter}`}
            steps={stepsFor(letter)}
            source="KimiDeltaAttention.forward"
            serving="KimiK3DeltaAttention.forward"
          />
          <Prose>
            {letter === "v"
              ? "Values retain their magnitude; only queries and keys are L2-normalized."
              : "Convolution carries recent token order into these channels before the recurrent update."}
          </Prose>
        </div>
      ))}
      <Graph
        id="kda-decay"
        title="Forget gate"
        heading="Build the per-channel decay"
        steps={gate}
        source="KimiDeltaAttention.forward"
        serving="KimiK3DeltaAttention.forward"
      />
      <Prose>
        The update coefficient has a separate projection and one scalar per head, while decay varies across key
        channels.
      </Prose>
      <Graph
        id="kda-beta"
        title="Update strength"
        heading="Build the delta coefficient"
        steps={beta}
        source="KimiDeltaAttention.forward"
        serving="KimiK3DeltaAttention.forward"
      />
      <Prose>
        The following operations occur in order for each token. The scalar browser code implements the pinned naive FLA
        equations, not the optimized GPU schedule.
      </Prose>
      <Graph
        id="kda-recur"
        title="Recurrent delta update"
        heading="Forget, correct and read"
        steps={recurrence}
        edges={[
          ...recurrence.slice(1, -1).map((s, i) => ({ from: recurrence[i].id, to: s.id })),
          { from: "write", to: "store", kind: "state" },
        ]}
        source="KimiDeltaAttention.forward"
        serving="KimiK3DeltaAttention.forward"
      />
      <Prose>
        The output gate and the recurrence output meet in a fused gated normalization, followed by the stream
        projection.
      </Prose>
      <Graph
        id="kda-output"
        title="KDA output"
        heading="Normalize, gate and project"
        steps={output}
        source="KimiDeltaAttention.forward"
        serving="KimiK3DeltaAttention.forward"
      />
      <Check question="Why can KDA's stored memory stay fixed while context grows?">
        Each new token updates the same head-local matrix and convolution history. It does not append a key/value vector
        per token. Full-attention layers still append token state.
      </Check>
      <Takeaways
        items={[
          `Each KDA layer keeps ${n(f.recurrenceScalars.value)} recurrent scalars per sequence, plus ${n(f.convolutionScalars.value)} convolution-history scalars.`,
          `Unlike token-softmax attention, the delta rule first subtracts the value already predicted by the key.`,
          `The output gate is full-rank; the forget path has an intermediate width of ${n(f.dim.value)}.`,
        ]}
      />
    </Section>
  );
}
