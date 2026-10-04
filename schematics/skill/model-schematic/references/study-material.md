# Teaching with the page

The page is study material: someone should finish it able to explain the model at a
whiteboard. A correct diagram is necessary and not enough.

## Order ideas the way a learner needs them

- Whole before parts. The overview comes first and every later section says where it
  sits in that picture.
- One new idea per figure. If a figure introduces two (say, a low-rank path and a
  positional scheme), split it or stage it with a control.
- Name a thing after showing what it does. "Queries and keys are first projected to a
  small latent, then expanded per head" before "this is multi-head latent attention".
- Say why. For every distinctive mechanism, one sentence on what it buys compared with
  the plain alternative (fewer cached scalars, fewer active parameters per token, a
  longer usable context) with the number derived from facts.

## Make sizes visible

- Give every op on the data path a `width`. The gauge under the box turns a list of
  numbers into a shape the eye reads: a bottleneck, an expansion, a return.
- Use `TensorScale` wherever two tensors of very different size are compared in prose.
  If the text says "much smaller", the reader should be able to see how much.

## Every section ends with

- `Takeaways`: two to four plain sentences a reader should be able to repeat. Not a
  summary of the section's contents; the claims worth remembering.
- At least one `Check` on sections with a mechanism: a question the figure lets the
  reader answer, with the answer and its reasoning hidden until opened. Good questions
  ask for a consequence ("what is cached, and how wide?", "which layers would change
  if this config value doubled?", "why can this op not see other positions?"), never
  for a number to be read back.

## Words for a learner

- The lede says what the section is for, in one or two sentences, without jargon the
  page has not yet introduced.
- Put every term the page introduces in the `Glossary` (in the last section), one
  sentence each, in order of first use.
- Inspector `what` text explains the op's purpose, not its name. "Normalises each
  position's vector before attention reads it" rather than "applies RMSNorm".
- Captions point: "notice that the cached latent is a fourteenth of the expanded
  keys", not "figure of the cache".

## Do not

- Pad. A section that says nothing the figure does not is deleted.
- Explain general deep learning. Assume the reader knows what a linear layer and a
  softmax are; spend the words on what is specific to this model.
- Claim behaviour. The page describes architecture as written in code. What a
  mechanism "learns" or "is good at" is outside what the sources establish.
