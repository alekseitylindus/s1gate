# s1gate

s1gate runs local System One decision models natively. It pulls a Checkpoint into a local Model
Store and judges questions against it, without network access at judgement time.

## Language

**Model Source**:
An upstream repository that publishes Checkpoints s1gate can execute.
_Avoid_: provider, upstream repo, model

**Checkpoint**:
One resolved revision of a Model Source, materialized in the Model Store. Identified by its resolved
revision, never by a moving branch name.
_Avoid_: model, weights, bundle, snapshot

**Pull**:
Fetching a Checkpoint's required files and recording its Provenance. The only operation that reaches
the network.
_Avoid_: download, fetch, install, sync

**Model Store**:
The on-disk collection of pulled Checkpoints, each held under the name it was pulled with.
_Avoid_: cache, registry, models dir

**Provenance**:
The recorded facts identifying a pulled Checkpoint: Model Source, requested revision, resolved
revision, and each file's size and checksum.
_Avoid_: metadata, lockfile, manifest

**Published Checksum**:
The checksum a Model Source publishes for one of a Checkpoint's files, together with the algorithm it
is expressed in. Pull verifies each file against it and records it beside the local SHA-256.
_Avoid_: etag, upstream hash, remote checksum

**Parameter Manifest**:
The expected parameter names and shapes a Backend derives from a Checkpoint's own configuration
before loading any weights.
_Avoid_: key list, index, tensor table

**Backend**:
The implementation that judges questions against a Checkpoint. Laya is the only Backend in the first
milestone; Jev is a planned one.
_Avoid_: engine, runtime, driver, model type

**System One Call**:
One invocation of a Backend: a State plus the Questions to judge against it. Every Question in a call
is judged in a single forward pass.
_Avoid_: request, predict, batch, run

**State**:
The evidence a Question is judged against.
_Avoid_: context, input, document, prompt

**Question**:
One item to be judged within a System One Call, identified by a caller-chosen id.
_Avoid_: item, query, task

**Question Type**:
The kind of Answer a Question expects: `choice`, `score`, or `noul`. Exactly three exist.
_Avoid_: judgment kind, mode, qtype

**Criteria**:
The answer space a Question defines — labelled options for `choice`, ordered levels for `score`, the
false/true pair for `noul`. Set at call time, so a new answer space needs no retraining.
_Avoid_: labels, classes, options

**Option**:
One candidate answer rendered into a Question's prompt, for `choice` and `noul`.
_Avoid_: label, class

**Level**:
One ordered step of a `score` Question's Criteria.
_Avoid_: step, bucket, bin

**Marker**:
The position in a Question's prompt that carries one Option's score. An Answer is the distribution
over a Question's Markers.
_Avoid_: slot, mask position, logit index

**Answer**:
A Backend's result for one Question.
_Avoid_: result, prediction, response, output

**Calibration Temperature**:
The scaling applied to a Question's Markers before they become an Answer, selected per Question Type
and option count. Fitted outside s1gate; never learned at judgement time.
_Avoid_: temperature (unqualified — a Checkpoint carries an unused parameter of that name)

**Confidence**:
The reported certainty of an Answer: normalized entropy for `choice` and `score`, the probability of
the stronger side for `noul`.
_Avoid_: score, certainty, margin

**Action**:
The escalate/act signal every Answer carries, produced independently of the Question Type.
_Avoid_: escalation, cost

**Parity**:
Agreement between s1gate's output and the original implementation's output, stage by stage, within
tolerances measured rather than assumed.
_Avoid_: equivalence, correctness, matching
