# s1gate

s1gate judges System One Calls with local or remote models. It pulls local Checkpoints into a Model
Store and can call a remote model API when the caller selects one.

## Language

**Model Source**:
An upstream repository that publishes Checkpoints s1gate can execute.
_Avoid_: provider, upstream repo, model

**Checkpoint**:
One resolved revision of a Model Source, materialized in the Model Store. Identified by its resolved
revision, never by a moving branch name.
_Avoid_: model, weights, bundle, snapshot

**Pull**:
Fetching a Checkpoint's required files and recording its Provenance. The only operation that
downloads Checkpoints.
_Avoid_: download, fetch, install, sync

**Model Store**:
The on-disk collection of pulled Checkpoints, each held under the full name of the Model Source it
came from.
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
The implementation that judges Questions in a System One Call, either against a local Checkpoint or
through a remote model API. Laya is local; Jev is remote.
_Avoid_: engine, runtime, driver, model type

**Model Router**:
The component that resolves a System One Call's Model Identifier to the Backend that judges it.
_Avoid_: HTTP router, endpoint handler

**Model Identifier**:
The caller-chosen value of `model` in a System One Call, naming the model that judges its
Questions. `convaiinnovations/laya` names Laya's Model Source; `jev-latest` names a remote model alias.
_Avoid_: Model Source when the identifier does not name a local source

**Available Model Identifier**:
A Model Identifier whose local Checkpoint has a completed Pull record and all required files, or
whose remote Backend has a non-empty configured API key. Availability does not guarantee that
inference will succeed.
_Avoid_: installed model, verified model

**Resolved Model Identifier**:
The versioned `model` value a remote Backend returns after judging a call made with a model alias.
It identifies the model that produced the Answers.
_Avoid_: alias, requested model

**System One Call**:
One evaluation of Questions against a State using a Model Identifier. A local Backend judges every
Question in a call in a single forward pass.
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
The reported certainty of a `choice` or `score` Answer, derived from its probability distribution.
A `noul` Answer reports only the probability of true.
_Avoid_: score, certainty, margin

**Action**:
The escalate/act signal produced independently of the Question Type.
_Avoid_: escalation, cost

**Parity**:
Agreement between s1gate's output and the original implementation's output, stage by stage, within
tolerances measured rather than assumed.
_Avoid_: equivalence, correctness, matching
