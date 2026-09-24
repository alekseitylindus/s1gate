# s1gate

s1gate judges System One Calls with local Laya Checkpoints or `TypeSafe`'s hosted Jev Backend. Pull
stores local Checkpoints; choosing Jev sends the call to `TypeSafe`.

The vocabulary is [CONTEXT.md](CONTEXT.md); the decisions behind the design are in
[docs/adr](docs/adr).

## Requirements

macOS with the Xcode Command Line Tools, and Rust 1.88 or newer. Inference runs on the CPU through
Candle, linked against the system Accelerate framework; no Python, interpreter, CUDA or Metal
toolchain is involved (ADR-0001).

## Build

```console
$ cargo build --release
```

The binary lands at `target/release/s1gate` and links nothing but Accelerate and system libraries.

## Commands

### `s1gate pull`

Pull is the only command that downloads Checkpoints (ADR-0013). Without an argument it lists the
curated Model Sources, each with the revision the Model Store already holds for it:

```console
$ s1gate pull
convaiinnovations/laya
$ s1gate pull convaiinnovations/laya
s1gate: pulling model.safetensors
s1gate: pulling rl_agent_config.json
s1gate: pulling encoder/config.json
s1gate: pulling tokenizer/tokenizer.json
s1gate: pulling tokenizer/tokenizer_config.json
pulled convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982 into /Users/you/.local/share/s1gate/models/convaiinnovations/laya (5 files)
```

`--revision REF` pulls a revision other than the default branch. The Checkpoint is stored under the
Model Source's own name, so a name already holding another revision is refused; `--force` replaces
it.

Every file is streamed to `*.part` and renamed into place only after its size and published
checksum check out, and the Provenance record is written last, so an interrupted Pull leaves the
Checkpoint as it was.

### `s1gate models`

List the Model Identifiers currently available to `infer`, one per line, as the process's Model
Router answers it (ADR-0018). A local Model Source is listed when its completed Pull record and
every required file are present; the files are not hashed, so a listed Checkpoint can still fail to
load. `jev-latest` is listed when a non-empty `TypeSafe` API key is configured through
`TYPESAFE_API_KEY` or `typesafe.api_key` in the configuration file. The command makes no network
request. It prints an empty list when no Model Identifier is available:

```console
$ s1gate models
convaiinnovations/laya
jev-latest
```

A configuration it cannot read — an unreadable or invalid `config.toml`, or a `TYPESAFE_API_KEY`
that is empty — is a failure and exit code 1, as it is for every other command (ADR-0018), rather
than an empty list.

### `s1gate verify`

Verify stored Checkpoints against their Provenance, offline. With `--name` it verifies one Model
Source and stops at the first failure; without it, every Checkpoint in the Model Store is checked,
each failure is reported, and the exit code is 1 if any of them failed:

```console
$ s1gate verify
verified convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982 (5 files)
```

### `s1gate infer`

Judge exactly one System One Call, read as JSON on stdin. Set `model` to
`convaiinnovations/laya` to use its stored Checkpoint, or any `jev-*` identifier to use
`TypeSafe`'s hosted Backend. `infer` and `serve` resolve that identifier through the same Model
Router (ADR-0017). Laya reports the requested Model Identifier. Jev reports the
versioned Resolved Model Identifier from `TypeSafe`. The Answers are written as JSON on stdout:

```console
$ s1gate infer < call.json
```

A call is a State plus the Questions to judge against it:

```json
{
  "model": "convaiinnovations/laya",
  "state": "Ticket #4821: the customer was charged twice for order A-5512.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, invoices, refunds, and plan charges.",
        "account": "Sign-in, credentials, and profile changes."
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgently should this ticket be answered?",
      "criteria": ["low", "normal", "high", "immediate"]
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "Is the customer at risk of churning?"
    }
  }
}
```

A Question is one of exactly three types. `choice` picks a labelled Option and reports the
distribution over them; `score` reports the probability-weighted index over ordered Levels;
`noul` reports the probability of the true side. Answers keep the caller's Question ids. `choice`
accepts 1 to 255 named Options locally; `score` accepts 1 to 10 Levels locally. Jev calls follow
the published `TypeSafe` schema, which places no Option count limit on Choice. `state` can be a
string, object, or array. `instructions` can also be omitted or null. The numbers below only show
the shape of one response:

```json
{
  "model": "convaiinnovations/laya",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.9988, "account": 0.0012 },
      "confidence": 0.9812
    },
    "urgency": {
      "type": "score",
      "score": 2.9142,
      "legend": { "0": "low", "1": "normal", "2": "high", "3": "immediate" },
      "probabilities": { "0": 0.0, "1": 0.0001, "2": 0.0856, "3": 0.9143 },
      "confidence": 0.4417
    },
    "churn_risk": {
      "type": "noul",
      "noul": 0.7314
    }
  },
  "usage": { "input_tokens": 214, "output_tokens": 0 }
}
```

To use Jev, set `model` to `jev-latest` and provide a `TypeSafe` API key through configuration or
`TYPESAFE_API_KEY`. That choice sends the call's State and Questions to `TypeSafe`. Laya uses its
stored Checkpoint and makes no network request.

You can store the key in `$XDG_CONFIG_HOME/s1gate/config.toml`, or in `~/.config/s1gate/config.toml`
when `XDG_CONFIG_HOME` is unset or empty:

```toml
[typesafe]
api_key = "your-api-key"
# Optional. Defaults to TypeSafe's production endpoint.
endpoint = "https://api.typesafe.ai/v1/systemone"
```

Restrict this file to your user, for example with `chmod 600 ~/.config/s1gate/config.toml`. A set
`TYPESAFE_API_KEY` takes precedence for that process. An empty environment value is an error and
does not fall back to the file. `typesafe.endpoint` overrides the default Jev endpoint; the model
list is read from `models` on the same base, so a proxy reached through `typesafe.endpoint`
answers `GET /v1/models` too.

Jev HTTP 422 responses exit with code 2 because `TypeSafe` rejected the System One Call. Authentication,
other HTTP, network, and malformed-response failures exit with code 1. Error output includes the HTTP
status and a short API message when available; it omits the API key and request body. HTTP 429 and 529
responses are retried up to two times. Each retry honors a `Retry-After` delay in seconds or HTTP date,
or waits 200 ms and then 400 ms when the header is absent or invalid. After the final attempt, `infer`
reports the last failure and writes no success JSON.

A description reaches the prompt as the caller wrote it: a string as it stands, anything structured
as JSON. `choice` Criteria is an object whose values are strings, objects, arrays, or `null`.
`score` Criteria is an ordered array of string, object, or array descriptions. `noul` Criteria may
give either or both of `true` and `false` descriptions with those same types.

Numbers are rounded half-to-even at four decimal places, matching the original implementation.

### `s1gate serve`

Serve System One Calls at `POST /v1/systemone`. The listener defaults to `127.0.0.1:8080`;
`--host` and `--port` set another address. Send the same JSON Call used by `infer` with
`Content-Type: application/json`. Valid Calls receive HTTP 200 with the `TypeSafe` response shape.
Invalid Calls receive HTTP 422 with a `detail` array.

At startup, the server loads each present local Checkpoint and stops if one is invalid. It skips
absent Checkpoints. Local Calls run sequentially against the loaded Checkpoint until restart, so
changes to the Model Store take effect after a restart. The server never Pulls a Checkpoint.
Incoming Authorization is optional; Jev Calls use the server's configured `TYPESAFE_API_KEY`, which
no request header can replace.

A Call whose Model Identifier is a `jev-*` value is forwarded to `TypeSafe`, whether it names an
alias such as `jev-latest` or a version. The server answers with `TypeSafe`'s own result: a judged
Call returns the resolved `model`, its named `answers`, and its `usage`, while a refusal keeps
`TypeSafe`'s status, JSON body, and `Retry-After` header when it sent one. HTTP 429 and 529 answers
are retried up to two times under the same policy `infer` applies (ADR-0015). Any other status is
answered once, as it stands. A failure that leaves no answer of `TypeSafe`'s to forward — no
configured key, no route to the API, or a 200 that is not one complete response — answers HTTP 502
with a `detail` array. Remote Calls proceed concurrently, and a remote Call never waits for local
inference (ADR-0015).

`GET /v1/models` returns a TypeSafe-shaped `models` array. It lists every local Backend loaded at
startup, followed by the models `TypeSafe` currently serves when a key is configured. Laya appears
as `convaiinnovations/laya` with release date `2026-09-18` when its Checkpoint was loaded; remote
aliases appear with the description and release date `TypeSafe` publishes for them. The remote list
comes from the model-list endpoint beside `typesafe.endpoint`, which defaults to
`https://api.typesafe.ai/v1/models`.

Without a key, the list holds the local models only and makes no remote request. A failed remote
list is answered as `TypeSafe` answered it — its final status, its JSON body, and its `Retry-After`
header when it sent one — rather than as a partial list, so a client never reads an incomplete list
as the whole set. Versioned `jev-*` identifiers are accepted by `POST /v1/systemone` whether or not
the list names them.

## Model Store

The Model Store lives at `$XDG_DATA_HOME/s1gate/models`, defaulting to
`~/.local/share/s1gate/models`. Each Checkpoint sits under its Model Source's full name,
`<owner>/<name>`, beside the `provenance.json` that records the requested and resolved revisions
and each file's size and checksum (ADR-0011). One Model Source holds one Checkpoint.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | runtime error |
| `2` | usage error |

## License

Apache-2.0. See [LICENSE](LICENSE).
