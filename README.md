# s1gate

s1gate runs local System One decision models natively. It pulls a Checkpoint into a local Model
Store and judges questions against it, without network access at judgement time.

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

Pull is the only command that reaches the network (ADR-0003). Without an argument it lists the
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

### `s1gate verify`

Verify stored Checkpoints against their Provenance, offline. With `--name` it verifies one Model
Source and stops at the first failure; without it, every Checkpoint in the Model Store is checked,
each failure is reported, and the exit code is 1 if any of them failed:

```console
$ s1gate verify
verified convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982 (5 files)
```

### `s1gate infer`

Judge exactly one System One Call, read as JSON on stdin, against a stored Checkpoint. The Answers
are written as JSON on stdout:

```console
$ s1gate infer --name convaiinnovations/laya < call.json
```

A call is a State plus the Questions to judge against it:

```json
{
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
`noul` reports the probability of the true side. Every Answer carries the Action signal beside it,
under `action.act_probability`. Answers keep the caller's Question ids — the numbers below only show
the shape of one:

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "route": {
      "type": "choice",
      "action": { "act_probability": 0.0123 },
      "choice": "billing",
      "probabilities": { "billing": 0.9988, "account": 0.0012 },
      "confidence": 0.9812
    },
    "urgency": {
      "type": "score",
      "action": { "act_probability": 0.0123 },
      "score": 2.9142,
      "legend": { "0": "low", "1": "normal", "2": "high", "3": "immediate" },
      "probabilities": { "0": 0.0, "1": 0.0001, "2": 0.0856, "3": 0.9143 },
      "confidence": 0.4417
    },
    "churn_risk": {
      "type": "noul",
      "action": { "act_probability": 0.0123 },
      "noul": 0.7314,
      "confidence": 0.7314
    }
  },
  "usage": { "input_tokens": 214, "output_tokens": 0 }
}
```

Numbers are rounded half-to-even at four decimal places, matching the original implementation.

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
