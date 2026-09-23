# One Model Router serves every transport

`infer` and `serve` resolve a System One Call's Model Identifier through the same Model Router, so a
transport never repeats the selection branch and the two interfaces cannot drift. The rules are
ordered: an exact local Model Identifier selects its local Backend; any other identifier that starts
with `jev-` is forwarded to TypeSafe with the process's configured key; anything else is an
unsupported-model error. `jev-*` is the only wildcard: a typo fails instead of silently reaching
another Backend. ADR-0012's single `model` field stays the only model selector.

The Model Router takes a serialized System One Call and returns either the judging Backend's response
or the remote Backend's Refusal, and it lists available Model Identifiers the same way, so it neither
reads nor writes a transport stream. The HTTP server is ADR-0015's; model discovery is ADR-0016's; a
later MCP endpoint can call the same Router without a new selection path.

The two transports differ only in where the local Backend comes from. A CLI process builds a Model
Router over the Model Store and selects the Checkpoint per Call, because it exits after one Call. The
server loads each present local Checkpoint once at startup, fails startup when a present Checkpoint
does not verify or load, and skips absent ones, so repeated HTTP Calls do not reload weights. Its
Router holds the loaded Backend behind a lock: local Calls are judged one at a time, while remote
Calls proceed concurrently (ADR-0015). Loaded Checkpoints are never reloaded or Pulled, so a change
to the Model Store takes effect at restart.

A Call whose local Checkpoint is missing is an error naming the Pull that would provide it, not a
fallback to a remote Backend, and an absent credential fails only a Call that needs the remote one.
