# Model discovery merges the live TypeSafe model list

`GET /v1/models` answers in the TypeSafe model-list shape: entries a client can discover, each with a
`name`, a `description`, and a `release_date`. The Model Router builds that list from the local
Backends the server loaded at startup, followed by the models TypeSafe itself lists, read from the
model-list endpoint on the base of the configured System One endpoint. The credential and the
forwarding rules are ADR-0015's. A process with no configured key answers its local entries alone and
makes no remote request, so a local-only server stays usable and offline.

The remote list is the set of remote aliases a client can send, not decoration. A failed list request
therefore answers as TypeSafe answered it — its final status, its JSON body, and its `Retry-After`
header when it sent one — after the same bounded 429/529 retries a forwarded Call gets. The server
never falls back to the local entries alone: a client cannot tell a partial list from the whole set,
and would read an unreachable remote model as an unavailable one.

Identifiers the list omits remain usable. TypeSafe serves versioned `jev-*` IDs it publishes no alias
for, and the Model Router forwards any `jev-*` identifier to it, so a client that learned an ID from
anywhere may still call it.
