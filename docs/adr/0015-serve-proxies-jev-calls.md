# The HTTP server proxies Jev Calls with TypeSafe's own answers

`s1gate serve` forwards a `jev-*` System One Call to `TypeSafe` with the server's configured key and
answers its client with what `TypeSafe` itself returned. A judged Call keeps the resolved `model`,
the named `answers`, and the `usage` (ADR-0014). A refusal keeps `TypeSafe`'s status, its JSON body,
including error data, and its `Retry-After` header when it sent one. An HTTP client therefore reads
the failure `TypeSafe` reported rather than a reserialization of it, and backs off on the delay
`TypeSafe` asked for. The bounded 429/529 retry policy the CLI path already applies runs before the
final answer is forwarded; no other status is retried.

The CLI's diagnostics stay its own: `infer` prints a short message with neither the credential nor
the request data, instead of the remote body. One refusal carries both the forwardable answer and
that safe message, so no transport re-parses or re-redacts it.

A failure that leaves no answer of `TypeSafe`'s to forward — no configured credential, no route to
the API, or a 200 that is not one complete response — answers HTTP 502, because there is no status
or body to forward.

The Model Router holds the lock that serializes local inference, and each connection is served on
its own thread. Remote Calls therefore proceed concurrently, and a remote Call never queues behind
local inference.
