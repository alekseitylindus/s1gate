# Infer reports the resolved remote model

`infer` accepts the TypeSafe JSON call and validates it before selecting a Backend. For local Laya,
the response `model` remains the requested Model Identifier. For `jev-latest`, `infer` returns the
versioned `model` value from TypeSafe. This replaces ADR-0012's promise to repeat the requested
identifier in every response. The resolved version lets callers attribute each Answer to the model
that produced it even when the remote alias moves.
