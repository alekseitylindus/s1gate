# Checkpoints are validated against a derived Parameter Manifest

Before any weight is materialized, every expected parameter must be present with the expected shape,
and an unexpected parameter is an error. The binding's own module-level loader silently ignores keys it
does not recognize, which would produce a model that runs happily on uninitialized parameters.

The Parameter Manifest is derived from the Checkpoint's own configuration — encoder configuration,
head depth, action costs — rather than hardcoded, so it cannot drift away from the architecture it
describes. Consequence: an unfamiliar checkpoint layout fails loudly at load time instead of producing
plausible but wrong Answers.
