# Availability is one answer from the Model Router, and a configuration that cannot be read is a failure

The Model Router answers which Model Identifiers are available, for every transport: `s1gate models`
prints what it says, and `GET /v1/models` shapes it into the model list ADR-0016 describes. The
answer is decided from the Pull record and the presence of the Checkpoint's required files — never
by hashing them, which is what `verify` is for — so an Available Model Identifier can still fail to
load. `s1gate models` stays offline, listing `jev-latest` because a credential is configured rather
than because `TypeSafe` was asked; the HTTP list reads the live remote list and answers a refusal as
itself (ADR-0016).

A configuration that cannot be read is a failure for every command and every transport: an
unreadable file, invalid TOML, or an unusable key is reported, not treated as an absence. A file
that is not there, and an environment that names no Model Store location, are not failures: they
mean an empty local list.

Before this, `s1gate models` carried its own copy of the credential rule and read the Model Store by
a rule no other caller used, so it could print nothing and exit 0 where `infer` reported an error,
and could ignore `typesafe.endpoint` and list a remote Model Identifier that could not be called.
Silence was the reason that drift survived: an empty list is indistinguishable from "nothing is
configured". The cost of the decision is that `s1gate models` now fails where it used to print
nothing, which is a contract change for that command and is documented in the README.

Consequence: the Model Router holds the Model Store it resolves local Backends from, resolved once
from the environment rather than per Call, so availability and judgement read one location.
