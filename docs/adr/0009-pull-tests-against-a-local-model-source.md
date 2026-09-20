# Pull's tests run against a local Model Source

Pull's HTTP endpoint is a value, not a constant: the library builds a `Hub` at a base URL, the CLI
builds it at the Model Source's host, and tests build it at a local server that answers the way the
Model Source does — revision lookups, the resolve redirect, LFS and git blob etags.

The alternative was a trait with a fake implementation for tests. It was rejected because the parts
of Pull most worth testing are exactly the ones an in-process fake replaces: following the presigned
redirect, reading the commit and checksum headers off the first response, streaming the body to
`*.part`, and refusing bytes that do not match. With a local server, that whole path runs for real,
and a test failure means the Pull path is wrong rather than that the fake is out of step.

Consequence: the base URL stays out of the command line and out of configuration — it is the
operator's business only in that `pull` reaches the Curated Model Source, and only `pull` does
(ADR-0003). A test that needs a different Model Source adds it to the curated set and serves it
locally.
