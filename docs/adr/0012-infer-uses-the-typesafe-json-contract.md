# Infer uses the TypeSafe JSON contract

The response `model` rule below is amended by ADR-0014 for remote model aliases.

`infer` reads one System One Call from stdin and writes one response to stdout using the TypeSafe
request and response shapes. The request's required `model` field is the Model Identifier;
`infer` does not take a separate model name flag. The response repeats that identifier. Currently
only `convaiinnovations/laya` is supported and selects its local Model Source, while Provenance
identifies the resolved Checkpoint revision. Future identifiers may select proxies. This gives the
command one model selector and keeps its JSON compatible with the TypeSafe format.

The public Answer shapes omit the local Action signal and omit `confidence` for `noul`, following
TypeSafe's response. A future HTTP interface is a separate decision.
