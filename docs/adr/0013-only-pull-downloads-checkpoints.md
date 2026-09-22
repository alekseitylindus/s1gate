# Only Pull downloads Checkpoints

Pull alone downloads Checkpoints into the Model Store. `infer` may call the remote TypeSafe API when
the System One Call selects `jev-latest`; local Laya inference still uses no network. This narrows
ADR-0003's ban on network access: the intended guarantee was that judgement never downloads model
files implicitly, while an explicitly selected remote model necessarily requires a network call.
