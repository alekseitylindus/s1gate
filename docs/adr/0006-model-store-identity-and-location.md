# The Model Store is keyed by caller-chosen names under XDG locations

A Checkpoint is stored under the name given at Pull, not under a slug derived from its Model Source.
The set of Model Sources s1gate supports is small and curated, so an operator-chosen name is the
stable handle: it survives a Model Source moving or being renamed, and it is what configuration refers
to.

The store lives at `$XDG_DATA_HOME/s1gate` (default `~/.local/share/s1gate`) and configuration will
live at `$XDG_CONFIG_HOME/s1gate` (default `~/.config/s1gate`). The platform-native macOS location
would be `~/Library/Application Support`; XDG was chosen deliberately so one layout and one
environment variable address the store on every platform. Moving the store later would move users'
pulled Checkpoints, so the location is recorded now.

The first milestone reads no configuration file; the location is fixed here so the second milestone
does not have to invent one.
