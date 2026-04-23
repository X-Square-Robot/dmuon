"""Per-runtime backends for non-dedicated parameter management.

Each subpackage implements the machinery that handles the parameters
DMuon does NOT own (FSDP2-sharded / DDP-replicated / ...) and the
variant of dedicated-parameter lifecycle appropriate for that runtime:

* :mod:`dmuon._backends.fsdp2` — dedicated params with packed-buffer
  broadcast/reshard + FSDP2 monkey-patch for non-dedicated params.
* :mod:`dmuon._backends.ddp` — dedicated params live on every rank;
  non-dedicated params get a post-accumulate-grad all-reduce hook.

Internal.
"""
