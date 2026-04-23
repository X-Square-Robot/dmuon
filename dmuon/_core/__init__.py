"""Core shared infrastructure for DMuon.

Houses the pieces that are agnostic to the non-dedicated-parameter
runtime (FSDP2 vs DDP): owner-rank normalisation, the balanced
partition algorithm, the communication context, state-machine /
autograd hooks, and small utilities borrowed verbatim from FSDP2.

Internal — users should not import from here directly. See
``dmuon.dedicate_params`` / ``dmuon.dedicate_params_ddp`` for the
public entry points.
"""
