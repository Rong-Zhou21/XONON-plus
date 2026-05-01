"""XENON-plus decisioner: retrieval-augmented decision scorer (RADS).

This package is intentionally self-contained. It depends only on PyTorch and
the on-disk case library (`src/optimus1/memories/.../cases.json`). It does not
modify the case library schema, the env wrapper, or the planner.

Modules:
- feature: vocabularies + structured feature extraction (~50d)
- encoder: QueryEncoder / CaseEncoder
- rads:    main RADS model + multi-task losses
- runtime: lightweight inference wrapper (loaded by future case_memory hook)
"""
