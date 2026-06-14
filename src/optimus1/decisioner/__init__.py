"""XENON-plus decisioner: retrieval-augmented decision scorer (RADS).

RADS is intentionally self-contained. The lightweight option selector in this
package is used by the env wrapper to schedule environment-aware skills and
write option-event records next to the case library.

Modules:
- feature: vocabularies + structured feature extraction (~50d)
- encoder: QueryEncoder / CaseEncoder
- rads:    main RADS model + multi-task losses
- runtime: lightweight inference wrapper (loaded by future case_memory hook)
- option_selector: rule-gated environment skill scheduler
"""
