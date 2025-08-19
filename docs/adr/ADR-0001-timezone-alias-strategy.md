# ADR-0001: Timezone Handling & Legacy Alias Strategy

Date: 2025-08-18
Status: Accepted
Authors: Recovery Team

## Context

During the August 2025 recovery, tests and runtime stability were degraded due to:

1. Naive (timezone-unaware) datetimes causing warnings & inconsistent comparisons.
2. Refactored models/settings missing legacy attribute names expected by older tests & scripts.
3. Need to integrate async providers & DB operations within the PySide6 event loop without UI freezes.

## Decision

1. Introduce a single helper `now_utc()` returning timezone-aware UTC datetimes; replace all `datetime.utcnow()` usages.
2. Preserve backward compatibility by injecting legacy attribute aliases (e.g., `created_at`) inside model constructors when absent.
3. Maintain a lightweight in-memory `Embedding` shim for tests until the richer embedding path is finalized.
4. Integrate `qasync` for the GUI event loop to safely run async tasks (DB, providers) without blocking.
5. Document required future refactor path to migrate dynamic alias injection to explicit fields/properties with tests.

## Rationale

- Ensures deterministic, comparable timestamps across DB + services.
- Avoids breaking existing tests during staged refactor.
- Provides immediate async capability with minimal surface change.
- Creates a clear audit trail (this ADR) for eventual deprecation of shims.

## Consequences

Positive:

- Stable tests for model creation & timestamp assertions.
- Cleaner future diff when removing shims (centralized list).
- Reduced risk of subtle timezone bugs.

Negative / Risks:

- Temporary complexity: dynamic `setattr` obscures model schema.
- Potential mypy type opacity for dynamically added attributes.

## Alternatives Considered

- Immediate migration to explicit new fields (rejected: higher short-term breakage risk).
- Using naive UTC + manual tz attachment (rejected: error-prone).
- Spinning a second thread for async tasks (rejected: added synchronization complexity).

## Migration Plan (Future)

1. Add explicit fields mirroring each alias (e.g., `created_at`) with `@property` for legacy name.
2. Add regression tests asserting both access paths return identical values.
3. Emit deprecation warnings (log) when legacy names accessed (phase 2).
4. Remove dynamic alias injection after two minor versions OR once external scripts updated.

## References

- `.github/copilot-instructions.md` Golden Rules & Backward Compatibility.
- `app/database/models.py` (`now_utc()` + alias injections).
- `app/core/config.py` (settings extras + legacy fields).

## Status Tracking

| Item                          | Current | Target Removal                                   |
| ----------------------------- | ------- | ------------------------------------------------ |
| Dynamic model alias injection | Active  | After v1.2 (tentative)                           |
| Embedding shim class          | Active  | Replace with service-backed model (post RAG MVP) |

---

This ADR should be updated if deprecation timelines or strategies change.
