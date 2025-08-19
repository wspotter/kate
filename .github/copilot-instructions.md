# GitHub Copilot Instructions for the Kate Project

> Last Updated: 2025-08-18

These instructions guide AI assistants (and contributors) on how to operate safely within the Kate codebase. Follow them to avoid regressions and preserve backward compatibility established during the August 2025 recovery cycle.

## 1. Project Intent

Kate is a PySide6 desktop LLM client with optional RAG, voice, and provider integrations. Primary goals:

1. Fast, reliable chat UI (3-column layout) with multi-provider support.
2. Stable async foundation (qasync + SQLAlchemy async).
3. Backward compatible data & settings models (legacy tests must pass).
4. Progressive enablement of RAG + voice features WITHOUT breaking core UX if dependencies are missing.

## 2. Golden Rules

1. Do NOT remove legacy alias shims in database models or settings without adding explicit replacement fields & updating all tests.
2. Keep GUI launch ( `python -m app.main` ) green before and after changes.
3. Avoid adding heavy imports (transformers, audio, vision) at top-level of modules used by the main window; lazy-load inside functions/services.
4. Maintain timezone-aware timestamps (use `now_utc()` helper instead of `datetime.utcnow()`).
5. Never silently swallow exceptions in core services—log via loguru with context.
6. Prefer adding focused tests before large refactors (database, providers, RAG pipeline).
7. Keep optional dependencies optional (guard imports; degrade gracefully).
8. Preserve existing public function/class names unless a deprecation path is documented.

## 3. Architectural Pillars

| Layer            | Key Modules                              | Notes                                          |
| ---------------- | ---------------------------------------- | ---------------------------------------------- |
| Core Application | `app/core/application.py`, `app/main.py` | Startup, event loop, service bootstrap         |
| Configuration    | `app/core/config.py`                     | Pydantic v2 settings w/ extras + legacy fields |
| Database         | `app/database/models.py`, `manager.py`   | SQLAlchemy async + legacy attribute aliasing   |
| Providers        | `app/providers/*`                        | Uniform interface; streaming; error handling   |
| Services         | `app/services/*`                         | Assistant, RAG, search, voice (some optional)  |
| UI               | `app/ui/*`                               | PySide6 widgets, panels, layout                |
| Web API          | `app/web_server.py`                      | Minimal FastAPI (health + websocket)           |

## 4. Backward Compatibility Strategy

Older tests and scripts expect certain flat attributes or legacy field names. Current approach:

- Constructor alias injection: Models set attributes like `created_at` if not provided.
- Settings model allows unknown extras via `model_config` for forward/backward tolerance.
- Lightweight in-memory `Embedding` shim retained for tests even if a richer embedding path exists elsewhere.

When refactoring:

- Replace dynamic `setattr` logic with explicit fields + `@property` accessors incrementally.
- Add regression tests before removing a shim.

Authoritative record: see `docs/adr/ADR-0001-timezone-alias-strategy.md` for rationale, migration path, and deprecation timeline governing timezone policy & legacy alias shims.

## 5. Timezone & Deterministic Timestamps

Use `now_utc()` helper (timezone-aware, UTC) for all default timestamps. Never reintroduce naive datetimes. Tests assume aware objects.

## 6. Async & Event Loop Guidance

- Main entry uses qasync—avoid nested event loops; use `asyncio.create_task` within UI components.
- Long-running blocking code must be moved into `asyncio.to_thread` or worker threads.
- Shut down cleanly: ensure tasks respond to cancellation; catch benign loop stop RuntimeError only where already documented.

## 7. Optional Dependency Handling

Wrap imports:

```python
try:
    import chromadb
except Exception:  # broad guard is acceptable for optional features
    chromadb = None
```

Then feature-gate:

```python
if chromadb is None:
    logger.warning("Chroma not available; semantic retrieval disabled")
```

## 8. Testing Conventions

| Category           | Command                                                                        | Notes                              |
| ------------------ | ------------------------------------------------------------------------------ | ---------------------------------- |
| Full suite         | `poetry run pytest`                                                            | Run before large merges            |
| Fast models subset | `pytest tests/test_database.py::TestDatabaseModels::test_document_creation -q` | Useful for quick model sanity      |
| Type check         | `poetry run mypy app/`                                                         | Address new errors immediately     |
| Lint               | `poetry run ruff check app/ tests/`                                            | Maintain style & catch unused code |

Add tests for:

1. New database fields (creation + retrieval).
2. Behavioral changes in providers (streaming vs. non-streaming).
3. Legacy alias removal (assert both old & new still work until deprecation window ends).

## 9. Safe Editing Checklist (Pre-Commit)

1. Run targeted pytest for affected area.
2. Launch GUI once and close to confirm no regressions.
3. Run mypy (ensure no new errors; warnings acceptable only if pre-existing).
4. Run ruff; auto-fix trivial issues if possible.
5. Update docs if public API or user workflow changes.

## 10. Logging & Error Handling

- Use `logger = loguru.logger`.
- Include contextual metadata (ids, counts) in log lines.
- Avoid printing directly; rely on central logging config.
- For recoverable provider failures, downgrade to warning; for silent feature disablement, log once at startup.

## 11. Performance Guardrails

- Avoid loading large ML models during initial GUI render; defer until first use.
- Batch database writes where possible in import or indexing flows.
- Keep main thread responsive (never block over 50ms synchronous work in UI callbacks).

## 12. Refactor Policy

Refactors must accompany:

- Rationale in PR description.
- Before/after behavioral summary.
- Added or updated tests validating unchanged externally visible behavior.

## 13. Security Considerations

- Never execute untrusted code from LLM outputs automatically.
- Validate plugin manifests (signature / schema) before enabling (future hardening).
- Avoid leaking API keys in logs—mask values after first 4 characters.

## 14. Common Pitfalls to Avoid

| Pitfall                                 | Impact                      | Avoidance                      |
| --------------------------------------- | --------------------------- | ------------------------------ |
| Replacing aware datetimes with naive    | Test failures & subtle bugs | Always use `now_utc()`         |
| Eager heavy imports in UI modules       | Slower startup / freezes    | Lazy-load internally           |
| Removing legacy fields abruptly         | Legacy test breakage        | Add deprecation period + tests |
| Swallowing provider exceptions silently | Hard to debug               | Log with context               |

## 15. When Unsure

Prefer: small, reversible changes + added test. If an architectural decision is ambiguous, document a short ADR (Architecture Decision Record) in `docs/adr/ADR-<sequence>-<slug>.md` (create folder if absent).

---

Following these instructions keeps the project stable while enabling incremental advancement toward production readiness. Update this file whenever recovery strategies, compatibility layers, or core architectural patterns change.
