# Kate Development Checklist (Editable)

> This is a living document. Add, reorder, or mark tasks. Use simple [ ] / [x] markers.

## Core Stability

- [ ] Resolve PySide related mypy noise (stubs or ignore config)
- [ ] Eliminate remaining lint errors after assistant integration
- [ ] Conversation history retention (multi-turn context)
- [ ] Token budgeting / pruning strategy
- [ ] Add regression tests for legacy alias shims (models & settings)
- [ ] Begin transition: replace dynamic alias injection with explicit properties (phase 1)
- [ ] Enforce `now_utc()` usage via lint/custom check (prevent naive datetimes)

## Assistants & Prompting

- [ ] Hot reload assistants.json (file watcher)
- [ ] In-app assistant editor (create / edit / delete)
- [ ] Per-assistant default temperature & max_tokens mapping
- [ ] Display active assistant + model in chat header

## Chat & Providers

- [ ] Reintroduce stable streaming with chunk aggregation
- [ ] Provider abstraction for additional backends (OpenAI, Anthropic, etc.)
- [ ] Automatic model install/check guidance if missing
- [ ] Retry / exponential backoff on transient provider failures

## RAG Pipeline

- [ ] Document ingestion pipeline end-to-end
- [ ] Embedding generation & storage implementation
- [ ] Retrieval service top-k with similarity scoring
- [ ] Prompt assembly with cited sources
- [ ] Real evaluation metrics (relevance, coherence, factuality)
- [ ] Source attribution rendering in UI

## Evaluation & Analytics

- [ ] Replace stub evaluation for non-RAG responses with latency + token stats
- [ ] Persist evaluation results (DB table)
- [ ] Add evaluation dashboard filtering & export (CSV/JSON)
- [ ] Latency histogram & error rate metrics logger

## Persistence & Storage

- [ ] Database schema for conversations, messages, documents
- [ ] Migration framework (alembic or lightweight custom)
- [ ] Vector store backend (SQLite+FAISS or similar)

## UI / UX Enhancements

- [ ] Loading / busy indicators per panel (chat, retrieval, indexing)
- [ ] Theme switching GUI
- [ ] Resizable / collapsible panels remembered in settings
- [ ] Export conversation to Markdown / JSON

## Error Handling

- [ ] Central error dialog component
- [ ] Structured error codes & remediation hints
- [ ] Logging context enrichment (conversation_id, assistant_id)

## Testing

- [ ] Unit test harness initialization
- [ ] Assistant loader tests (valid/invalid/missing)
- [ ] Provider mock tests (chat completion & failures)
- [ ] RAG retrieval ranking tests (future)
- [ ] Performance smoke test (N sequential chats)
- [ ] Headless UI smoke test (widget creation, signal wiring)

## DevOps / CI

- [ ] GitHub Actions workflow (lint, type, test, build)
- [ ] Cache poetry deps & mypy/ruff caches
- [ ] Coverage reporting
- [ ] Release pipeline (tag -> build artifacts)
- [ ] Web server smoke test (health + websocket) integrated in CI
- [ ] ADR presence check (ensure ADR-0001 referenced in docs)

## Performance

- [ ] Profile synchronous sections for UI blocking
- [ ] Reuse single event loop for RAG worker tasks
- [ ] Embedding caching / deduplication

## Documentation

- [ ] CONTRIBUTING.md
- [ ] README feature matrix update
- [ ] Architecture diagram (update when RAG ships)
- [ ] Debugging playbook (expanded)

## Security / Privacy (Future)

- [ ] Configurable redaction of sensitive user inputs in logs
- [ ] API key vault / encrypted storage
- [ ] Sandboxed tool execution environment

## Stretch / Innovation

- [ ] Plugin system for tool functions
- [ ] Multi-document comparative synthesis mode
- [ ] Conversation quality scoring over time
- [ ] Offline packaging of selected models

---

Feel free to add sections as the project grows.
