# Kate Developer User Guide

This guide explains the architecture, workflows, and day-to-day development tasks for the Kate LLM Desktop Client.

## 1. Overview

Kate is a PySide6 desktop client providing chat + (future) Retrieval-Augmented Generation (RAG) with local or remote LLM providers (currently Ollama). It layers a modular architecture for future providers, document indexing, and evaluation.

### Layered Architecture

1. **UI Layer**: PySide6 widgets (`main_window`, `chat_area`, `assistant_panel`, etc.).
2. **Application/Core**: Lifecycle (`KateApplication`), event bus, configuration, theme management.
3. **Services**: RAG, retrieval, embedding, vector store, evaluation, search, updates.
4. **Providers**: Abstractions + concrete adapters (e.g. `ollama_provider`).
5. **Persistence**: Database manager and models (future: conversation/history, document metadata, embeddings).
6. **Utilities**: Logging, platform/system helpers.
7. **Configuration & Data**: `assistants.json`, themes, settings.

## 2. Key Components

| Area          | File(s)                                                              | Purpose                                           |
| ------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| Main Window   | `app/ui/main_window.py`                                              | Composes 3-column UI, wires signals               |
| Chat          | `app/ui/components/chat_area.py`                                     | Manages message flow, provider calls, RAG worker  |
| Assistants    | `app/ui/components/assistant_panel.py`, `app/config/assistants.json` | Assistant selection, model settings, eval metrics |
| Providers     | `app/providers/base.py`, `app/providers/ollama_provider.py`          | Provider interface + Ollama implementation        |
| Evaluation    | `app/services/rag_evaluation_service.py`                             | Response scoring (stub + future full RAG)         |
| Retrieval/RAG | `app/services/*rag*`                                                 | Retrieval orchestration (in progress)             |
| Logging       | `app/utils/logging.py`                                               | Configures loguru                                 |
| Themes        | `app/themes/`                                                        | Theme base + manager                              |
| App Lifecycle | `app/core/application.py`                                            | Startup, provider init, timers                    |

## 3. Data Flow (Non-RAG Chat)

User types message → `ChatArea` adds user bubble → builds `ChatCompletionRequest` → `ollama_provider.chat_completion()` → response parsed → assistant bubble added → evaluation stub emitted → assistant panel metrics update.

## 4. Assistants

Assistants are defined in `app/config/assistants.json`. Each entry includes:

```json
"coding": {
  "name": "Code Engineer",
  "description": "...",
  "system_prompt": "...",
  "avatar": "💻",
  "provider": "ollama",
  "model": "codellama"
}
```

Modify or add entries; restart the app to reload. A future enhancement may add hot reload.

## 5. Adding a Provider

1. Create `app/providers/<new>_provider.py` implementing methods from `BaseLLMProvider`.
2. Update application startup to instantiate + connect.
3. Add mapping logic for selecting default model.
4. Extend assistant definitions referencing the new provider.

## 6. RAG Pipeline (Planned)

Planned stages:

1. Document ingestion (`DocumentProcessor`) → chunking & metadata.
2. Embeddings (`EmbeddingService`) → vector store.
3. Retrieval (`RetrievalService`) → top-k chunk selection.
4. Augmented prompt assembly.
5. Provider completion (with context citation tags).
6. Evaluation (`RAGEvaluationService`) scoring relevance, coherence, etc.

## 7. Evaluation Metrics

Current state: stub metrics for non-RAG responses. Future: real metrics triggered post-retrieval with citation accuracy & source coverage.

## 8. Logging & Debugging

- Use `logger.bind(component="Name")` per module.
- Levels: INFO (lifecycle), DEBUG (flow details), ERROR (failures).
- Consider enabling JSON logs for CI via environment flag (future).
- Timezone policy enforced: always use `now_utc()` helper (see ADR-0001).
- Legacy alias shims must not be removed without tests (see ADR-0001).

## 9. Development Environment

Install dependencies:

```
poetry install
```

Run the app:

```
poetry run python -m app.main

Optional minimal web server (separate terminal):

```

poetry run python -m app.web_server

```

```

Run tests (after adding):

```
poetry run pytest
```

Format & lint:

```
poetry run black app/ tests/
poetry run ruff check app/ tests/
```

Type checking:

```
poetry run mypy app/
```

## 10. Planned Improvements (Condensed)

- Multi-turn conversation context with token budgeting.
- Stable streaming re-enable (guarded).
- Conversation & document persistence.
- Assistant create/edit UI (writes JSON).
- CI pipeline (lint, test, build artifacts).
- Real RAG retrieval + evaluation metrics.
- Plugin provider architecture.

## 11. Testing Strategy (Initial Targets)

| Test                   | Focus                                  |
| ---------------------- | -------------------------------------- |
| Assistant JSON load    | Valid/invalid/missing file fallback    |
| Provider health check  | Status transitions                     |
| Chat request build     | System prompt & model settings applied |
| Conversation history   | Accumulation & pruning                 |
| RAG retrieval (future) | Top-k correctness                      |

## 12. Contribution Workflow

1. Create feature branch.
2. Run lint + type checks locally.
3. Add/Update tests.
4. Submit PR (include description & screenshots/log excerpts where relevant).
5. CI must pass before merge.

## 13. Release & Distribution (Planned)

PyInstaller build task already present. Future: GitHub Actions pipeline produces cross-platform artifacts + changelog.

## 14. Glossary

- **Assistant**: Configured persona with system prompt + provider + model.
- **RAG**: Retrieval-Augmented Generation; adds document context.
- **Evaluation**: Post-response scoring metrics.
- **Vector Store**: Embedding index for similarity search.

## 15. Support / Troubleshooting

| Symptom               | Action                                                              |
| --------------------- | ------------------------------------------------------------------- |
| "Disconnected" status | Confirm Ollama running; re-run provider init logs                   |
| Empty assistant list  | Validate JSON syntax; check path in `_load_assistants()`            |
| No response           | Check logs for provider exception; ensure model installed in Ollama |
| Metrics all 0.5       | RAG not active; stub values displayed                               |

## 16. Roadmap Snapshot

See `dev_checklist.md` for an editable living roadmap.

---

Maintain this guide as architecture evolves. Keep sections concise but current.
