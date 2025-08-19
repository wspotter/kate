# Kate LLM Client - Technical Architecture & Implementation Plan

## Project Overview

Kate is a modern desktop LLM client built with Python and PySide6, designed to replace the Electron-based "Augie-fresh" codebase. The application features a modern 3-column UI similar to Cherry Studio and supports multiple LLM providers with advanced features.

## Current Status (Aug 18 2025): Stabilized Post-Recovery

Emergency restoration phase completed. Application launches cleanly with async Qt (qasync), database models validated via targeted tests, and legacy compatibility layer ensures older tests & configs continue to work. Remaining focus shifts to full-suite validation, RAG pipeline re-verification, and progressive refactor of temporary alias shims.

Refer to `docs/adr/ADR-0001-timezone-alias-strategy.md` for the formal record covering timezone policy and alias shims.

### Addendum: Recovery Technical Adjustments

| Area           | Adjustment                               | Rationale                                              |
| -------------- | ---------------------------------------- | ------------------------------------------------------ |
| Event Loop     | Integrated `qasync`                      | Async providers & DB ops without blocking UI           |
| Time Handling  | Introduced `now_utc()`                   | Eliminate naive datetime warnings & ensure consistency |
| Legacy Fields  | Dynamic alias shims (models & settings)  | Maintain test pass without immediate schema migration  |
| Shutdown       | Graceful suppression of benign loop-stop | Prevent misleading fatal error noise                   |
| Document Model | Auto hash + word count on init           | Deterministic metadata & test stability                |
| Web Server     | Minimal FastAPI health + WS restored     | Enables lightweight API / remote control               |

These adjustments MUST remain until explicit migration & deprecation plan is executed.

## Legacy Compatibility Layer

During recovery, prior test suites referenced attributes not present in refactored models/settings. To bridge:

1. Model constructors (`Document`, `Conversation`, `Message`, `DocumentChunk`) inject legacy attribute names when omitted.
2. A lightweight non-ORM `Embedding` shim emulates prior structure for tests.
3. Settings model (`config.py`) permits extra fields and carries legacy flat attributes.

Refactor Path:

- Introduce explicit fields for any still-used aliases.
- Add regression tests verifying both legacy + canonical names.
- Issue deprecation notice (comment) before removal.

## Timezone & Timestamp Policy

All timestamps are timezone-aware UTC; use the shared `now_utc()` helper. Never call `datetime.utcnow()` directly. This ensures consistency across DB records, tests, and potential cross-system synchronization.

## Optional Dependency Degradation

Heavy or optional stacks (e.g., transformers, chromadb, voice libs) must not block core chat UI. Guard imports and log a single warning when disabling optional features.

## Architecture Overview

### Core Framework

```
Kate LLM Client
├── Core Application (KateApplication)
├── PySide6 UI Framework
├── SQLAlchemy 2.0 Async Database
├── Event-Driven Architecture
├── Multi-Provider LLM Support
├── Modern Theme System
└── Plugin Architecture
```

### UI Layout - 3-Column Design

```
┌─────────────────────────────────────────────────────────────┐
│ Menu Bar & Toolbar                                          │
├──────────┬─────────────────────────┬────────────────────────┤
│          │                         │                        │
│ Sidebar  │      Chat Area          │   Assistant Panel      │
│          │                         │                        │
│ - Convos │ - Message Bubbles       │ - Model Settings       │
│ - Search │ - Input Field           │ - System Prompts       │
│ - Models │ - Streaming Support     │ - Parameters           │
│          │ - File Attachments      │ - Tools/Plugins        │
│          │                         │                        │
├──────────┴─────────────────────────┴────────────────────────┤
│ Status Bar                                                  │
└─────────────────────────────────────────────────────────────┘
```

## 24-Component Implementation Roadmap

### Phase 1: Foundation (✅ COMPLETED)

1. **Project Structure** - Poetry, pyproject.toml, modern Python tooling
2. **Core Application** - KateApplication class, PySide6 main window
3. **Configuration** - Pydantic-based settings with hot-reload
4. **Database Models** - SQLAlchemy 2.0 async models for conversations/messages
5. **Event System** - Type-safe EventBus for component communication

### Phase 2: LLM Integration (✅ COMPLETED)

6. **Base Provider** - BaseLLMProvider interface, OpenAI implementation
7. **Data Models** - Pydantic models for conversations, messages, responses
8. **UI Components** - Modern chat UI with MessageBubble widgets
9. **Conversation Manager** - Async database operations and caching
10. **Multi-Provider** - Anthropic, Ollama providers with error handling

### Phase 3: Advanced Features (✅ COMPLETED)

11. **Provider Manager** - Health monitoring, model discovery, connection pooling
12. **Assistant System** - Persona management and prompt templates
13. **Extended Providers** - Gemini, Groq, Cohere, Mistral integrations
14. **Document Processing** - PDF, image, OCR processing pipeline
15. **MCP Client** - JSON-RPC protocol, tool discovery, UI integration

### Phase 4: Enterprise Features (✅ COMPLETED)

16. **Plugin Architecture** - Plugin interface, security validation
17. **Voice Processing** - Speech recognition, TTS, async audio handling
18. **Theme System** - Qt stylesheets, dark/light modes, custom themes
19. **Search Service** - Full-text, semantic search, conversation indexing
20. **Translation** - LLM-powered translation, multi-language UI

### Phase 5: Distribution & Quality (✅ COMPLETED)

21. **Build System** - PyInstaller configuration, resource bundling
22. **Update Manager** - Secure auto-update, signature verification
23. **Testing Suite** - Comprehensive pytest with async testing, UI automation
24. **Documentation** - User docs, API docs, developer guides

## Current Technical State

### ✅ Successfully Implemented

- **Core Application Framework**: KateApplication with async lifecycle
- **Database System**: SQLAlchemy 2.0 with async operations, all models working
- **Configuration**: Pydantic v2 with `pydantic-settings` integration
- **Theme System**: Qt stylesheets with "Kate Dark" theme applied
- **Event System**: Type-safe EventBus with proper event definitions
- **LLM Providers**: Base provider interface with multiple implementations
- **UI Architecture**: 3-column layout with PySide6 components

### ⚠️ Pending / In Progress

- **Full Test Suite Execution**: Only model creation subset run so far
- **RAG Pipeline Smoke Test**: Document → chunk → embed → retrieve path unverified this cycle
- **Alias Refactor**: Dynamic legacy attribute shims planned to convert to explicit fields or properties
- **Settings Consolidation**: Centralized settings UI postponed
- **Service Health Checks**: Not yet implemented

### 🎯 Immediate Next Steps (Current Cycle)

1. Run full pytest suite & catalog failures
2. Add regression test for legacy alias layer
3. Smoke test web server (HTTP + WebSocket)
4. Perform minimal RAG pipeline validation (conditional on dependencies)
5. Draft health check scaffold for core services
6. Introduce explicit properties to begin alias deprecation path (non-breaking)

## Key Technical Decisions

### Database: SQLAlchemy 2.0 Async

- **Models**: Conversation, Message, Assistant, FileAttachment
- **Migration**: Renamed `metadata` columns to `extra_data` (reserved name conflict)
- **Connection**: SQLite with aiosqlite for development, PostgreSQL-ready

### UI Framework: PySide6 (Qt6)

- **Layout**: QSplitter-based 3-column responsive design
- **Components**: Custom widgets for chat bubbles, conversation sidebar
- **Theming**: CSS-like Qt stylesheets with dark/light mode support

### Architecture Patterns

- **Event-Driven**: Decoupled components via EventBus
- **Async-First**: All I/O operations use async/await patterns
- **Type-Safe**: Pydantic models with full type annotation
- **Plugin System**: Dynamic loading with security validation

## File Structure (To Be Restored)

```
kate/
├── pyproject.toml                 # Project configuration
├── README.md                      # Project documentation
├── app/
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Application entry point
│   ├── core/                     # Core framework
│   │   ├── application.py        # KateApplication class
│   │   ├── config.py            # Settings management
│   │   └── events.py            # Event system
│   ├── database/                 # Database layer
│   │   ├── manager.py           # Database manager
│   │   └── models.py            # SQLAlchemy models
│   ├── ui/                       # User interface
│   │   ├── main_window.py       # Main 3-column window
│   │   └── components/          # UI components
│   ├── providers/                # LLM providers
│   │   ├── base.py              # Base provider interface
│   │   ├── openai_provider.py   # OpenAI implementation
│   │   └── ...                  # Other providers
│   ├── services/                 # Business logic
│   │   ├── conversation_manager.py
│   │   ├── search_service.py
│   │   └── ...
│   ├── themes/                   # Theme system
│   │   ├── manager.py           # Theme manager
│   │   └── ...
│   └── utils/                    # Utilities
├── tests/                        # Test suite
├── docs/                         # Documentation
└── build/                        # Build configuration
```

## Success Criteria

### Current Cycle (Stabilization)

- [x] Application launches without errors
- [x] Targeted database model tests pass
- [x] Timezone-aware timestamps implemented
- [x] qasync integrated for async GUI
- [ ] Full test suite green / annotated
- [ ] RAG pipeline smoke test executed
- [ ] Web server endpoints verified
- [ ] Alias regression test added

### Short-term (Functional Phase)

- [ ] Real LLM provider integration working
- [ ] Conversation persistence and loading
- [ ] Theme switching functional
- [ ] Basic chat functionality end-to-end

### Long-term (Feature Complete)

- [ ] All 24 components fully integrated and tested
- [ ] Plugin system operational
- [ ] Advanced features (voice, document processing, etc.)
- [ ] Production-ready build and distribution

## Risk Mitigation

### File Loss Prevention

- Immediate Git repository initialization
- Regular commits during restoration
- Backup to GitHub repository
- Local backup creation

### Timeline Considerations

- Focus on core functionality first
- Defer advanced features until base is stable
- Prioritize UI functionality over feature completeness
- Ensure each component works before adding next

---

**Last Updated**: August 18, 2025 - Post-Recovery Stabilization
**Status**: Core stable; validation & RAG re-verification pending
**Priority**: Expand tests, confirm pipeline, reduce interim shims
