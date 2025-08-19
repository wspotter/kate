"""Headless test for AssistantPanel basic loading & reload logic (no qtbot)."""
import json
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.core.events import EventBus
from app.services.assistant_service import AssistantService
from app.ui.components.assistant_panel import AssistantPanel


def test_assistant_panel_load_and_reload(tmp_path: Path):  # type: ignore[unused-argument]
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    assistants_path = tmp_path / "assistants.json"
    assistants_path.write_text(json.dumps({
        "one": {
            "name": "One",
            "description": "First",
            "provider": "ollama",
            "model": "mistral"
        }
    }), encoding="utf-8")

    service = AssistantService(config_path=assistants_path)
    panel = AssistantPanel(EventBus())
    panel.assistant_service = service
    panel.reload_assistants()

    assert panel.assistant_combo.count() == 1
    assert panel.get_current_assistant_id() == "one"

    assistants_path.write_text(json.dumps({
        "two": {
            "name": "Two",
            "description": "Second",
            "provider": "ollama",
            "model": "mistral"
        }
    }), encoding="utf-8")

    service.reload()
    panel.reload_assistants()
    assert panel.assistant_combo.count() == 1
    assert panel.get_current_assistant_id() == "two"
