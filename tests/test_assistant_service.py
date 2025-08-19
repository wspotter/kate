"""Tests for AssistantService loading & validation."""
import json
from pathlib import Path

from app.services.assistant_service import AssistantService


def test_fallback_when_no_file(tmp_path: Path):
    service = AssistantService(config_path=tmp_path / "missing.json")
    assistants = service.get_assistants()
    assert "general" in assistants
    assert assistants["general"]["model"] == "mistral"


def test_load_valid_file(tmp_path: Path):
    cfg = {
        "general": {
            "name": "General Assistant",
            "description": "Helps generally",
            "provider": "ollama",
            "model": "mistral",
            "system_prompt": "You are helpful"
        },
        "coding": {
            "name": "Coder",
            "description": "Writes code",
            "provider": "ollama",
            "model": "codellama"
        }
    }
    p = tmp_path / "assistants.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    service = AssistantService(config_path=p)
    assistants = service.get_assistants()
    assert set(assistants.keys()) == {"general", "coding"}
    assert assistants["coding"]["model"] == "codellama"


def test_reload(tmp_path: Path):
    p = tmp_path / "assistants.json"
    p.write_text(json.dumps({
        "a": {
            "name": "A",
            "description": "desc",
            "provider": "ollama",
            "model": "mistral"
        }
    }), encoding="utf-8")
    service = AssistantService(config_path=p)
    assert set(service.get_assistants().keys()) == {"a"}
    # Modify file
    p.write_text(json.dumps({
        "b": {
            "name": "B",
            "description": "desc",
            "provider": "ollama",
            "model": "mistral"
        }
    }), encoding="utf-8")
    service.reload()
    assert set(service.get_assistants().keys()) == {"b"}


def test_invalid_entries_skipped(tmp_path: Path):
    p = tmp_path / "assistants.json"
    # second entry invalid (missing provider/model)
    p.write_text(json.dumps({
        "valid": {
            "name": "Ok",
            "description": "desc",
            "provider": "ollama",
            "model": "mistral"
        },
        "invalid": {
            "name": "Bad",
            "description": "desc"
        }
    }), encoding="utf-8")
    service = AssistantService(config_path=p)
    assistants = service.get_assistants()
    assert set(assistants.keys()) == {"valid"}
