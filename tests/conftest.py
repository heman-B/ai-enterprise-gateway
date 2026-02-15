# tests/conftest.py
# Pytest-Konfiguration und gemeinsame Fixtures
import pytest
from gateway.models import ChatMessage, ChatCompletionRequest


@pytest.fixture
def sample_messages():
    """Standardmäßige Test-Nachrichten."""
    return [ChatMessage(role="user", content="Was ist die Hauptstadt von Deutschland?")]


@pytest.fixture
def sample_request(sample_messages):
    """Standard-Testanfrage mit minimalem Inhalt."""
    return ChatCompletionRequest(messages=sample_messages)
