from pathlib import Path

import pytest

ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def test_audio():
    """Предоставляет тестовый аудиофайл для всех тестов"""
    return str(ASSETS_DIR / "example.wav")


@pytest.fixture(scope="session")
def long_audio():
    """Предоставляет длинный тестовый аудиофайл для longform-тестов"""
    return str(ASSETS_DIR / "long_example.wav")
