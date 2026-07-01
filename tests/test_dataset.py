import pytest
from src.data.loader import PDNCLoader
from pathlib import Path
from src.utils.config import get_data_dir

def test_loader_initialization():
    loader = PDNCLoader()
    assert loader.data_dir == get_data_dir()
    assert loader.quotes_file == get_data_dir() / "quotes.json"

def test_loader_handles_missing_file(caplog):
    loader = PDNCLoader("data/nonexistent")
    list(loader.load())
    assert "Quotes file not found" in caplog.text

def test_loader_loads_correctly(tmp_path):
    quotes_file = tmp_path / "quotes.json"
    quotes_file.write_text('[{"text": "Hello", "speaker": "Alice"}, {"text": "World"}]')
    loader = PDNCLoader(tmp_path)
    quotes = list(loader.load())
    assert len(quotes) == 2
    assert quotes[0]["speaker"] == "Alice"
    assert "speaker" not in quotes[1] # missing field handled by just not being there

def test_loader_handles_malformed_file(tmp_path, caplog):
    quotes_file = tmp_path / "quotes.json"
    quotes_file.write_text('[{malformed json')
    loader = PDNCLoader(tmp_path)
    list(loader.load())
    assert "Failed to parse JSON" in caplog.text

def test_loader_handles_empty_file(tmp_path, caplog):
    quotes_file = tmp_path / "quotes.json"
    quotes_file.write_text('')
    loader = PDNCLoader(tmp_path)
    list(loader.load())
    assert "Failed to parse JSON" in caplog.text
