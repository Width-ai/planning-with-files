"""Tests for the agentic tool-use flow.

Covers tool handlers (api.tools), the agentic loop (api.claude_client),
and the FastAPI /query endpoint (api.main).
"""

import tempfile

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from api.tools import (
    handle_list_files,
    handle_search_files,
    handle_read_file,
    handle_write_file,
    execute_tool,
)


# ===================================================================
# 1. Tool Handler Tests
# ===================================================================


class TestListFiles:
    def test_lists_txt_files(self, tmp_path):
        (tmp_path / "page_001_test.txt").write_text("content1")
        (tmp_path / "page_002_other.txt").write_text("content2")
        (tmp_path / "not_txt.md").write_text("ignored")
        result = handle_list_files(str(tmp_path))
        assert "page_001_test.txt" in result
        assert "page_002_other.txt" in result
        assert "not_txt.md" not in result

    def test_pattern_filter(self, tmp_path):
        (tmp_path / "page_001_ariana.txt").write_text("c1")
        (tmp_path / "page_002_bob.txt").write_text("c2")
        result = handle_list_files(str(tmp_path), pattern="*ariana*")
        assert "ariana" in result
        assert "bob" not in result

    def test_missing_folder(self):
        result = handle_list_files("/nonexistent/path")
        assert "does not exist" in result.lower() or "error" in result.lower()

    def test_no_matches(self, tmp_path):
        result = handle_list_files(str(tmp_path))
        assert "no" in result.lower()


class TestSearchFiles:
    def test_finds_term_in_content(self, tmp_path):
        (tmp_path / "page_001.txt").write_text("Ariana worked overtime on Tuesday")
        (tmp_path / "page_002.txt").write_text("Bob had no overtime")
        result = handle_search_files(str(tmp_path), "Ariana")
        assert "page_001" in result

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "page_001.txt").write_text("OVERTIME was discussed")
        result = handle_search_files(str(tmp_path), "overtime")
        assert "page_001" in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "page_001.txt").write_text("nothing relevant")
        result = handle_search_files(str(tmp_path), "nonexistent_term_xyz")
        assert "no" in result.lower() or "0" in result


class TestReadFile:
    def test_reads_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world content")
        result = handle_read_file(str(f))
        assert "Hello world content" in result

    def test_truncates_long_files(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 6000)
        result = handle_read_file(str(f))
        assert "truncated" in result.lower()
        # Content is truncated at 5000 chars plus the truncation notice
        assert len(result) < 6000

    def test_missing_file(self):
        result = handle_read_file("/nonexistent/file.txt")
        assert "error" in result.lower() or "not found" in result.lower()


class TestWriteFile:
    def test_writes_content(self, tmp_path):
        result = handle_write_file("notes.md", "hello world", work_dir=tmp_path)
        assert "Wrote" in result
        assert (tmp_path / "notes.md").read_text() == "hello world"

    def test_append_mode(self, tmp_path):
        (tmp_path / "findings.md").write_text("line1\n")
        result = handle_write_file("findings.md", "line2\n", mode="append", work_dir=tmp_path)
        assert "append" in result
        assert (tmp_path / "findings.md").read_text() == "line1\nline2\n"

    def test_rejects_absolute_path(self, tmp_path):
        result = handle_write_file("/etc/passwd", "bad", work_dir=tmp_path)
        assert "error" in result.lower()

    def test_rejects_path_traversal(self, tmp_path):
        result = handle_write_file("../../etc/passwd", "bad", work_dir=tmp_path)
        assert "error" in result.lower()

    def test_rejects_no_work_dir(self):
        result = handle_write_file("notes.md", "hello")
        assert "error" in result.lower()

    def test_creates_subdirectories(self, tmp_path):
        result = handle_write_file("sub/dir/notes.md", "nested", work_dir=tmp_path)
        assert "Wrote" in result
        assert (tmp_path / "sub" / "dir" / "notes.md").read_text() == "nested"


class TestExecuteTool:
    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {})
        assert "unknown" in result.lower()

    def test_dispatches_list_files(self, tmp_path):
        (tmp_path / "page_001.txt").write_text("content")
        result = execute_tool("list_files", {"folder_path": str(tmp_path)})
        assert "page_001.txt" in result

    def test_dispatches_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("read me")
        result = execute_tool("read_file", {"file_path": str(f)})
        assert "read me" in result

    def test_dispatches_search_files(self, tmp_path):
        (tmp_path / "page_001.txt").write_text("findable term here")
        result = execute_tool("search_files", {"folder_path": str(tmp_path), "search_term": "findable"})
        assert "page_001" in result

    def test_dispatches_write_file(self, tmp_path):
        result = execute_tool(
            "write_file",
            {"file_path": "test.md", "content": "hello"},
            work_dir=tmp_path,
        )
        assert "Wrote" in result
        assert (tmp_path / "test.md").read_text() == "hello"


# ===================================================================
# 2. Claude Client Agentic Loop Tests (mocked)
# ===================================================================


from api.claude_client import ask_claude, ClaudeAPIError, APIKeyMissingError, QueryStats


class TestAskClaude:
    @patch("api.claude_client.get_client")
    def test_simple_text_response(self, mock_get_client, tmp_path):
        """Test when Claude answers immediately without tool calls."""
        (tmp_path / "test.txt").write_text("content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Based on the documents, the answer is 42."
        mock_response.content = [mock_text_block]

        mock_client.messages.create.return_value = mock_response

        answer, sources, stats = ask_claude("What is the answer?", str(tmp_path))
        assert "42" in answer
        assert isinstance(sources, list)
        assert stats.api_calls == 1
        assert stats.tool_rounds == 0

    @patch("api.claude_client.get_client")
    def test_tool_use_then_answer(self, mock_get_client, tmp_path):
        """Test multi-turn: Claude calls a tool, then answers."""
        (tmp_path / "page_001_test.txt").write_text("test content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First response: Claude calls list_files
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "list_files"
        tool_block.id = "tool_1"
        tool_block.input = {"folder_path": str(tmp_path)}

        response1 = MagicMock()
        response1.stop_reason = "tool_use"
        response1.content = [tool_block]

        # Second response: Claude answers
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer based on the files."

        response2 = MagicMock()
        response2.stop_reason = "end_turn"
        response2.content = [text_block]

        mock_client.messages.create.side_effect = [response1, response2]

        answer, sources, stats = ask_claude("query", str(tmp_path))
        assert "answer" in answer.lower()
        assert stats.api_calls == 2
        assert stats.tool_rounds == 1
        assert stats.tool_calls == {"list_files": 1}

    @patch("api.claude_client.get_client")
    def test_read_file_tracks_sources(self, mock_get_client, tmp_path):
        """Test that read_file calls add to sources list."""
        (tmp_path / "page_001_ariana.txt").write_text("Ariana's content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First response: Claude calls read_file
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "read_file"
        tool_block.id = "tool_1"
        tool_block.input = {"file_path": str(tmp_path / "page_001_ariana.txt")}

        response1 = MagicMock()
        response1.stop_reason = "tool_use"
        response1.content = [tool_block]

        # Second response: answer
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Ariana worked overtime."

        response2 = MagicMock()
        response2.stop_reason = "end_turn"
        response2.content = [text_block]

        mock_client.messages.create.side_effect = [response1, response2]

        answer, sources, stats = ask_claude("query", str(tmp_path))
        assert "page_001_ariana.txt" in sources
        assert stats.tool_calls == {"read_file": 1}

    @patch("api.claude_client.get_client")
    def test_write_file_in_loop(self, mock_get_client, tmp_path):
        """Test that write_file tool calls work in the agentic loop."""
        (tmp_path / "page_001.txt").write_text("content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First response: Claude calls write_file
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "write_file"
        tool_block.id = "tool_1"
        tool_block.input = {"file_path": "plan.md", "content": "Search for overtime"}

        response1 = MagicMock()
        response1.stop_reason = "tool_use"
        response1.content = [tool_block]

        # Second response: answer
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Done."

        response2 = MagicMock()
        response2.stop_reason = "end_turn"
        response2.content = [text_block]

        mock_client.messages.create.side_effect = [response1, response2]

        answer, sources, stats = ask_claude("query", str(tmp_path))
        assert answer == "Done."
        # write_file should not add to sources
        assert sources == []
        assert stats.tool_calls == {"write_file": 1}

    @patch("api.claude_client.get_client")
    def test_work_dir_cleaned_up(self, mock_get_client, tmp_path):
        """Test that the temp working dir is cleaned up after the loop."""
        (tmp_path / "page_001.txt").write_text("content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "answer"
        mock_response.content = [mock_text_block]

        mock_client.messages.create.return_value = mock_response

        # Capture the work_dir by patching tempfile.mkdtemp
        created_dirs = []
        original_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        with patch("api.claude_client.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            ask_claude("query", str(tmp_path))

        # The temp dir should have been cleaned up
        assert len(created_dirs) == 1
        assert not Path(created_dirs[0]).exists()

    @patch("api.claude_client.get_client")
    def test_multiple_tool_calls(self, mock_get_client, tmp_path):
        """Test that multiple tool calls in one response are all processed."""
        (tmp_path / "page_001.txt").write_text("content one")
        (tmp_path / "page_002.txt").write_text("content two")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First response: Claude calls read_file twice
        tool_block1 = MagicMock()
        tool_block1.type = "tool_use"
        tool_block1.name = "read_file"
        tool_block1.id = "tool_1"
        tool_block1.input = {"file_path": str(tmp_path / "page_001.txt")}

        tool_block2 = MagicMock()
        tool_block2.type = "tool_use"
        tool_block2.name = "read_file"
        tool_block2.id = "tool_2"
        tool_block2.input = {"file_path": str(tmp_path / "page_002.txt")}

        response1 = MagicMock()
        response1.stop_reason = "tool_use"
        response1.content = [tool_block1, tool_block2]

        # Second response: answer
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Both files read."

        response2 = MagicMock()
        response2.stop_reason = "end_turn"
        response2.content = [text_block]

        mock_client.messages.create.side_effect = [response1, response2]

        answer, sources, stats = ask_claude("query", str(tmp_path))
        assert "page_001.txt" in sources
        assert "page_002.txt" in sources
        assert stats.tool_calls == {"read_file": 2}


# ===================================================================
# 3. FastAPI Endpoint Tests
# ===================================================================


from fastapi.testclient import TestClient
from api.main import app


class TestQueryEndpoint:
    @patch("api.main.ask_claude")
    def test_successful_query(self, mock_ask, tmp_path):
        stats = QueryStats(api_calls=2, tool_rounds=1, input_tokens=100, output_tokens=50)
        stats.tool_calls = {"list_files": 1}
        mock_ask.return_value = ("The answer is here.", ["page_001.txt"], stats)
        client = TestClient(app)
        response = client.post("/query", json={
            "query": "What happened?",
            "folder_path": str(tmp_path),
        })
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The answer is here."
        assert "page_001.txt" in data["sources"]
        assert data["stats"]["api_calls"] == 2
        assert data["stats"]["tool_rounds"] == 1
        assert data["stats"]["input_tokens"] == 100
        assert data["stats"]["output_tokens"] == 50
        assert data["stats"]["total_tokens"] == 150
        assert data["stats"]["tool_calls"] == {"list_files": 1}

    def test_missing_folder(self):
        client = TestClient(app)
        response = client.post("/query", json={
            "query": "test",
            "folder_path": "/nonexistent/folder",
        })
        assert response.status_code == 400

    def test_empty_query(self):
        client = TestClient(app)
        response = client.post("/query", json={
            "query": "",
            "folder_path": "/tmp",
        })
        assert response.status_code == 422  # Pydantic validation error

    def test_health(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @patch("api.main.ask_claude")
    def test_api_key_missing_returns_500(self, mock_ask, tmp_path):
        mock_ask.side_effect = APIKeyMissingError("ANTHROPIC_API_KEY not set")
        client = TestClient(app)
        response = client.post("/query", json={
            "query": "question",
            "folder_path": str(tmp_path),
        })
        assert response.status_code == 500

    @patch("api.main.ask_claude")
    def test_claude_api_error_returns_502(self, mock_ask, tmp_path):
        mock_ask.side_effect = ClaudeAPIError("Rate limit exceeded")
        client = TestClient(app)
        response = client.post("/query", json={
            "query": "question",
            "folder_path": str(tmp_path),
        })
        assert response.status_code == 502
