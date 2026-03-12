"""Integration tests for SSE chat endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
class TestChatAPI:
    async def test_post_chat_returns_job_id(self):
        with patch("app.services.job_manager.create_job", new_callable=AsyncMock) as m:
            m.return_value = "abc123"
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                res = await ac.post(
                    "/api/chat",
                    json={"model_id": "test-model", "prompt": "hi"},
                )
            assert res.status_code == 200
            assert res.json() == {"job_id": "abc123"}

    async def test_post_chat_validates_body(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            res = await ac.post("/api/chat", json={"bad": "body"})
        assert res.status_code == 422

    async def test_cancel_returns_status(self):
        with patch("app.services.job_manager.cancel_job", return_value=True):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                res = await ac.post("/api/chat/someid/cancel")
            assert res.json()["status"] == "cancelled"

    async def test_cancel_not_found(self):
        with patch("app.services.job_manager.cancel_job", return_value=False):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                res = await ac.post("/api/chat/nope/cancel")
            assert res.json()["status"] == "not_found"


@pytest.mark.asyncio
class TestSSEStream:
    async def test_sse_streams_tokens(self):
        async def fake_stream(job_id):
            yield {"event": "token", "data": "Hello"}
            yield {"event": "token", "data": " world"}
            yield {"event": "done", "data": ""}

        with patch("app.services.job_manager.token_stream", side_effect=fake_stream):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                res = await ac.get("/sse/chat/test123")

            assert res.status_code == 200
            assert "text/event-stream" in res.headers["content-type"]
            body = res.text
            assert "event: token\ndata: Hello" in body
            assert "event: done" in body

    async def test_sse_streams_error(self):
        async def fake_stream(job_id):
            yield {"event": "error", "data": "Failed to create llama_context"}

        with patch("app.services.job_manager.token_stream", side_effect=fake_stream):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                res = await ac.get("/sse/chat/bad123")

            body = res.text
            assert "event: error" in body
            assert "llama_context" in body


@pytest.mark.asyncio
class TestCancelDuringStream:
    async def test_cancel_mid_stream_truncates_output(self):
        """SSE stream should end early when cancel is called mid-generation."""
        cancelled = False

        async def fake_stream(job_id):
            nonlocal cancelled
            yield {"event": "token", "data": "Hello"}
            yield {"event": "token", "data": " world"}
            if cancelled:
                yield {"event": "done", "data": ""}
                return
            yield {"event": "token", "data": " more"}
            yield {"event": "done", "data": ""}

        with (
            patch(
                "app.services.job_manager.token_stream", side_effect=fake_stream
            ),
            patch(
                "app.services.job_manager.cancel_job", return_value=True
            ) as mock_cancel,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                cancel_res = await ac.post("/api/chat/job1/cancel")
                assert cancel_res.json()["status"] == "cancelled"
                mock_cancel.assert_called_once_with("job1")

                cancelled = True
                sse_res = await ac.get("/sse/chat/job1")

            body = sse_res.text
            assert "event: token\ndata: Hello" in body
            assert "event: done" in body
            assert "more" not in body

    async def test_cancel_returns_cancelled_for_active_job(self):
        with patch("app.services.job_manager.cancel_job", return_value=True):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                res = await ac.post("/api/chat/active-job/cancel")
            assert res.status_code == 200
            assert res.json() == {"status": "cancelled"}
