from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from globalbot.backend.base import AIMessage
from globalbot.backend.llms.completions import init_completion_model, ONLINE_PROVIDERS, OFFLINE_PROVIDERS
from globalbot.backend.llms.completions.base import BaseCompletionLLM


class _EchoCompletion(BaseCompletionLLM):
    def _call(self, prompts, **kwargs):
        return AIMessage(content=f"echo: {prompts[0]}", total_tokens=5, prompt_tokens=3, completion_tokens=2)


class TestBaseCompletionLLM:
    def test_run_string(self):
        llm = _EchoCompletion()
        result = llm.run("hello")
        assert isinstance(result, AIMessage)
        assert result.text == "echo: hello"

    def test_run_list(self):
        llm = _EchoCompletion()
        result = llm.run(["hello", "world"])
        assert isinstance(result, AIMessage)

    def test_stream(self):
        llm = _EchoCompletion()
        chunks = list(llm.stream("test"))
        assert len(chunks) >= 1
        assert isinstance(chunks[-1], AIMessage)

    @pytest.mark.asyncio
    async def test_arun(self):
        llm = _EchoCompletion()
        result = await llm.arun("hello")
        assert isinstance(result, AIMessage)
        assert result.text == "echo: hello"


class TestInitCompletionModel:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown completion provider"):
            init_completion_model("unknown_xyz")

    def test_online_providers_set(self):
        assert "openai" in ONLINE_PROVIDERS

    def test_offline_providers_set(self):
        assert "vllm" in OFFLINE_PROVIDERS

    @patch("globalbot.backend.llms.completions.openai.OpenAI")
    def test_init_openai(self, mock_openai):
        mock_openai.return_value = MagicMock()
        llm = init_completion_model("openai", api_key="test-key")
        assert "openai-completion" in llm.name

    @patch("globalbot.backend.llms.completions.vllm.OpenAI")
    def test_init_vllm(self, mock_openai):
        mock_openai.return_value = MagicMock()
        llm = init_completion_model("vllm", model="llama3", base_url="http://localhost:8000/v1")
        assert "vllm-completion" in llm.name


class TestOpenAICompletion:
    @patch("globalbot.backend.llms.completions.openai.OpenAI")
    def test_call(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].text = "completed text"
        mock_response.usage.total_tokens = 10
        mock_response.usage.prompt_tokens = 7
        mock_response.usage.completion_tokens = 3
        mock_client.completions.create.return_value = mock_response

        from globalbot.backend.llms.completions.openai import OpenAICompletion
        llm = OpenAICompletion(api_key="test")
        result = llm.run("prompt text")
        assert result.text == "completed text"
        assert result.total_tokens == 10

    @patch("globalbot.backend.llms.completions.openai.OpenAI")
    def test_stream(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.choices[0].text = "comp"
        chunk2 = MagicMock()
        chunk2.choices[0].text = "leted"
        mock_client.completions.create.return_value = iter([chunk1, chunk2])

        from globalbot.backend.llms.completions.openai import OpenAICompletion
        llm = OpenAICompletion(api_key="test")
        chunks = list(llm.stream("hello"))
        assert chunks[-1].text == "completed"


class TestVLLMCompletion:
    @patch("globalbot.backend.llms.completions.vllm.OpenAI")
    def test_call(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].text = "vllm completed"
        mock_response.usage.total_tokens = 8
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_client.completions.create.return_value = mock_response

        from globalbot.backend.llms.completions.vllm import VLLMCompletion
        llm = VLLMCompletion(model="llama3", base_url="http://localhost:8000/v1")
        result = llm.run("prompt")
        assert result.text == "vllm completed"
        assert result.total_tokens == 8