from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from globalbot.backend.base import AIMessage, HumanMessage, SystemMessage
from globalbot.backend.llms.chats.base import _normalise, _to_openai_messages, BaseChatLLM
from globalbot.backend.llms.chats import init_chat_model, ONLINE_PROVIDERS, OFFLINE_PROVIDERS


class _EchoChat(BaseChatLLM):
    def _call(self, messages, **kwargs):
        text = " | ".join(m.text for m in messages)
        return AIMessage(content=f"echo: {text}", total_tokens=10, prompt_tokens=7, completion_tokens=3)


class TestNormalise:
    def test_string(self):
        result = _normalise(["hello"])
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].text == "hello"

    def test_dict_user(self):
        result = _normalise([{"role": "user", "content": "hi"}])
        assert isinstance(result[0], HumanMessage)

    def test_dict_assistant(self):
        result = _normalise([{"role": "assistant", "content": "hello"}])
        assert isinstance(result[0], AIMessage)

    def test_dict_system(self):
        result = _normalise([{"role": "system", "content": "you are a bot"}])
        assert isinstance(result[0], SystemMessage)

    def test_passthrough_messages(self):
        msgs = [HumanMessage(content="a"), AIMessage(content="b")]
        result = _normalise(msgs)
        assert result == msgs


class TestToOpenAIMessages:
    def test_role_mapping(self):
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi"), AIMessage(content="bye")]
        result = _to_openai_messages(msgs)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_content(self):
        msgs = [HumanMessage(content="test content")]
        result = _to_openai_messages(msgs)
        assert result[0]["content"] == "test content"


class TestBaseChatLLM:
    def test_run_string_input(self):
        llm = _EchoChat()
        result = llm.run("hello")
        assert isinstance(result, AIMessage)
        assert "echo:" in result.text

    def test_run_list_input(self):
        llm = _EchoChat()
        result = llm.run([HumanMessage(content="test")])
        assert isinstance(result, AIMessage)

    def test_stream(self):
        llm = _EchoChat()
        chunks = list(llm.stream("hello"))
        assert len(chunks) >= 1
        assert isinstance(chunks[-1], AIMessage)

    def test_chat_shortcut(self):
        llm = _EchoChat()
        text = llm.chat("hello", system="you are a bot")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_token_tracking(self):
        llm = _EchoChat()
        result = llm.run("hi")
        assert result.total_tokens == 10
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 3

    @pytest.mark.asyncio
    async def test_arun(self):
        llm = _EchoChat()
        result = await llm.arun("hello")
        assert isinstance(result, AIMessage)


class TestInitChatModel:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            init_chat_model("unknown_xyz")

    def test_online_providers_set(self):
        assert "openai" in ONLINE_PROVIDERS
        assert "anthropic" in ONLINE_PROVIDERS
        assert "google" in ONLINE_PROVIDERS

    def test_offline_providers_set(self):
        assert "ollama" in OFFLINE_PROVIDERS
        assert "vllm" in OFFLINE_PROVIDERS
        assert "hf" in OFFLINE_PROVIDERS
        assert "onnx" in OFFLINE_PROVIDERS

    @patch("globalbot.backend.llms.chats.openai.OpenAI")
    def test_init_openai(self, mock_openai):
        mock_openai.return_value = MagicMock()
        llm = init_chat_model("openai", api_key="test-key", model="gpt-4o-mini")
        assert llm.name == "openai/gpt-4o-mini"

    def test_init_openai_with_langchain(self):
        import sys
        mock_lc_cls = MagicMock()
        mock_lc_cls.return_value = MagicMock()
        mock_langchain_openai = MagicMock()
        mock_langchain_openai.ChatOpenAI = mock_lc_cls

        with patch.dict(sys.modules, {"langchain_openai": mock_langchain_openai}):
            from importlib import reload
            import globalbot.backend.llms.chats.langchain_based as lb_mod
            reload(lb_mod)
            llm = lb_mod.LCChatOpenAI(api_key="key", model="gpt-4o-mini")
            assert "lc/openai" in llm.name

    @patch("globalbot.backend.llms.chats.ollama.Client")
    def test_init_ollama(self, mock_client):
        llm = init_chat_model("ollama", model="llama3.1:8b")
        assert "ollama" in llm.name

    @patch("globalbot.backend.llms.chats.vllm.OpenAI")
    def test_init_vllm(self, mock_openai):
        mock_openai.return_value = MagicMock()
        llm = init_chat_model("vllm", model="llama3", base_url="http://localhost:8000/v1")
        assert "vllm" in llm.name


class TestChatOpenAI:
    @patch("globalbot.backend.llms.chats.openai.OpenAI")
    def test_call(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage.total_tokens = 20
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        from git.GlobalBot.globalbot.backend.llms.chats.openai_impl import ChatOpenAI
        llm = ChatOpenAI(api_key="test", model="gpt-4o-mini")
        result = llm.run("Hi")
        assert result.text == "Hello!"
        assert result.total_tokens == 20

    @patch("globalbot.backend.llms.chats.openai.OpenAI")
    def test_stream(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hel"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = "lo!"
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

        from git.GlobalBot.globalbot.backend.llms.chats.openai_impl import ChatOpenAI
        llm = ChatOpenAI(api_key="test", model="gpt-4o-mini")
        chunks = list(llm.stream("Hi"))
        assert chunks[-1].text == "Hello!"


class TestChatVLLM:
    @patch("globalbot.backend.llms.chats.vllm.OpenAI")
    def test_call(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "vllm response"
        mock_response.usage.total_tokens = 15
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        from globalbot.backend.llms.chats.vllm import ChatVLLM
        llm = ChatVLLM(model="llama3", base_url="http://localhost:8000/v1")
        result = llm.run("test")
        assert result.text == "vllm response"