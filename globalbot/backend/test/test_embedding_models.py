from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from globalbot.backend.embeddings import init_embedding_model, ONLINE_PROVIDERS, OFFLINE_PROVIDERS
from globalbot.backend.embeddings.base import BaseEmbeddings


class _DummyEmbeddings(BaseEmbeddings):
    dim: int = 4

    def _embed_documents(self, texts, **kwargs):
        return [[float(i) * 0.1] * self.dim for i in range(len(texts))]

    def _embed_query(self, text, **kwargs):
        return [0.1, 0.2, 0.3, 0.4]


class TestBaseEmbeddings:
    def test_run_returns_list_of_vectors(self):
        emb = _DummyEmbeddings()
        result = emb.run(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 4

    def test_embed_query(self):
        emb = _DummyEmbeddings()
        result = emb.embed_query("hello")
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_arun(self):
        emb = _DummyEmbeddings()
        result = await emb.arun(["hello"])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_aembed_query(self):
        emb = _DummyEmbeddings()
        result = await emb.aembed_query("hello")
        assert len(result) == 4

    def test_as_langchain_adapter(self):
        emb = _DummyEmbeddings()
        lc = emb.as_langchain()
        assert hasattr(lc, "embed_documents")
        assert hasattr(lc, "embed_query")
        result = lc.embed_documents(["a", "b"])
        assert len(result) == 2
        q = lc.embed_query("test")
        assert len(q) == 4


class TestInitEmbeddingModel:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            init_embedding_model("unknown_xyz")

    def test_online_providers_set(self):
        assert "openai" in ONLINE_PROVIDERS
        assert "azure" in ONLINE_PROVIDERS
        assert "cohere" in ONLINE_PROVIDERS

    def test_offline_providers_set(self):
        assert "ollama" in OFFLINE_PROVIDERS
        assert "hf" in OFFLINE_PROVIDERS
        assert "fastembed" in OFFLINE_PROVIDERS

    @patch("globalbot.backend.embeddings.openai.OpenAI")
    def test_init_openai(self, mock_openai):
        mock_openai.return_value = MagicMock()
        emb = init_embedding_model("openai", api_key="test-key", model="text-embedding-3-small")
        assert "openai" in emb.name

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

    def test_init_ollama(self):
        emb = init_embedding_model("ollama", model="nomic-embed-text")
        assert "ollama" in emb.name


class TestOpenAIEmbeddings:
    @patch("globalbot.backend.embeddings.openai.OpenAI")
    def test_embed_documents(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]), MagicMock(embedding=[0.4, 0.5, 0.6])]
        mock_client.embeddings.create.return_value = mock_response

        from globalbot.backend.embeddings.openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(api_key="test")
        result = emb.run(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("globalbot.backend.embeddings.openai.OpenAI")
    def test_embed_query(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        from globalbot.backend.embeddings.openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(api_key="test")
        result = emb.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]

    @patch("globalbot.backend.embeddings.openai.OpenAI")
    def test_dimensions_param(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response

        from globalbot.backend.embeddings.openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(api_key="test", dimensions=512)
        emb.embed_query("hello")
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs.get("dimensions") == 512


class TestFastEmbedEmbeddings:
    def test_embed_documents(self):
        import sys
        import numpy as np
        mock_instance = MagicMock()
        mock_instance.embed.return_value = iter([np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])])
        mock_fastembed = MagicMock()
        mock_fastembed.TextEmbedding.return_value = mock_instance

        with patch.dict(sys.modules, {"fastembed": mock_fastembed}):
            from importlib import reload
            import globalbot.backend.embeddings.fastEmbed as fe_mod
            reload(fe_mod)
            emb = fe_mod.FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5")
            result = emb.run(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_embed_query(self):
        import sys
        import numpy as np
        mock_instance = MagicMock()
        mock_instance.embed.return_value = iter([np.array([0.1, 0.2, 0.3])])
        mock_fastembed = MagicMock()
        mock_fastembed.TextEmbedding.return_value = mock_instance

        with patch.dict(sys.modules, {"fastembed": mock_fastembed}):
            from importlib import reload
            import globalbot.backend.embeddings.fastEmbed as fe_mod
            reload(fe_mod)
            emb = fe_mod.FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5")
            result = emb.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]