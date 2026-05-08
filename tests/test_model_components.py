"""Unit tests for model components — RMSNorm, RoPE, AttentionGQA, SwiGLU_FFN, TransformerBlock, Decoder."""

import torch
import pytest
from src.config import ModelConfig
from src.model import RMSNorm, RoPE, AttentionGQA, SwiGLU_FFN, TransformerBlock, Decoder
from src.paged_cache import BlockTable, BlockAllocator

@pytest.fixture
def config():
    """LLaMA 3.2 3B config with reduced layers/context for fast testing."""
    cfg = ModelConfig.llama_3_2_3b()
    cfg.num_layers = 2
    cfg.context_length = 128
    return cfg


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=3072, eps=1e-5)
        x = torch.randn(2, 10, 3072)
        out = norm(x)
        assert out.shape == x.shape

    def test_gamma_shape(self):
        norm = RMSNorm(dim=3072, eps=1e-5)
        assert norm.gamma.shape == (3072,)

    def test_gamma_initialized_to_ones(self):
        norm = RMSNorm(dim=3072, eps=1e-5)
        assert torch.allclose(norm.gamma, torch.ones(3072))


class TestRoPE:
    def test_table_shapes(self):
        rope = RoPE(head_dim=128, max_seq_len=1024, rope_theta=500000.0)
        assert rope.costable.shape == (1024, 128)
        assert rope.sintable.shape == (1024, 128)

    def test_output_shape(self):
        rope = RoPE(head_dim=128, max_seq_len=1024, rope_theta=500000.0)
        x = torch.randn(2, 24, 5, 128)  # (B, n_heads, T, head_dim)
        positions = torch.arange(5)
        out = rope(x, positions)
        assert out.shape == x.shape

    def test_different_positions(self):
        rope = RoPE(head_dim=128, max_seq_len=1024, rope_theta=500000.0)
        x = torch.randn(2, 24, 3, 128)
        pos_a = torch.tensor([0, 1, 2])
        pos_b = torch.tensor([10, 11, 12])
        out_a = rope(x, pos_a)
        out_b = rope(x, pos_b)
        # Different positions should produce different outputs
        assert not torch.allclose(out_a, out_b)


class TestAttentionGQA:
    def test_prefill(self):
        attn = AttentionGQA(
            num_heads=24, d_model=3072, num_kv_heads=8,
            head_dim=128, max_seq_len=1024, rope_theta=500000.0
        )
        x = torch.randn(2, 5, 3072)
        positions = torch.arange(5)
        output, cache = attn(x, positions)
        assert output.shape == (2, 5, 3072)
        assert cache[0].shape == (2, 8, 5, 128)  # K cache: n_kv_heads, not n_heads
        assert cache[1].shape == (2, 8, 5, 128)  # V cache

    def test_decode_with_cache(self):
        attn = AttentionGQA(
            num_heads=24, d_model=3072, num_kv_heads=8,
            head_dim=128, max_seq_len=1024, rope_theta=500000.0
        )
        # Prefill
        x = torch.randn(2, 5, 3072)
        positions = torch.arange(5)
        _, cache = attn(x, positions)

        # Decode
        x2 = torch.randn(2, 1, 3072)
        positions2 = torch.tensor([5])
        output2, cache2 = attn(x2, positions2, cache)
        assert output2.shape == (2, 1, 3072)
        assert cache2[0].shape == (2, 8, 6, 128)  # cache grew by 1


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU_FFN(d_model=3072, d_ff=8192)
        x = torch.randn(2, 5, 3072)
        out = ffn(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    def test_prefill(self, config):
        block = TransformerBlock(config)
        x = torch.randn(2, 5, 3072)
        positions = torch.arange(5)
        out, cache = block(x, positions)
        assert out.shape == (2, 5, 3072)

    def test_decode_with_cache(self, config):
        block = TransformerBlock(config)
        x = torch.randn(2, 5, 3072)
        positions = torch.arange(5)
        _, cache = block(x, positions)

        x2 = torch.randn(2, 1, 3072)
        positions2 = torch.tensor([5])
        out2, cache2 = block(x2, positions2, cache)
        assert out2.shape == (2, 1, 3072)


class TestDecoder:
    def test_prefill(self, config):
        model = Decoder(config)
        token_ids = torch.randint(0, config.vocab_size, (1, 5))
        positions = torch.arange(5)
        logits, caches = model(token_ids, positions)
        assert logits.shape == (1, 5, config.vocab_size)
        assert len(caches) == config.num_layers

    def test_decode_with_cache(self, config):
        model = Decoder(config)
        # Prefill
        token_ids = torch.randint(0, config.vocab_size, (1, 5))
        positions = torch.arange(5)
        _, caches = model(token_ids, positions)

        # Decode
        token_ids2 = torch.randint(0, config.vocab_size, (1, 1))
        positions2 = torch.tensor([5])
        logits2, caches2 = model(token_ids2, positions2, caches)
        assert logits2.shape == (1, 1, config.vocab_size)
        assert caches2[0][0].shape[2] == 6  # cache grew by 1







