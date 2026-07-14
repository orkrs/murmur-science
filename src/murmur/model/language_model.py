"""Murmur causal language model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from murmur.model.block import TransformerBlock
from murmur.model.cache import MurmurCache
from murmur.model.config import ModelConfig
from murmur.model.norms import RMSNorm
from murmur.model.recurrent import RecurrentCore


@dataclass
class CausalLMOutput:
    """Output from causal language model.

    Attributes:
        logits: Language model logits [batch, seq_len, vocab_size]
        loss: Optional cross-entropy loss
        depths: Per-sequence recurrent depths
        diagnostics: Optional diagnostic information
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    depths: torch.Tensor | None = None
    diagnostics: dict | None = None


class MurmurForCausalLM(nn.Module):
    """Murmur-RSM causal language model.

    Architecture:
        embedding → Prelude → LN(e) → Core×T → Coda → final RMSNorm → tied head
    """

    def __init__(self, config: ModelConfig):
        """Initialize model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.prelude = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.q_heads, config.kv_heads, config.head_dim,
                config.ffn_dim, config.max_seq_len, config.rope_theta,
                config.norm_eps, config.residual_scale
            )
            for _ in range(config.n_prelude)
        ])

        self.core = RecurrentCore(
            config.n_core, config.d_model, config.q_heads, config.kv_heads,
            config.head_dim, config.ffn_dim, config.max_seq_len,
            config.rope_theta, config.norm_eps, config.residual_scale, config.mixer
        )

        self.coda = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.q_heads, config.kv_heads, config.head_dim,
                config.ffn_dim, config.max_seq_len, config.rope_theta,
                config.norm_eps, config.residual_scale
            )
            for _ in range(config.n_coda)
        ])

        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.injection_norm = RMSNorm(config.d_model, config.norm_eps)

        if config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        depths: torch.Tensor | None = None,
        cache: MurmurCache | None = None,
        use_cache: bool = False,
        positions: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            labels: Target token IDs for loss [batch, seq_len]
            depths: Per-sequence recurrent depths [batch]

        Returns:
            CausalLMOutput with logits and optional loss
        """
        batch, seq_len = input_ids.shape

        if depths is None:
            depths = torch.full((batch,), self.config.min_depth, device=input_ids.device)
        depths = depths.to(device=input_ids.device, dtype=torch.long)
        if torch.any(depths < self.config.min_depth) or torch.any(depths > self.config.max_depth):
            raise ValueError("depths must be within [min_depth, max_depth]")
        if use_cache and cache is None:
            cache = MurmurCache.empty(
                self.config.n_prelude,
                self.config.n_core,
                self.config.max_depth,
                self.config.n_coda,
            )

        hidden = self.token_embedding(input_ids)

        new_prelude = []
        for block_idx, block in enumerate(self.prelude):
            block_cache = cache.prelude[block_idx] if cache is not None else None
            hidden, block_cache = block(
                hidden, positions, block_cache, use_cache
            )
            new_prelude.append(block_cache)

        injection = self.injection_norm(hidden)
        core_result = self.core(
            hidden,
            injection,
            depths,
            positions,
            cache.core if cache is not None else None,
            use_cache,
        )
        if use_cache:
            hidden, new_core = core_result
        else:
            hidden = core_result
            new_core = []

        new_coda = []
        for block_idx, block in enumerate(self.coda):
            block_cache = cache.coda[block_idx] if cache is not None else None
            hidden, block_cache = block(hidden, positions, block_cache, use_cache)
            new_coda.append(block_cache)

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        new_cache = MurmurCache(new_prelude, new_core, new_coda) if use_cache else None
        return CausalLMOutput(logits=logits, loss=loss, depths=depths, diagnostics={"cache": new_cache})

    def count_parameters(self) -> dict[str, int]:
        """Count unique parameters by component.

        Returns:
            Dictionary with parameter counts
        """
        seen = set()
        counts = {
            "embedding": 0,
            "prelude": 0,
            "core": 0,
            "coda": 0,
            "final_norm": 0,
            "lm_head": 0,
            "total_unique": 0,
        }

        for name, param in self.named_parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            numel = param.numel()

            if "token_embedding" in name:
                counts["embedding"] += numel
            elif "prelude" in name:
                counts["prelude"] += numel
            elif "core" in name:
                counts["core"] += numel
            elif "coda" in name:
                counts["coda"] += numel
            elif "final_norm" in name:
                counts["final_norm"] += numel
            elif "lm_head" in name:
                counts["lm_head"] += numel

            counts["total_unique"] += numel

        return counts
