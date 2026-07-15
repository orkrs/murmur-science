"""Single-GPU token-based trainer."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from murmur.config import RunConfig
from murmur.model.language_model import MurmurForCausalLM
from murmur.reproducibility import seed_everything
from murmur.training.checkpoint import load_checkpoint, save_checkpoint_atomic
from murmur.training.metrics import MetricsWriter
from murmur.training.optim import build_adamw
from murmur.training.schedule import cosine_warmup_lambda


class Trainer:
    def __init__(self, model: MurmurForCausalLM, train_loader: DataLoader, config: RunConfig, run_dir: Path, device: str | torch.device = "cuda", sampler=None):
        self.model = model.to(device)
        self.loader = train_loader
        self.sampler = sampler
        self.config = config
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = build_adamw(model, config.train.learning_rate, config.train.weight_decay)
        total_steps = max(1, config.train.max_tokens // (config.data.batch_size * config.data.sequence_length))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: cosine_warmup_lambda(step, config.train.warmup_steps, total_steps),
        )
        self.autocast_enabled = (config.train.fp16 or config.train.bf16) and self.device.type == "cuda"
        self.autocast_dtype = torch.bfloat16 if config.train.bf16 else torch.float16
        scaler_enabled = config.train.fp16 and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):  # PyTorch 2.0 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.metrics = MetricsWriter(self.run_dir / "metrics.jsonl")
        self.step = 0
        self.tokens_seen = 0
        self.depth_generator = torch.Generator(device="cpu").manual_seed(config.train.seed + 17)
        seed_everything(config.train.seed)

    def _depths(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            self.config.model.min_depth,
            self.config.model.max_depth + 1,
            (batch_size,),
            generator=self.depth_generator,
        )

    def train(self, max_steps: int | None = None) -> dict:
        self.model.train()
        target_tokens = self.config.train.max_tokens
        if max_steps is not None:
            target_tokens = min(target_tokens, max_steps * self.config.data.batch_size * self.config.data.sequence_length)
        iterator = iter(self.loader)
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        while self.tokens_seen < target_tokens:
            try:
                tokens = next(iterator)
            except StopIteration:
                iterator = iter(self.loader)
                tokens = next(iterator)
            tokens = tokens.to(self.device, non_blocking=True)
            depths = self._depths(tokens.shape[0]).to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_enabled,
            ):
                output = self.model(tokens, labels=tokens, depths=depths)
                loss = output.loss / self.config.train.grad_accum_steps
            self.scaler.scale(loss).backward()
            if (self.step + 1) % self.config.train.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.metrics.write({"step": self.step, "loss": float(output.loss.detach()), "grad_norm": float(grad_norm), "lr": self.optimizer.param_groups[0]["lr"], "tokens_seen": self.tokens_seen})
            self.step += 1
            self.tokens_seen += int(tokens.shape[0] * max(1, tokens.shape[1] - 1))
            if self.step % self.config.train.checkpoint_interval == 0:
                self.save_checkpoint()
        self.save_checkpoint(name="last")
        return {"steps": self.step, "tokens_seen": self.tokens_seen, "seconds": time.perf_counter() - started}

    def save_checkpoint(self, name: str | None = None) -> Path:
        name = name or f"step-{self.step:08d}"
        return save_checkpoint_atomic(
            self.run_dir / "checkpoints" / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            {"step": self.step, "tokens_seen": self.tokens_seen, "config": self.config.to_dict()},
            {"sampler": self.sampler.state_dict() if self.sampler is not None else None, "depth_generator": self.depth_generator.get_state()},
        )

    def resume(self, checkpoint: Path) -> dict:
        extra = {}
        meta = load_checkpoint(checkpoint, self.model, self.optimizer, self.scheduler, self.scaler, extra)
        self.step = int(meta["step"])
        self.tokens_seen = int(meta["tokens_seen"])
        if self.sampler is not None and extra.get("sampler") is not None:
            self.sampler.load_state_dict(extra["sampler"])
        if extra.get("depth_generator") is not None:
            self.depth_generator.set_state(extra["depth_generator"])
        return meta
