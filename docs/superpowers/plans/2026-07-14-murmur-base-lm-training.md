# Murmur Base LM Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить внутри `science` полноценный, воспроизводимый код обучения базовой causal language model Murmur-RSM на одной NVIDIA T4: от токенизатора и подготовки данных до обучения, checkpoint/resume, evaluation и cached generation.

**Architecture:** Код модели принадлежит Murmur и реализует `embedding → Prelude → shared recurrent Core × T → Coda → tied LM head`. Обязательный рабочий путь использует GQA/SwiGLU и не зависит от сторонних CUDA-ядер; официальный Mamba 3 подключается отдельным опциональным адаптером после аппаратного gate. Обучение выполняется собственным компактным PyTorch trainer без зависимости от Hugging Face Trainer.

**Tech Stack:** Python 3.11, PyTorch 2.x с CUDA 12.x, NumPy, SentencePiece, safetensors, Hugging Face `datasets` только на стадии подготовки корпуса, pytest, ruff, TensorBoard.

## Global Constraints

- Все новые файлы создаются только под `science/`; `code/` и `website/` не изменяются.
- Обязательная среда исполнения — одна NVIDIA T4 с 16 GB VRAM.
- Основной precision на T4 — FP16 autocast с `GradScaler`; FP32 используется в reference-тестах.
- GQA recurrent baseline обязан полностью обучаться и генерировать без установленного `mamba-ssm`.
- Mamba 3 берётся из официального `state-spaces/mamba` через закреплённую Git-ревизию; её математика и CUDA-ядра не копируются и не переписываются.
- Mamba 3 является опциональным mixer: провал compile/runtime gate не блокирует обучение GQA-модели.
- Токенизатор — собственный SentencePiece BPE с byte fallback, vocab 32k или 48k и exact byte round-trip.
- Embedding и LM head связаны одним tensor; recurrent Core действительно разделяет параметры между всеми проходами.
- Runtime cache не разделяется между loop occurrences: каждому проходу соответствуют отдельные KV/SSM states.
- Первая обучаемая конфигурация — Smoke S, примерно 30–38M уникальных параметров; Prototype P 140–158M запускается только после прохождения smoke-gates.
- Основной objective v0 — causal next-token cross-entropy только по финальному выходу.
- Для сравнений фиксируются tokenizer hash, manifest hash, data order, число обработанных токенов, seed и hardware fingerprint.
- Checkpoint должен позволять детерминированное продолжение: model, optimizer, scheduler, scaler, RNG, sampler cursor и `tokens_seen` сохраняются атомарно.
- Реализация ведётся через тесты; каждая задача заканчивается отдельным проверяемым результатом и коммитом.

## Decision: брать репозиторий Mamba или писать основу самостоятельно

Выбран гибридный вариант:

1. **Не форкать весь Mamba-репозиторий.** Его trainer, fixed-depth scaffold и layout не соответствуют shared recurrent Core и усложнят честный GQA fallback.
2. **Написать собственные model/data/train слои Murmur.** Это оставляет под контролем weight sharing, random depth, cache, checkpoint и экспериментальные абляции.
3. **Подключить официальный Mamba 3 тонким адаптером.** Конкретная upstream-ревизия и окружение записываются в lock/manifest после проверки на T4.

Полный форк официального репозитория быстрее только для fixed-depth smoke test, но создаёт долг по удалению чужого trainer и затрудняет доказательство shared weights. Полностью собственная реализация Mamba 3 слишком рискованна: одна ошибка discretization/cache сделает все результаты недостоверными. Выбранный вариант минимизирует оба риска.

## Target Layout

```text
science/
  pyproject.toml
  README.md
  configs/
    smoke_gqa.toml
    smoke_mamba3.toml
    prototype_gqa.toml
  docs/
    superpowers/plans/...
  notebooks/
    kaggle_t4_smoke.ipynb
  scripts/
    hardware_probe.py
    pin_mamba.py
    train_tokenizer.py
    prepare_data.py
    train.py
    evaluate.py
    generate.py
    param_count.py
    resume_test.py
  src/murmur/
    __init__.py
    config.py
    reproducibility.py
    tokenizer.py
    data/
      manifest.py
      packing.py
      dataset.py
      sampler.py
    model/
      config.py
      norms.py
      rotary.py
      attention.py
      mlp.py
      block.py
      recurrent.py
      language_model.py
      cache.py
      mixers/
        protocol.py
        gqa.py
        mamba3.py
    training/
      loss.py
      optim.py
      schedule.py
      checkpoint.py
      metrics.py
      trainer.py
    evaluation/
      language_model.py
      depth.py
      generation.py
  tests/
    unit/...
    integration/...
    acceptance/...
  third_party/
    NOTICE.md
    mamba.lock.json
  artifacts/.gitkeep
```

---

### Task 1: Изолированный Python-проект и проверяемая конфигурация

**Files:**
- Create: `science/pyproject.toml`
- Create: `science/README.md`
- Create: `science/src/murmur/__init__.py`
- Create: `science/src/murmur/config.py`
- Create: `science/src/murmur/model/config.py`
- Create: `science/configs/smoke_gqa.toml`
- Create: `science/configs/prototype_gqa.toml`
- Create: `science/tests/unit/test_config.py`

**Interfaces:**
- Produces: `load_run_config(path: Path) -> RunConfig`; frozen dataclasses `RunConfig`, `ModelConfig`, `DataConfig`, `TrainConfig`, `EvalConfig`.
- Consumes: только Python standard library.

- [ ] **Step 1: Написать failing-тест загрузки и строгой валидации TOML.**

```python
def test_load_smoke_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[train]\nseed=17\nunknown=true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown"):
        load_run_config(path)
```

- [ ] **Step 2: Запустить `python -m pytest tests/unit/test_config.py -v`; ожидается FAIL из-за отсутствующего `murmur.config`.**
- [ ] **Step 3: Реализовать frozen dataclasses, рекурсивную проверку неизвестных/пропущенных ключей, диапазонов batch/sequence/steps и сериализацию resolved config.**
- [ ] **Step 4: Создать `smoke_gqa.toml`: `d_model=512`, vocab `32000`, Prelude/Coda `1/1`, Core `2` блока, Q/KV heads `8/2`, context `512`, `T=1..4`, FP16, gradient accumulation и token-based stopping.**
- [ ] **Step 5: Создать `prototype_gqa.toml`: `d_model=1024`, Prelude/Coda `2/2`, Core `4`, Q/KV `16/4`, context `1024`, без разрешения запуска до acceptance gate.**
- [ ] **Step 6: Запустить `python -m pytest tests/unit/test_config.py -v` и `python -m ruff check src tests`; ожидается PASS.**
- [ ] **Step 7: Commit: `git commit -m "build(science): initialize Murmur training package"`.**

### Task 2: Hardware probe и закрепление официальной Mamba

**Files:**
- Create: `science/scripts/hardware_probe.py`
- Create: `science/scripts/pin_mamba.py`
- Create: `science/third_party/NOTICE.md`
- Create: `science/third_party/mamba.lock.json`
- Create: `science/tests/unit/test_hardware_probe.py`

**Interfaces:**
- Produces: `collect_hardware_fingerprint() -> dict[str, JSONValue]`; `mamba.lock.json` с repository, commit, checked_at и license.
- Consumes: PyTorch runtime и `git ls-remote`/`git rev-parse` только в `pin_mamba.py`.

- [ ] **Step 1: Написать тест стабильной JSON-схемы fingerprint: Python, OS, torch, CUDA, GPU name, compute capability, total VRAM, FP16/BF16 support.**
- [ ] **Step 2: Реализовать probe без падения на CPU; отсутствие CUDA возвращает `cuda_available=false`, а CLI завершает hardware gate кодом 2.**
- [ ] **Step 3: Реализовать `pin_mamba.py`, который получает SHA `refs/heads/main` официального URL, проверяет 40-символьный commit и атомарно пишет lock.**
- [ ] **Step 4: В `NOTICE.md` записать, что Mamba остаётся отдельной optional dependency и используется по лицензии upstream; бинарные wheels и веса community-моделей не включаются.**
- [ ] **Step 5: На Kaggle выполнить `python scripts/hardware_probe.py --output artifacts/hardware.json`; gate требует `Tesla T4`, compute capability `7.5`, CUDA и FP16.**
- [ ] **Step 6: Запустить unit-тесты; ожидается PASS на CPU и GPU.**
- [ ] **Step 7: Commit: `git commit -m "feat(science): add T4 hardware and Mamba revision gates"`.**

### Task 3: Собственный byte-fallback токенизатор

**Files:**
- Create: `science/src/murmur/tokenizer.py`
- Create: `science/scripts/train_tokenizer.py`
- Create: `science/tests/unit/test_tokenizer.py`
- Create: `science/tests/fixtures/tokenizer_corpus.txt`

**Interfaces:**
- Produces: `MurmurTokenizer.train(...)`, `encode(text) -> list[int]`, `decode(ids) -> str`, `fingerprint() -> str`, свойства BOS/EOS/PAD IDs.
- Consumes: SentencePiece; вход — UTF-8 JSONL/text shards.

- [ ] **Step 1: Написать тест exact round-trip для русского, английского, Python-кода, пробелов, emoji, combining marks и произвольных валидных UTF-8 строк.**
- [ ] **Step 2: Написать тест неизменности special-token IDs и SHA-256 fingerprint модели.**
- [ ] **Step 3: Реализовать SentencePiece BPE с `byte_fallback=true`, identity normalization, `unk/bos/eos/pad` IDs `0/1/2/3`, без добавления dummy prefix.**
- [ ] **Step 4: CLI обучает кандидаты 32k/48k, фиксирует seed/arguments/corpus hashes и печатает bytes-per-token по EN/RU/code/math.**
- [ ] **Step 5: Добавить отказ, если vocab превышает `uint16` или любой round-trip fixture не совпадает.**
- [ ] **Step 6: Запустить `pytest tests/unit/test_tokenizer.py -v`; ожидается PASS.**
- [ ] **Step 7: Commit: `git commit -m "feat(science): add reproducible byte-fallback tokenizer"`.**

### Task 4: Immutable manifest, очистка и упаковка train/validation данных

**Files:**
- Create: `science/src/murmur/data/manifest.py`
- Create: `science/src/murmur/data/packing.py`
- Create: `science/src/murmur/data/dataset.py`
- Create: `science/src/murmur/data/sampler.py`
- Create: `science/scripts/prepare_data.py`
- Create: `science/tests/unit/test_manifest.py`
- Create: `science/tests/unit/test_packing.py`
- Create: `science/tests/unit/test_sampler.py`

**Interfaces:**
- Produces: `DatasetManifest`, `pack_documents(...)`, `PackedTokenDataset`, `StatefulTokenSampler.state_dict()/load_state_dict()`.
- Data format: contiguous little-endian `uint16` `.bin` shards плюс JSON `.idx`; каждый sample имеет ровно `sequence_length + 1` токенов.

- [ ] **Step 1: Написать failing-тест: одинаковые документы/seed дают byte-identical shards и manifest hash.**
- [ ] **Step 2: Написать тест отсутствия train/validation document-hash overlap и тест сохранения sampler cursor.**
- [ ] **Step 3: Реализовать manifest с dataset revision, license, source, filters, tokenizer hash, shard SHA-256, accepted/rejected counts и token counts.**
- [ ] **Step 4: Реализовать document-level split до packing, дедупликацию по нормализованному SHA-256 и EOS между документами. Не склеивать хвост одного split с другим.**
- [ ] **Step 5: Реализовать memory-mapped dataset и детерминированный shuffle без загрузки корпуса в RAM.**
- [ ] **Step 6: CLI принимает локальные JSONL и явно разрешённые HF dataset revisions; network download отделён от train runtime.**
- [ ] **Step 7: Выполнить unit-тесты и небольшой round-trip `prepare_data → dataset`; ожидается точное совпадение token IDs.**
- [ ] **Step 8: Commit: `git commit -m "feat(science): add reproducible packed-data pipeline"`.**

### Task 5: ModelConfig, RMSNorm, RoPE, GQA и SwiGLU

**Files:**
- Modify: `science/src/murmur/model/config.py`
- Create: `science/src/murmur/model/norms.py`
- Create: `science/src/murmur/model/rotary.py`
- Create: `science/src/murmur/model/attention.py`
- Create: `science/src/murmur/model/mlp.py`
- Create: `science/src/murmur/model/block.py`
- Create: `science/tests/unit/model/test_attention.py`
- Create: `science/tests/unit/model/test_block.py`

**Interfaces:**
- Produces: `ModelConfig`, `RMSNorm`, `RotaryEmbedding`, `GQAAttention`, `SwiGLU`, `TransformerBlock`.
- Tensor contract: hidden `[batch, sequence, d_model]`; attention cache `[batch, kv_heads, cached_sequence, head_dim]`.

- [ ] **Step 1: Написать shape/dtype tests и GQA reference test, сравнивающий grouped K/V с явным `repeat_interleave`.**
- [ ] **Step 2: Написать causal test: изменение будущего токена не меняет предыдущие logits.**
- [ ] **Step 3: Реализовать pre-norm blocks, bias-free projections, RoPE и `torch.nn.functional.scaled_dot_product_attention`; GQA K/V повторяются совместимым с T4 способом.**
- [ ] **Step 4: Реализовать SwiGLU `down(silu(gate(x)) * up(x))` и нормальную инициализацию с residual scaling.**
- [ ] **Step 5: Добавить config invariants: `d_model % q_heads == 0`, `q_heads % kv_heads == 0`, чётный RoPE dimension.**
- [ ] **Step 6: Запустить FP32 forward/backward tests и проверить отсутствие NaN.**
- [ ] **Step 7: Commit: `git commit -m "feat(science): implement GQA transformer primitives"`.**

### Task 6: Fixed-depth R0 и shared recurrent R1 Core

**Files:**
- Create: `science/src/murmur/model/mixers/protocol.py`
- Create: `science/src/murmur/model/mixers/gqa.py`
- Create: `science/src/murmur/model/recurrent.py`
- Create: `science/tests/unit/model/test_recurrent.py`
- Create: `science/tests/integration/test_r0_r1_forward.py`

**Interfaces:**
- Produces: `SequenceMixer` protocol, `GQAMixer`, `StableInputInjection`, `RecurrentCore.forward(hidden, injection, depths) -> hidden`.
- `depths` имеет форму `[batch]`; Core исполняется до `max(depths)`, а завершившиеся sequences сохраняют прежнее состояние через mask.

- [ ] **Step 1: Написать identity-тест: все проходы R1 ссылаются на один объект Core и один набор `data_ptr()` параметров.**
- [ ] **Step 2: Написать `T=1` test и masked per-sequence test для depths `[1, 2, 4]`.**
- [ ] **Step 3: Реализовать стабильную инъекцию по контракту архитектурного документа: отрицательная diagonal continuous-time `A`, дискретный decay в `(0,1]`, отдельная проекция неизменного Prelude output `e`.**
- [ ] **Step 4: Реализовать shared Core как список физических блоков, весь список повторяется T раз с общими параметрами; random T семплируется один раз на sequence.**
- [ ] **Step 5: Логировать state norm, update norm, gradient norm и диапазон decay на каждом loop index.**
- [ ] **Step 6: Сравнить R0/R1 при `T=1` в контролируемой конфигурации; shape, causal behavior и gradients должны совпадать там, где веса синхронизированы.**
- [ ] **Step 7: Commit: `git commit -m "feat(science): add fixed and recurrent GQA cores"`.**

### Task 7: Полная MurmurForCausalLM и подсчёт уникальных параметров

**Files:**
- Create: `science/src/murmur/model/language_model.py`
- Create: `science/scripts/param_count.py`
- Create: `science/tests/unit/model/test_language_model.py`

**Interfaces:**
- Produces: `MurmurForCausalLM.forward(input_ids, labels=None, depths=None, cache=None) -> CausalLMOutput`.
- `CausalLMOutput` содержит `logits`, optional `loss`, `cache`, `depths`, `diagnostics`.

- [ ] **Step 1: Написать тест tied weights: `lm_head.weight is token_embedding.weight`.**
- [ ] **Step 2: Написать тест causal loss со shift `logits[:, :-1]` против `labels[:, 1:]`, игнорируя PAD через `ignore_index=-100`.**
- [ ] **Step 3: Реализовать topology `embedding → Prelude → LN(e) → Core×T → Coda → final RMSNorm → tied head`.**
- [ ] **Step 4: Реализовать уникальный parameter count по object identity/storage, а также отдельные `N_nonloop`, `N_core`, `N_effective(T)`.**
- [ ] **Step 5: Проверить Smoke S: вывести фактическое число параметров и объяснить расхождение с ориентиром 30–38M по vocab и компонентам. Блокировать запуск только при несовпадении с точным `expected_parameter_range`, записанным в resolved config после выбора токенизатора.**
- [ ] **Step 6: Выполнить tiny overfit на одном batch: loss должна заметно снижаться за 100 шагов.**
- [ ] **Step 7: Commit: `git commit -m "feat(science): assemble Murmur causal language model"`.**

### Task 8: KV/SSM cache и cached-generation parity

**Files:**
- Create: `science/src/murmur/model/cache.py`
- Create: `science/src/murmur/evaluation/generation.py`
- Create: `science/scripts/generate.py`
- Create: `science/tests/unit/model/test_cache.py`
- Create: `science/tests/integration/test_cached_decode.py`

**Interfaces:**
- Produces: `MurmurCache`, `LoopCache`, `prefill(...)`, `decode_step(...)`, greedy/top-k generation CLI.
- Cache индексируется по physical block и loop occurrence; tensors между loop occurrences не alias.

- [ ] **Step 1: Написать test, доказывающий отсутствие alias между cache loop 0/1/3.**
- [ ] **Step 2: Написать full-forward vs token-by-token parity для `T=1,2,4`: FP32 max abs error `≤1e-5`; FP16 `allclose(atol=1e-2, rtol=1e-2)` и одинаковые greedy tokens.**
- [ ] **Step 3: Реализовать attention KV cache с position offset и ограничением max sequence length.**
- [ ] **Step 4: Реализовать prefill/decode API так, чтобы training path не создавал cache.**
- [ ] **Step 5: CLI загружает tokenizer/config/safetensors и генерирует с фиксированным seed.**
- [ ] **Step 6: Выполнить integration tests на CPU FP32 и T4 FP16.**
- [ ] **Step 7: Commit: `git commit -m "feat(science): add recurrent cache and generation"`.**

### Task 9: Опциональный официальный Mamba 3 adapter и T4 gate

**Files:**
- Create: `science/src/murmur/model/mixers/mamba3.py`
- Create: `science/configs/smoke_mamba3.toml`
- Create: `science/tests/unit/model/test_mamba3_adapter.py`
- Create: `science/tests/acceptance/test_mamba3_t4.py`

**Interfaces:**
- Produces: `Mamba3Mixer` с тем же `SequenceMixer` contract, capability report и explicit `MambaUnavailableError`.
- Consumes: только официальный `mamba_ssm.modules.mamba3.Mamba3` из revision, записанной в lock.

- [ ] **Step 1: Написать import test: без optional dependency GQA импортируется, а создание `Mamba3Mixer` даёт понятную ошибку с инструкцией установки.**
- [ ] **Step 2: Реализовать adapter без копирования Mamba-кода; параметры `d_model`, `d_state`, `headdim`, `layer_idx`, `chunk_size=64` приходят из `ModelConfig`.**
- [ ] **Step 3: Добавить отдельные inference states на каждый physical block/loop occurrence через официальный cache API.**
- [ ] **Step 4: T4 acceptance gate: clean install, FP16 forward/backward, 100 optimizer steps без NaN, затем 1000 synthetic steps без divergence, cached parity и замер peak VRAM/tokens/s.**
- [ ] **Step 5: Если compile или acceptance gate падает, записать machine-readable status `unsupported` и продолжить pipeline с `mixer="gqa"`; автоматической подмены во время уже начатого run не делать.**
- [ ] **Step 6: Commit: `git commit -m "feat(science): integrate optional official Mamba 3 mixer"`.**

### Task 10: Optimizer, scheduler, FP16 и token-accurate trainer

**Files:**
- Create: `science/src/murmur/training/loss.py`
- Create: `science/src/murmur/training/optim.py`
- Create: `science/src/murmur/training/schedule.py`
- Create: `science/src/murmur/training/metrics.py`
- Create: `science/src/murmur/training/trainer.py`
- Create: `science/scripts/train.py`
- Create: `science/tests/unit/training/test_schedule.py`
- Create: `science/tests/integration/test_training_step.py`

**Interfaces:**
- Produces: `Trainer.fit()`, AdamW parameter groups, warmup+cosine scheduler, JSONL/TensorBoard metrics.
- Trainer прекращает обучение по `max_tokens`, а не только по step count; `tokens_seen` считает непаддинговые target tokens.

- [ ] **Step 1: Написать tests для weight-decay groups: norms, biases и embeddings без decay; matrix weights с decay.**
- [ ] **Step 2: Написать scheduler boundary tests для warmup, cosine minimum LR и resume at arbitrary step.**
- [ ] **Step 3: Реализовать gradient accumulation, FP16 autocast/GradScaler, unscale-before-clipping, skip-step при non-finite gradients и fail-fast после заданного числа skips.**
- [ ] **Step 4: Реализовать random per-sequence T с детерминированным `torch.Generator`, final-only CE и логированием CE, bits/byte, LR, grad norm, tokens/s, peak VRAM, T histogram.**
- [ ] **Step 5: Реализовать validation без изменения sampler/RNG train и раннее сохранение best checkpoint по validation bits/byte.**
- [ ] **Step 6: CLI поддерживает `--config`, `--resume`, `--device`, `--run-dir`, `--override key=value`; resolved config всегда сохраняется в run-dir.**
- [ ] **Step 7: Integration test обучает tiny model 20 шагов и подтверждает снижение loss, изменение weights и отсутствие NaN.**
- [ ] **Step 8: Commit: `git commit -m "feat(science): implement token-accurate FP16 trainer"`.**

### Task 11: Атомарные checkpoints и детерминированный resume

**Files:**
- Create: `science/src/murmur/training/checkpoint.py`
- Create: `science/src/murmur/reproducibility.py`
- Create: `science/scripts/resume_test.py`
- Create: `science/tests/unit/training/test_checkpoint.py`
- Create: `science/tests/integration/test_deterministic_resume.py`

**Interfaces:**
- Produces: `save_checkpoint_atomic(...)`, `load_checkpoint(...)`, `capture_rng_state()`, `restore_rng_state()`.
- Model weights сохраняются в safetensors; trusted local optimizer/scaler/RNG state — отдельным versioned PyTorch state file; manifest содержит SHA-256 каждого файла.

- [ ] **Step 1: Написать interrupted-write test: незавершённый temp directory не считается checkpoint.**
- [ ] **Step 2: Написать resume equivalence test: 10 непрерывных шагов равны 5 + save/load + 5 по batch IDs, LR, `tokens_seen` и параметрам в FP32 reference mode.**
- [ ] **Step 3: Сохранять model/optimizer/scheduler/scaler, CPU/CUDA RNG, depth generator, sampler state, data cursor, config/hash, tokenizer/hash, manifest/hash и hardware fingerprint.**
- [ ] **Step 4: Писать checkpoint в sibling temp directory, fsync важных файлов, создавать `COMPLETED` последним и публиковать через `os.replace`.**
- [ ] **Step 5: Resume обязан отклонять несовпадающие tokenizer/data/config hashes, кроме явно allowlisted runtime-параметров логирования.**
- [ ] **Step 6: Запустить stress-test с принудительным исключением и восстановлением.**
- [ ] **Step 7: Commit: `git commit -m "feat(science): add deterministic atomic checkpoint resume"`.**

### Task 12: Evaluation по доменам, глубине и производительности

**Files:**
- Create: `science/src/murmur/evaluation/language_model.py`
- Create: `science/src/murmur/evaluation/depth.py`
- Create: `science/scripts/evaluate.py`
- Create: `science/tests/unit/evaluation/test_metrics.py`
- Create: `science/tests/integration/test_depth_sweep.py`

**Interfaces:**
- Produces: CE, perplexity, bits/byte, domain metrics, quality-vs-T report, OTR inputs, latency/VRAM/cache report.
- Evaluation output: immutable JSON с checkpoint/config/data hashes и одним record на domain × T.

- [ ] **Step 1: Написать hand-calculated tests для CE/perplexity/bits-per-byte и aggregation с разным числом tokens.**
- [ ] **Step 2: Реализовать eval для EN/RU/code/math validation splits при `T=1..8`; запретить выбор только лучшего T без публикации всей кривой.**
- [ ] **Step 3: Реализовать benchmark warmup, синхронизацию CUDA, p50/p95 prefill/decode latency, tokens/s, peak VRAM и cache bytes/sequence.**
- [ ] **Step 4: Добавить synthetic state tests: associative recall, parity/modular state и state tracking как отдельный диагностический suite.**
- [ ] **Step 5: CLI пишет `evaluation.json`, `depth_curve.csv` и краткую Markdown-сводку.**
- [ ] **Step 6: Commit: `git commit -m "feat(science): add domain depth and T4 evaluation"`.**

### Task 13: Kaggle T4 notebook и end-to-end smoke run

**Files:**
- Create: `science/notebooks/kaggle_t4_smoke.ipynb`
- Create: `science/tests/acceptance/test_smoke_pipeline.py`
- Modify: `science/README.md`

**Interfaces:**
- Notebook вызывает только versioned scripts; в notebook не дублируется model/trainer logic.
- Produces: run bundle с environment, config, manifest, logs, checkpoints, evaluation и sample generations.

- [ ] **Step 1: Создать notebook с ячейками: environment probe, dependency install, optional Mamba gate, tokenizer/data artifact verification, smoke train, resume test, eval, generation, artifact export.**
- [ ] **Step 2: Добавить timeout-aware checkpoint cadence, чтобы каждые 10–15 минут существовал завершённый checkpoint на persistent Kaggle Dataset/output target.**
- [ ] **Step 3: Acceptance pipeline на tiny fixture выполняет `train_tokenizer → prepare_data → train → interrupt/resume → evaluate → generate` одной командой.**
- [ ] **Step 4: Реальный Smoke S gate на T4: 100 шагов, затем 1000 synthetic/малых data шагов, отсутствие NaN, cached parity, успешный resume и снижение validation CE.**
- [ ] **Step 5: README документирует точные команды локального CPU-test и Kaggle T4 run, структуру artifacts и процедуру восстановления.**
- [ ] **Step 6: Commit: `git commit -m "docs(science): add Kaggle T4 end-to-end workflow"`.**

### Task 14: R0/R1/M0 ablation harness и go/no-go для Prototype P

**Files:**
- Create: `science/configs/ablations/r0_fixed_gqa.toml`
- Create: `science/configs/ablations/r1_recurrent_gqa.toml`
- Create: `science/configs/ablations/m0_recurrent_mamba3.toml`
- Create: `science/scripts/run_ablation.py`
- Create: `science/scripts/check_gate.py`
- Create: `science/tests/unit/test_gate.py`

**Interfaces:**
- Produces: сравнимые run specs, `ablation_summary.json`, `GateDecision(allowed, reasons, evidence)`.
- Все arms используют одинаковые tokenizer/manifest/data order/seen tokens; отчёт содержит unique params, active params/FLOPs и wall-clock.

- [ ] **Step 1: Написать gate tests для каждого отказа: divergence, cache mismatch, failed resume, incomplete license manifest, insufficient seeds и проигрыш M0 по выбранной метрике.**
- [ ] **Step 2: Реализовать R0 fixed-depth GQA, R1 shared recurrent GQA и M0 only-mixer-change configs без дополнительных mechanism changes.**
- [ ] **Step 3: Реализовать parameter/active-FLOP accounting и запрет сравнения runs с разными hashes/data order/token counts.**
- [ ] **Step 4: Выполнить короткий 100–300M-token budget на каждый жизнеспособный arm, затем три seed для решения о масштабировании.**
- [ ] **Step 5: `check_gate.py` разрешает `prototype_gqa.toml`/Mamba-вариант только при прохождении всех условий раздела 11.3 архитектурного документа; решение сохраняется рядом с evidence.**
- [ ] **Step 6: Commit: `git commit -m "feat(science): add reproducible architecture ablation gates"`.**

### Task 15: Финальная верификация полноценного base-training pipeline

**Files:**
- Modify: `science/README.md`
- Create: `science/docs/training_runbook.md`
- Create: `science/docs/model_card_template.md`

**Interfaces:**
- Produces: готовый к длительному запуску runbook и проверенный release checklist.

- [ ] **Step 1: Запустить `python -m pytest tests/unit -v`; ожидается 100% PASS.**
- [ ] **Step 2: Запустить integration/acceptance suite на CPU tiny config и T4 smoke config; GPU-specific Mamba test допускает только PASS или документированный `unsupported`, но не silent skip.**
- [ ] **Step 3: Запустить `python -m ruff check src scripts tests` и `python -m ruff format --check src scripts tests`; ожидается PASS.**
- [ ] **Step 4: Выполнить чистое создание environment из `pyproject.toml`, чтобы исключить незадекларированные зависимости.**
- [ ] **Step 5: Проверить artifact bundle: resolved config, hashes, hardware, metrics, checkpoints, evaluation и generation samples присутствуют и проходят SHA validation.**
- [ ] **Step 6: В runbook зафиксировать команды подготовки корпуса, preflight, обучения, мониторинга, восстановления, evaluation и публикации model card.**
- [ ] **Step 7: Commit: `git commit -m "docs(science): finalize base model training runbook"`.**

## Definition of Done

Проект считается готовым к обучению базовой модели, когда:

- чистая установка на Kaggle T4 запускает GQA Smoke S без ручного редактирования кода;
- tokenizer/data artifacts воспроизводимы и имеют immutable hashes;
- tiny end-to-end pipeline проходит на CPU, Smoke S — на T4 FP16;
- полный forward и cached decode совпадают для `T=1,2,4`;
- принудительно прерванное обучение детерминированно продолжается;
- модель переобучает tiny batch и снижает held-out CE в smoke run;
- R0/R1 сравниваются при одинаковых данных, токенах и seed; M0 запускается только при успешном Mamba T4 gate;
- ни одна часть обязательного pipeline не зависит от community checkpoints, pickle weights или непроверенных binary wheels;
- Prototype P остаётся заблокированным до machine-readable go/no-go решения.

## Recommended Execution Order

Критический путь: Tasks 1 → 3 → 4 → 5 → 6 → 7 → 10 → 11 → 12 → 13 → 14 → 15. Task 2 выполняется в начале доступной Kaggle-сессии; Task 8 нужен до длительного обучения; Task 9 можно завершать независимо после появления рабочего GQA baseline. Такой порядок сначала создаёт полностью обучаемую модель без Mamba-зависимости, затем добавляет ускоренный/экспериментальный mixer без риска заблокировать проект.
