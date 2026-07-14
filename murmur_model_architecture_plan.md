# Murmur-RSM: план собственной рекуррентной LM

**Версия:** 0.2  
**Дата верификации:** 14 июля 2026 года  
**Решение:** только Track B — собственная архитектура, токенизатор и веса  
**Масштаб первого решения:** 30–40M smoke → 140–160M prototype  
**Долгосрочная цель:** до 4B уникальных параметров после подтверждения scaling

---

## 1. Решение

Murmur-RSM следует разрабатывать с нуля как recurrent-depth LM:

1. **Parcae** — основной воспроизводимый каркас Prelude → shared Core × T → Coda и источник механизма стабильной инъекции входа.
2. **Mamba 3 SISO** — кандидат на mixer рекуррентного Core, но только после проверки на реальном Kaggle GPU.
3. **GQA** — фиксированный baseline и attention в Prelude/Coda.
4. **Mamba 3 MIMO** — поздняя аппаратная абляция, не зависимость v0.
5. **Собственный byte-fallback tokenizer и собственные веса** — без копирования Ouro/LAMBA/Tiny.

Это не дообучение готовой Mamba 3. В официальном репозитории на дату проверки нет опубликованного Mamba‑3 language-model checkpoint. Найденные community-модели полезны как инженерные референсы, но не являются надёжными донорами весов.

Kaggle используется для проверки kernels, обучения 30–40M baseline и ограниченных 140–160M прогонов. Модель 4B на Kaggle с нуля нереалистична; 4B остаётся целью масштабирования после доказательства эффективности на меньших моделях.

---

## 2. Как читать утверждения

В документе разделены четыре уровня достоверности:

| Метка | Значение |
|---|---|
| **Проверено** | Видно в статье, коде, конфигурации или файлах репозитория |
| **Заявлено автором** | Есть в model card/README, но нет достаточных логов или воспроизводимого train pipeline |
| **Расчёт** | Получено из опубликованной конфигурации и кода; нужно перепроверить скриптом после реализации |
| **Проектное решение** | Наш выбор, который ещё должен пройти абляцию |

Публичный результат считается воспроизведённым только при наличии закреплённой версии кода, manifest данных, конфигурации, seed, логов и независимого eval.

---

## 3. Что дают источники

| Источник | Проверенный вывод | Роль в Murmur-RSM | Не переносить как факт |
|---|---|---|---|
| Fable 5 / Mythos 5 | Anthropic сообщает об общей базовой модели и различиях в safeguards; подробная архитектура не раскрыта | Продуктовая цель: длинные задачи, устойчивость, работа с инструментами | Mamba, looped layers, MoE, число параметров |
| OpenMythos | Теоретическая реконструкция Prelude/Core/Coda с recurrent depth, attention и MoE | Каталог идей и интерфейсов | «Открытая архитектура Mythos» или подтверждённое качество |
| Parcae | Стабилизированный looped Core, input injection и per-sequence sampling глубины проверены на LM; опубликованы код и веса нескольких размеров | **Главный scaffold и baseline** | Автоматический перенос scaling на 4B |
| Mamba 3 | Официальные статья и код; SISO/MIMO, новая discretization и complex state | Кандидат mixer после hardware gate | Наличие готовой Mamba‑3 LM для fine-tune |
| Ouro-1.4B | 1,434,652,673 параметра, 24 physical MHA-слоя, весь stack повторяется четыре раза | Внешний benchmark/возможный teacher позднее | Донор архитектуры или весов для Track B |
| LT² | Статья описывает hybrid Full+GDN и conversion примерно на 1B токенов | Подтверждение направления hybrid conversion | Готовый публичный conversion pipeline для нашей модели |
| TRM / PTRM | Итеративное исправление ответа и Q-head показаны на puzzle-задачах | Поздний исследовательский референс | Готовый autoregressive LM backbone |

Главный вывод по Fable: длительная агентная работа определяется не только backbone. Нужны pretraining, post-training, tool-use, управление контекстом, память и agent harness. Архитектурный проект не должен обещать «аналог Fable» сам по себе.

---

## 4. Проверка найденных Hugging Face моделей

### 4.1 aifeifei798/Mamba3-MIMO-Tiny-HF

**Проверено по config, model wrapper и train script:**

- Apache-2.0 в model card;
- d_model 256, 2 слоя, d_state 64, headdim 64, MIMO rank 2, vocab 50,257;
- расчётное число параметров: **26,680,608**;
- embeddings и LM head не связаны;
- опубликованная обёртка — последовательность Mamba-блоков, финальный LayerNorm и LM head; отдельного FFN и явного residual wrapper нет;
- training script повторяет 100 эпох всего две английские строки при длине 64;
- низкий указанный loss показывает запоминание этих строк, а не языковое качество;
- загрузка требует custom code.

**Решение:** использовать только как smoke fixture для импорта, HF packaging и MIMO compile. Не использовать веса, loss или генерацию как quality baseline. Перед любым запуском просмотреть custom code; не включать trust_remote_code для непроверенной ревизии.

### 4.2 kdirgul/Mamba3-177M-GQA-Hybrid_LAMBA_V1.0

**Проверено по model card, CPU-коду и дереву файлов:**

- лицензия обозначена как **LAMBAv1-license-TBD**;
- fixed-depth архитектура, а не shared recurrent-depth;
- d_model 768, 20 слоёв: 17 Mamba 3 SISO и 3 GQA на индексах 5, 11, 17;
- GQA: 12 Q-heads и 3 KV-heads;
- Mamba: d_state 128, head 64, complex state, rotary fraction 0.5, без Conv1d;
- GatedMLP: intermediate 1500, округлено до 1536;
- vocab 48k, tied embeddings, context 2048;
- расчётное число параметров: **177,070,256**;
- опубликован checkpoint около 354 MB в pickle-совместимом формате;
- репозиторий содержит около 456 MB сторонних binary wheels;
- CPU-путь сам сообщает расхождение порядка 0.06–0.09 с используемым fork и не является bit-exact reference.

**Только заявлено автором:** обучение с нуля на 12B токенов на Colab A100, состав корпуса, final loss 2.50 и eval на N=300. В репозитории нет достаточного train pipeline, data manifest и логов для независимой проверки этих результатов.

**Решение:** архитектурный референс для сочетания SISO+GQA, но не база:

- не использовать/не распространять веса до прояснения лицензии;
- не считать заявленное обучение воспроизведённым;
- не устанавливать приложенные wheels; собирать зависимости из закреплённого исходного кода в изолированной среде;
- не загружать непроверенный pickle обычным torch.load; предпочитать safetensors либо weights_only=True после проверки;
- не смешивать fork mamba-og с официальным state-spaces/mamba без parity tests.

### 4.3 Практический вывод

| Артефакт | Кодовая польза | Веса | Quality evidence | Роль |
|---|---:|---:|---:|---|
| Mamba3-MIMO-Tiny-HF | Средняя | Нет | Нет | Smoke fixture |
| LAMBA-177M | Средняя | Нет до лицензии | Неподтверждённое | Архитектурный референс |
| Parcae | Высокая | Да, для baseline | Статья + код | Основной scaffold |
| Official Mamba 3 | Высокая после gate | LM-весов нет | Статья + kernels | Mixer implementation |

---

## 5. Целевая архитектура v0

### 5.1 Топология

~~~text
tokens
  → own tied embedding
  → Prelude: GQA blocks
  → e = LN(Prelude output)
  → shared recurrent Core × T
  → Coda: GQA blocks
  → tied LM head
~~~

Для первого Mamba-кандидата Core содержит Mamba 3 SISO + dense SwiGLU. Attention остаётся только в Prelude/Coda, чтобы его KV-cache не умножался на T.

Стабильная инъекция следует проверяемой схеме Parcae:

~~~text
h[t+1] = A_bar ⊙ h[t] + B_bar(e) + R_theta(h[t], e)
A = diag(-exp(a))
~~~

Здесь R_theta — общий recurrent Core. Отрицательная диагональная continuous-time A после корректной discretization ограничивает линейную динамику. Это не является доказательством устойчивости всей нелинейной сети, поэтому обязательны мониторинг норм, gradient norm и spectral radius A_bar.

### 5.2 Важная экспериментальная дисциплина

Сначала воспроизводится **неизменённый Parcae baseline R1**. Инициализация h0, normalization, attention, optimizer и schedule берутся из закреплённого upstream-кода. Только после воспроизведения один компонент за раз заменяется на Mamba.

Нельзя одновременно добавлять Mamba, random T, loop embedding, Q-head и distillation: результат невозможно интерпретировать.

### 5.3 Конфигурации

| Профиль | Smoke S | Prototype P |
|---|---:|---:|
| d_model | 512 | 1024 |
| vocab-кандидаты | 32k / 48k | победитель tokenizer eval |
| Prelude / Coda | 1 / 1 GQA | 2 / 2 GQA |
| Shared Core | 2 SISO blocks | 4 SISO blocks |
| GQA Q/KV heads | 8 / 2 | 16 / 4 |
| SwiGLU intermediate | 1408 | 2816 |
| Mamba d_state | 64 | 128 |
| Mamba headdim | 64 | 64 |
| T train | 1–4 | 2–4, затем расширение |
| Начальный context | 512–1024 | 1024 → 2048 |
| Уникальные параметры, расчёт | около 30–38M | около 140–158M |

Диапазон параметров зависит прежде всего от vocab. После реализации единственным авторитетным числом становится вывод param_count.py по уникальным tensors с отдельным отчётом tied/shared weights.

### 5.4 Runtime-state и cache

Shared weights не означают shared runtime state.

- Каждому эффективному повтору Mamba Core нужен отдельный SSM state при autoregressive decode.
- Attention внутри recurrent Core требовала бы отдельный KV-cache для каждого loop occurrence.
- Поэтому cache, latency и active compute растут с T, даже когда число уникальных параметров постоянно.

Обязательный тест: сравнить logits полного forward с пошаговым cached decode для T = 1, 2, 4. Для FP32 цель — max absolute error не выше 1e-5 на малой reference-конфигурации. Для FP16/BF16 — allclose с atol/rtol 1e-2 и совпадение greedy tokens на фиксированных prompts; допуск уточняется относительно fixed-depth control.

---

## 6. Hardware gate для Mamba 3

### 6.1 Карта официального кода

| Файл | Что в нём проверять | Как использовать |
|---|---|---|
| selective_scan_interface.py | CUDA selective scan, fused inner path и reference path **Mamba‑1** | Regression/fallback reference; не выдавать за Mamba‑3 |
| modules/ssd_minimal.py | Короткая discrete SSD из Mamba‑2 и сравнение с fused chunk scan | Correctness oracle для SSD/fallback, не основа Mamba‑3 |
| modules/mamba3.py | Единственный основной Mamba‑3 module: SISO/MIMO branches, heavy-tail A, chunk size, step и inference cache | Источник Mamba‑3 mixer; не переписывать математику до parity |
| models/mixer_seq_simple.py | Block, residual/norm/MLP, hybrid MHA layers, Mamba1/2/3 selector, LM head и generation interface | Fixed-depth Mamba‑3 control и источник integration patterns |

В текущем mamba3.py default chunk_size равен 64; комментарий рекомендует 64 для SISO и 64/mimo_rank для MIMO. Модуль хранит при decode отдельные angle/dt, SSM, K и V states. Это подтверждает требование разделять cache по physical layer и по каждому recurrent occurrence.

mixer_seq_simple.py уже принимает ssm_cfg.layer = Mamba3 и умеет добавлять attention по attn_layer_idx, но создаёт обычный список разных слоёв. Его нельзя использовать как recurrent-depth модель без явного shared Core и теста identity параметров. Он нужен как fixed-depth control, а не как замена Parcae scaffold.

Официальный MIMO kernel наиболее оптимизирован и протестирован для узкого H100-профиля, включая sequence 2048, 32 heads, qk 128, v 64, rank 4 и chunk 16. Это не гарантирует работу или скорость на T4/P100.

До архитектурного решения notebook обязан записать:

- точную модель GPU и compute capability;
- CUDA, driver, PyTorch, Triton/TileLang и commit Mamba;
- доступность FP16/BF16;
- свободные VRAM/disk и ограничение сессии;
- compile time, peak VRAM, tokens/s и effective TFLOP/s.

**Gate Mamba 3 SISO:**

1. clean source build из закреплённого official commit;
2. forward/backward в поддерживаемой precision;
3. 100 шагов без NaN;
4. 1,000 synthetic шагов без divergence;
5. cached-decode parity;
6. speed/VRAM не хуже заранее зафиксированного допустимого отношения к GQA/GDN control;
7. повторный запуск из чистого Kaggle notebook.

Если gate не пройден, v0 не блокируется: Core остаётся GQA/Parcae, затем отдельно проверяются GDN или Mamba 2. MIMO разрешён только после SISO и только как отдельная hardware ablation.

---

## 7. Матрица минимальных абляций

| ID | Изменение относительно предыдущего | Что доказывает |
|---|---|---|
| R0 | Fixed-depth GQA, без shared weights | Parameter/FLOP control |
| R1 | Официальный Parcae GQA recurrent Core | Воспроизводимость recurrent-depth |
| M0 | В R1 только Core mixer → Mamba 3 SISO | Ценность SISO |
| M1 | В M0 только fixed T → per-sequence random T | Устойчивость к test-time depth |
| M2 | В M1 только loop-index embedding | Нужна ли специализация циклов |
| M3 | Один GQA refresh внутри Core | Нужен ли recurrent retrieval |
| H0 | MIMO вместо SISO | Качество/скорость на целевом GPU |

Для каждого сравнения нужны:

- одинаковый tokenizer, data order и число seen tokens;
- parameter-matched и active-FLOP-matched controls;
- минимум 3 seed для решений о масштабировании;
- один новый механизм на запуск;
- заранее записанная гипотеза и критерий остановки.

Per-sequence random T уменьшает variance и улучшает перенос между глубинами в Parcae, но не гарантирует пропорциональной экономии wall-clock: batch обычно считается до максимального T.

---

## 8. Обучение

### 8.1 Loss ladder

Начальный objective:

~~~text
L0 = causal cross-entropy по финальному выходу
~~~

Добавления разрешены только последовательно:

1. L0;
2. либо intermediate-exit CE, либо loop-consistency — не оба сразу;
3. опциональный online или малый top-k distillation;
4. Q-head/compute penalty только после стабильной LM.

Для T ≤ 4 сначала использовать full BPTT с activation checkpointing. Truncated/detached recurrence — отдельная абляция, потому что меняет gradient path.

Не хранить offline top-k512 teacher logits на 1B токенов: это примерно 2.05 TB даже при uint16 token IDs и FP16 logits. Top-k16/32 — около 64/128 GB при 4 bytes на entry; использовать только небольшой subset/shards либо online teacher.

### 8.2 Compute accounting

Для looped-модели считать не только уникальные параметры:

~~~text
N_eff = N_nonloop + E[T] × N_core
F_train ≈ 6 × D × N_eff
hours ≈ F_train / (F_effective × 3600)
~~~

Для ориентировочного Prototype P: N_nonloop около 94M, N_core около 61M, E[T] = 3, значит N_eff около 277M. Один миллиард train tokens — примерно 1.66e18 FLOPs. При измеренных 5–15 effective TFLOP/s это около 31–92 непрерывных GPU-часов без учёта простоев и compile overhead. Это расчёт для бюджета, не обещание Kaggle runtime.

Иллюстрация масштаба 4B: 80B токенов, выбранные лишь как грубые 20 tokens/parameter, дают около 1.92e21 FLOPs для обычной dense-модели. На одном GPU с 5–20 effective TFLOP/s это примерно 3–12 непрерывных GPU-лет ещё до recurrent overhead. Эвристика 20 tokens/parameter не является scaling law для looped LM.

### 8.3 Checkpoint и resume

Checkpoint сохраняет:

- model, optimizer, scheduler и scaler;
- RNG CPU/CUDA;
- sampler/data cursor и tokens_seen;
- config и commit hashes кода;
- dataset manifest/hash;
- tokenizer hash;
- метрики и hardware fingerprint.

Запись должна быть атомарной. До длинного запуска провести искусственное прерывание и доказать bitwise/near-bitwise продолжение следующего batch. Для Kaggle заранее выбрать проверенный persistent target: versioned Kaggle Dataset output либо объектное хранилище через Kaggle Secrets. Локальный notebook disk не считать persistent.

---

## 9. Токенизатор

Track B не наследует tokenizer Ouro или LAMBA.

Обучить два byte-fallback кандидата, 32k и 48k, на репрезентативной смеси EN/RU/code/math. Выбор делается по held-out:

- bytes/token по каждому домену и языку;
- chars/token внутри одного языка;
- tokens/word только как диагностическая метрика;
- доля truncation при фиксированном context;
- exact byte round-trip;
- стоимость embedding/LM head;
- устойчивость на Unicode, кириллице, коде и пробелах.

Нельзя выбирать vocab только по английской perplexity. Итоговый tokenizer фиксируется до основных сравнительных прогонов; любые изменения tokenizer аннулируют прямое сравнение loss.

---

## 10. Данные и лицензии

### 10.1 Предлагаемая pretraining-смесь

Стартовая гипотеза, а не установленный optimum:

| Домен | Доля |
|---|---:|
| English educational/web | 40% |
| Russian/multilingual | 20% |
| Code | 25% |
| Math | 15% |

Доли корректируются после tokenizer fertility и domain validation. Agent traces, tool-use и длинные reasoning trajectories относятся к post-training, а не должны заменять базовый pretraining corpus.

### 10.2 Кандидаты

| Dataset | Проверенный статус | Решение |
|---|---|---|
| FineWeb-Edu | English, ODC-By-1.0; источник Common Crawl | Базовый EN |
| FineWeb2, rus_Cyrl | Multilingual, ODC-By-1.0; источник Common Crawl | Кандидат RU |
| FineMath 4+ | English math subset, ODC-By-1.0 | Кандидат math |
| Stack-Edu / The Stack v2 | Файловые лицензии и provenance различаются | Только после license allowlist |
| OpenThoughts3 | Apache-2.0 metadata, synthetic reasoning | Поздний post-training после source audit |
| Nemotron-SFT-SWE-v3 | CC-BY-4.0 | Post-training с attribution |
| Nemotron-SFT-OpenCode-v1 | CC-BY-4.0 | Post-training с attribution |
| Nemotron-SFT-Math-v4 | CC-BY и CC-BY-SA по sample | Разделять по лицензии |
| Nemotron-SFT-ARC-AGI-v1 | Карточка содержит противоречивый/pending legal status | Исключить до разъяснения |

Для Common Crawl производных сохранить provenance и соблюдать Terms of Use. Для code:

- allowlist разрешённых лицензий;
- per-file attribution/provenance;
- удаление secrets, PII и generated/vendor/minified duplicates;
- регулярное применение removal lists;
- отдельная юридическая проверка до коммерческого release.

Перед обучением создать immutable manifest: dataset revision, query/config, license, shard hashes, фильтры, dedup version, число документов/токенов и rejected counts. Eval/private holdout hashes фиксируются до train.

---

## 11. Оценка

### 11.1 Метрики

- validation CE и bits/byte по EN, RU, code, math;
- качество при T = 1…8;
- tokens/s, peak VRAM, prefill и decode latency;
- SSM/KV cache bytes на sequence при каждом T;
- нормы состояния/градиента и spectral radius A_bar;
- associative recall, parity/modular state, state tracking;
- NIAH/RULER-подобные long-context тесты;
- общие LM, code и math eval с contamination audit.

Для tokenizer-независимого сравнения основной metric корпуса — bits/byte, не только perplexity.

### 11.2 Overthinking

До эксперимента выбрать score s_i(t), допустимое падение δ и набор задач:

~~~text
OTR(T) = mean_i[ max_{t<T} s_i(t) - s_i(T) > δ ]
~~~

Нужно строить quality-vs-T curve, а не сообщать только лучший T. Решение о Q-head/halting принимается лишь при устойчивом overthinking на нескольких seed.

### 11.3 Go / no-go

Переход 30–40M → 140–160M разрешён, если:

1. R1 воспроизводит ожидаемое направление Parcae относительно R0;
2. нет NaN/divergence на трёх seed;
3. cached decode совпадает с full forward;
4. resume stress-test пройден;
5. M0, если используется, даёт выигрыш по заранее выбранной quality/active-FLOP или quality/latency метрике;
6. random T не ухудшает общий validation и улучшает устойчивость по глубине;
7. data/license manifest завершён.

Точные минимальные эффекты и допустимые регрессии нужно пререгистрировать после R0 profiling, а не подгонять после результата.

---

## 12. Roadmap

### Phase 0 — воспроизводимость и hardware, 2–4 дня

- закрепить commits Parcae и official Mamba;
- сохранить license/SBOM manifest;
- измерить Kaggle hardware;
- запустить официальный SISO gate;
- отдельно проверить Tiny-MIMO как packaging fixture;
- реализовать full-vs-cached parity и resume test.

**Выход:** отчёт совместимости и решение SISO / fallback.

### Phase 1 — tokenizer и controls, около недели

- обучить 32k/48k tokenizer candidates;
- выбрать по EN/RU/code/math;
- обучить одинаковым бюджетом 30–40M R0 и R1;
- воспроизвести expected recurrent-depth behavior.

**Выход:** надёжный GQA recurrent baseline.

### Phase 2 — mixer ablation

- M0: заменить только Core mixer на Mamba 3 SISO;
- затем M1 random T;
- бюджет каждого раннего прогона ограничить 100–300M токенов;
- остановить ветку при проигрыше controls по active FLOPs и latency.

**Выход:** выбранный Core.

### Phase 3 — Prototype P, 140–160M

- context 1024, затем 2048;
- T = 2–4, E[T] около 3;
- до 1B токенов только при подтверждённом runtime budget;
- три seed для решающих сравнений;
- без MIMO, Q-head, MoE, depth-LoRA и distillation.

**Выход:** первая собственная base LM.

### Phase 4 — post-training

- instruction, code/math и tool-use data с provenance;
- опциональный online/top-k-small distillation;
- отдельные абляции loop consistency, intermediate exits и Q-head.

### Phase 5 — 300–500M

Только после положительного scaling 30M → 150M. Потребуется более стабильный rented/multi-GPU контур; Kaggle остаётся smoke/eval средой.

### Phase 6 — 1–1.5B и затем до 4B

- собрать scaling fit по loss, active FLOPs, latency и T;
- зафиксировать architecture before scale;
- перейти на distributed training;
- рассматривать 4B только при доказанном преимуществе recurrent Core и обеспеченном corpus/compute.

---

## 13. Что исключено из v0

- fine-tune или surgery чужого checkpoint;
- Ouro/LAMBA/Tiny embeddings, tokenizer или weights;
- Mamba 3 MIMO как обязательный компонент;
- MoE;
- ACT/per-token dynamic halting;
- Q-head и TRM-style correction;
- depth-LoRA;
- несколько attention refresh внутри Core;
- offline top-k512 distillation;
- обучение 4B на Kaggle;
- одновременное добавление нескольких непроверенных механизмов.

Возвращать эти идеи можно только отдельными абляциями после стабильного Prototype P.

---

## 14. Engineering layout

~~~text
murmur/
  configs/
  data/
    manifests/
  tokenizer/
  murmur/
    model/
      prelude.py
      recurrent_core.py
      coda.py
      mixers/
        gqa.py
        mamba3_siso.py
    cache.py
    train.py
    evaluate.py
  scripts/
    hardware_probe.py
    param_count.py
    resume_test.py
  tests/
    test_weight_sharing.py
    test_cache_parity.py
    test_gradient_parity.py
    test_stability.py
~~~

Минимальные CI-тесты:

- один и тот же Core object реально разделяет параметры между T;
- runtime cache разделён по loop occurrence;
- T=1 эквивалентен одному проходу;
- checkpointed/uncheckpointed gradients совпадают на tiny model;
- full/cached logits parity;
- tokenizer exact round-trip;
- deterministic resume;
- safetensors round-trip.

---

## 15. Нерешённые решения

До старта Phase 1:

1. доступный Kaggle GPU и стабильная precision;
2. final tokenizer 32k или 48k;
3. fallback mixer при провале Mamba 3;
4. точный R0/R1 token budget;
5. persistent checkpoint target;
6. допустимые thresholds quality/FLOP и quality/latency.

До масштабирования:

1. нужен ли internal attention refresh;
2. оптимальный train/test T;
3. есть ли overthinking;
4. нужен ли KD;
5. scaling slope и реальная стоимость 300M+.

---

## 16. Проверенные первичные источники

Архитектуры и код:

- [Anthropic: Claude Fable 5 and Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5)
- [OpenMythos repository](https://github.com/kyegomez/OpenMythos)
- [Parcae paper](https://arxiv.org/abs/2604.12946) и [repository](https://github.com/sandyresearch/parcae)
- [Mamba 3 paper](https://arxiv.org/abs/2603.15569), [official repository](https://github.com/state-spaces/mamba), [Mamba 3 module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py) и [MIMO kernel](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py)
- Official code references: [Mamba‑1 selective scan interface](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py), [Mamba‑2 minimal SSD](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py), [Mamba 1/2/3 LM scaffold](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py)
- [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B)
- [LT² paper](https://arxiv.org/abs/2605.20670) и [repository](https://github.com/chili-lab/LT2)
- [TRM repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) и [paper](https://arxiv.org/abs/2510.04871)

Проверенные community-модели:

- [Mamba3-MIMO-Tiny-HF](https://huggingface.co/aifeifei798/Mamba3-MIMO-Tiny-HF), [config](https://huggingface.co/aifeifei798/Mamba3-MIMO-Tiny-HF/blob/main/mamba3_hf_ready/config.json), [model code](https://huggingface.co/aifeifei798/Mamba3-MIMO-Tiny-HF/blob/main/mamba3_hf_ready/modeling_mamba3.py), [train script](https://huggingface.co/aifeifei798/Mamba3-MIMO-Tiny-HF/blob/main/1.train_mamba3.py)
- [Mamba3-177M-GQA-Hybrid LAMBA](https://huggingface.co/kdirgul/Mamba3-177M-GQA-Hybrid_LAMBA_V1.0), [CPU code](https://huggingface.co/kdirgul/Mamba3-177M-GQA-Hybrid_LAMBA_V1.0/blob/main/lamba_cpu.py), [checkpoint files](https://huggingface.co/kdirgul/Mamba3-177M-GQA-Hybrid_LAMBA_V1.0/tree/main/checkpoints), [binary wheels](https://huggingface.co/kdirgul/Mamba3-177M-GQA-Hybrid_LAMBA_V1.0/tree/main/wheels)

Данные:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) и [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
- [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- [Nemotron-SFT-SWE-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3)
- [Nemotron-SFT-OpenCode-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1)
- [Nemotron-SFT-Math-v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4)
- [Nemotron-SFT-ARC-AGI-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1)

---

## 17. Краткий вердикт

Самый быстрый путь к собственной рабочей архитектуре — не объединить все идеи сразу, а построить чистую лестницу доказательств:

**Parcae GQA baseline → Mamba 3 SISO Core → random recurrent depth → 140–160M prototype → post-training → scaling.**

Tiny-MIMO доказывает лишь возможность минимальной упаковки. LAMBA показывает интересную fixed-depth комбинацию SISO+GQA, но её веса блокируются лицензией и недостаточной воспроизводимостью. Ни одна из них не отменяет необходимость обучать Track B с нуля.
