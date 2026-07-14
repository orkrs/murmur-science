# Датасеты для Murmur: исследование и отбор на 14 июля 2026 года

## Краткий итог

Для первого полноценного обучения Murmur на одной NVIDIA T4 не нужен один гигантский датасет. Нужна воспроизводимая смесь небольших потоковых выборок из нескольких качественных источников.

Рекомендуемый основной набор источников:

1. **Английский образовательный текст:** [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), конфигурация `sample-10BT` или отдельные свежие Common Crawl snapshots.
2. **Русский текст:** [`epfml/FineWeb2-HQ`](https://huggingface.co/datasets/epfml/FineWeb2-HQ), конфигурация `rus_Cyrl`.
3. **Код:** [`common-pile/stackv2_edu_filtered`](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered).
4. **Математика:** [`HuggingFaceTB/finemath`](https://huggingface.co/datasets/HuggingFaceTB/finemath), конфигурация `finemath-4plus`.
5. **Наука:** [`common-pile/peS2o_filtered`](https://huggingface.co/datasets/common-pile/peS2o_filtered).

Это лучшая стартовая комбинация по совокупности качества, прозрачности происхождения, доступности через streaming и пригодности для маленькой базовой модели. Она предпочтительнее скачивания случайного готового «mega mix» с неясным составом.

Рекомендуемая смесь для первой модели Murmur Smoke, 27,9 млн уникальных параметров:

| Сегмент | Доля токенов | Источник |
|---|---:|---|
| Английский образовательный и общий текст | 35% | FineWeb-Edu |
| Русский текст | 25% | FineWeb2-HQ `rus_Cyrl` |
| Код | 20% | Common Pile Stack V2 Edu Filtered |
| Математика | 12% | FineMath-4+ |
| Научные тексты | 8% | Common Pile peS2o Filtered |

Текущие `10M max_tokens` в `smoke_gqa.toml` — только проверка исправности пайплайна. Первая содержательная сравнительная тренировка должна использовать не менее **100–300 млн токенов**. После подтверждения стабильности можно провести длинный прогон до **0,5–1 млрд токенов**. Для Prototype-модели около 140 млн параметров целевой состав можно вернуть к исследовательской пропорции 40% обычного текста, 20% русского, 25% кода и 15% математики/науки.

## Область исследования

Срез выполнен на **14 июля 2026 года**. Проверялись официальные dataset cards, публикации и лицензии создателей наборов. В список включены сильные публичные наборы для:

- базового causal language modeling с нуля;
- русского и английского языков;
- кода, математики и научных текстов;
- последующего SFT/reasoning/agentic post-training.

Невозможно буквально перечислить все пользовательские загрузки Hugging Face: их число постоянно меняется, а многие не имеют проверяемого происхождения. Поэтому ниже приведён широкий, но отборный каталог наборов с официальным владельцем, документированным процессом подготовки и практической ценностью для Murmur.

Статусы:

- **A — брать:** подходит в основной воспроизводимый пайплайн.
- **B — условно:** ценный источник, но нужна дополнительная обработка или проверка прав.
- **C — отложить:** лицензионные, технические или provenance-риски выше пользы для первого запуска.
- **SFT — только post-training:** нельзя смешивать с сырым pretraining без отдельного эксперимента.

## 1. Лучшие наборы для базового pretraining

### 1.1 Общий и образовательный текст

| Набор | Масштаб и версия | Лицензия/условия | Решение для Murmur |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1,3T токенов; v1.4.0 добавляет crawls по июнь 2025; есть `sample-10BT`, `100BT`, `350BT` | ODC-By 1.0 + Common Crawl Terms | **A.** Главный английский источник. Высокая образовательная плотность; streaming готов. |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 18,5T очищенных английских токенов | ODC-By + Common Crawl Terms | **B.** Полезен для разнообразия, но FineWeb-Edu эффективнее при ограниченном token budget. |
| [DCLM Baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | около 4T токенов / 3 млрд документов; 7,2 TB | CC BY 4.0 | **B.** Сильный альтернативный web-корпус и хороший источник для ablation, но не нужен одновременно с большим объёмом FineWeb-Edu в первом прогоне. |
| [Dolma 3 Mix 6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T) | около 6T токенов; смесь, использованная для OLMo 3 32B | ODC-By; заявлено research/educational use | **A/B.** Отличный готовый английский mix для воспроизведения, но огромен и частично пересекается с нашими специализированными источниками. |
| [Dolma 3 Pool](https://huggingface.co/datasets/allenai/dolma3_pool) | 9,31T токенов: 8,14T web, 972B science PDF, 137B code, 34,1B math и другие источники | ODC-By; research/educational use | **B.** Брать отдельные категории или качественные bins, а не весь pool. |
| [Bolmo Mix](https://huggingface.co/datasets/allenai/bolmo_mix) | компактная готовая смесь 172,7B токенов: web, science PDF, StackEdu, FineMath, arXiv, Wikipedia | ODC-By; research/educational use | **A/B.** Хороший воспроизводимый технический English-backbone и контрольная смесь; русский нужно добавлять отдельно. |
| [Common Pile / Comma v0.1 training dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) | 784M документов, 521 GB; консолидированный filtered mix из 30+ источников | открытые/общественное достояние, per-source лицензии; card помечен MIT | **A.** Лучший вариант, если приоритет — максимально чистое лицензирование и разнообразие. Удобнее использовать отдельные компоненты, чтобы управлять пропорциями. |
| [Dolma v1.7](https://huggingface.co/datasets/allenai/dolma) | около 2,3T токенов в v1.7, смесь web/code/science/books/encyclopedia | ODC-By | **B.** Надёжный старый baseline; Dolma 3 современнее. |

**Решение:** в первом Murmur run использовать FineWeb-Edu. Dolma 3 Mix, Bolmo Mix и DCLM оставить для последующих сравнительных запусков; одновременно смешивать все крупные web-корпуса бессмысленно из-за пересечений и ограниченного бюджета T4.

### 1.2 Русский и multilingual

| Набор | Масштаб и свойства | Лицензия/условия | Решение для Murmur |
|---|---|---|---|
| [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ), `rus_Cyrl` | 55 220 956 русских документов; top 10% FineWeb2 по model-based quality; авторы сообщают примерно шестикратную экономию pretraining-токенов | ODC-By + Common Crawl Terms | **A.** Главный русский источник для T4. Предпочтительнее полного FineWeb2. |
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), `rus_Cyrl` | v2.1.1; русский subset: около 588,6 млрд слов, 699 млн документов, 1,81 TB на диске | ODC-By + Common Crawl Terms | **A/B.** Основной резерв для расширения и дополнительной собственной фильтрации. Не использовать `test` split. |
| [HPLT 3.0](https://huggingface.co/datasets/HPLT/HPLT3.0), `rus_Cyrl` | релиз July 2025; 198 language-script pairs, 50 TB compressed; файлы отсортированы по WDS quality bins | CC0 относится к упаковке; авторы прямо не владеют исходными текстами | **B.** Очень ценный независимый от FineWeb2 источник. Для T4 брать только WDS 10 и 9. Важно: русский, английский и китайский в v3 не прошли глобальную дедупликацию — Murmur должен дедуплицировать сам. |
| [HPLT 2.0 Cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | около 8T токенов, 193 языка; в русских ablations был конкурентен FineWeb2 | web-content terms | **C.** Заменён HPLT 3.0; оставлять только для воспроизводимости старых экспериментов. |
| Официальные [Wikimedia dumps](https://dumps.wikimedia.org/) / [HF Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | высокоструктурированный энциклопедический русский текст | CC BY-SA/GFDL, нужна атрибуция и соблюдение share-alike условий | **B.** Добавить 2–5% как структурированный knowledge anchor, если ведём корректный attribution manifest. |

**Решение:** `FineWeb2-HQ/rus_Cyrl` — первая линия. Позже добавить верхние WDS bins HPLT 3.0 как независимый источник и провести source ablation. Полный FineWeb2 нужен только после того, как HQ subset окажется недостаточным.

### 1.3 Код

| Набор | Масштаб и свойства | Лицензия/условия | Решение для Murmur |
|---|---|---|---|
| [Common Pile Stack V2 Edu Filtered](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered) | 69,6M документов, 255 GB raw / 83 GB filtered files; actual text, per-document license metadata | только репозитории с лицензиями из сертифицированного Blue Oak списка; возможны ошибки детектора | **A.** Главный кодовый источник: проще и юридически чище исходного Stack v2. |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 125B токенов в 15 языках программирования; quality classifier; улучшал MultiPL-E | наследует условия The Stack v2; на HF лежат SWH IDs, содержимое нужно rehydrate из Software Heritage | **B.** Очень высокое качество, но сложнее загрузка и соблюдение per-file лицензий. |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2), актуальный usable release | более 3B файлов, 600+ языков, около 900B train-токенов | gated; исходные лицензии каждого файла, атрибуция, opt-out updates, Software Heritage terms | **C.** Не брать целиком в первый run. Только свежий разрешённый revision и только permissive subset после собственной проверки. |
| [NVIDIA Nemotron Pretraining Code v3](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3) | June 2026; metadata для 146,3M новых файлов, около 173B токенов нового GitHub-кода до 2025-09-30 | CC BY 4.0 для metadata | **B/C.** Самый свежий крупный source-code update, но это metadata, а не готовый текст. Нужны v1/v2, hydration и проверка лицензий. |
| [NVIDIA Nemotron Pretraining Code v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | около 340B новых GitHub-токенов + synthetic QA/review/rewrite/transpile | gated NVIDIA Data Agreement; model-training-only; возможны обязательства Qwen/DeepSeek/Phi | **C.** Только после отдельной юридической проверки; не включать в open-default pipeline. |
| [NVIDIA Nemotron CC Code v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | 427,9B токенов документации/кода из Common Crawl, LLM-cleaning | NVIDIA Data Agreement; third-party model terms | **C.** Качественный, но лицензионно сложный. |

**Решение:** использовать Common Pile Stack V2 Edu Filtered. Он уже содержит actual text и открыто-лицензионный фильтр. Stack-Edu оставить как второй источник для ablation. NVIDIA v3 — перспективный future source, но не drop-in датасет.

### 1.4 Математика

| Набор | Масштаб и свойства | Лицензия/условия | Решение для Murmur |
|---|---|---|---|
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), `finemath-4plus` | 9,6B токенов / 6,7M документов; high-quality subset; 13-gram decontamination против GSM8K, MATH, MMLU и ARC | ODC-By + Common Crawl Terms | **A.** Главный math pretraining source. Для маленькой модели 4+ лучше 3+. |
| FineMath `finemath-3plus` | 34B токенов / 21,4M документов | те же условия | **B.** Добавлять после исчерпания разнообразия 4+. |
| FineMath `infiwebmath-4plus` | 8,5B токенов / 6,3M документов | ODC-By + исходные условия | **B.** Источник дополнительного разнообразия; проверить пересечение с FineMath. |
| [NVIDIA Nemotron CC Math v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | 133B токенов 3+, 52B токенов 4+; Lynx extraction, LLM cleanup, dedup и decontamination | gated NVIDIA Open Data Agreement; возможны Phi-4 redistribution requirements | **C.** Технически сильный, но не нужен до юридической проверки. |

**Решение:** FineMath-4+ без дополнительных reasoning traces на базовом этапе. Решения задач и chains of thought относятся к SFT, а не к исходному pretraining mix.

### 1.5 Наука, книги и энциклопедические данные

| Набор | Масштаб и свойства | Лицензия/условия | Решение для Murmur |
|---|---|---|---|
| [Common Pile peS2o Filtered](https://huggingface.co/datasets/common-pile/peS2o_filtered) | 6,12M открыто лицензированных научных статей, 182,6 GB raw / 56,9 GB files; Grobid structure and quality filtering | per-document open licenses; provenance metadata | **A.** Главный science source. |
| [Common Pile Comma components](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) | arXiv, PubMed, DOAB, LibreTexts, OER Commons, Project Gutenberg, public-domain books, Wikimedia и др. | public domain/open licenses с per-source условиями | **A.** Выбирать отдельные компоненты по необходимости; не дублировать peS2o и Stack V2 Edu. |
| [Dolma 3 olmOCR Science PDFs](https://huggingface.co/datasets/allenai/dolma3_pool) | до 972B токенов science PDFs в pool | ODC-By; source/content rights всё равно требуют внимания | **B.** Сильный большой источник, но избыточен для первой T4-тренировки. |
| [olmOCR Mix 1025](https://huggingface.co/datasets/allenai/olmOCR-mix-1025) | 270 250 PDF pages, OCR с сохранением формул/таблиц | ODC-By; research/educational use | **B.** Полезен для качества PDF/разметки, но это в первую очередь OCR-training set, не готовый большой LM corpus. |
| [Common Pile Project Gutenberg / pre-1929 books](https://huggingface.co/common-pile) | художественная и нехудожественная public-domain литература | public domain, но юрисдикция и metadata всё равно проверяются | **B.** Не более 2–5%: улучшает long-form style, но повтор книг повышает memorization. |

## 2. Свежие специализированные pretraining-наборы NVIDIA 2026

Эти наборы важны для дальнейших экспериментов, но не входят в чистый первый pipeline.

| Набор | Дата/размер | Назначение | Решение |
|---|---|---|---|
| [Nemotron Pretraining Specialized v1.2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.2) | May/June 2026; около 599,5M samples, 53,6 GB | synthetic factual, moral, generative и multiple-choice data | **C.** CC BY на samples, но ответы созданы DeepSeek/Qwen/Mixtral; проверить downstream model terms. Использовать только отдельным ablation. |
| [Nemotron Pretraining Legal v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Legal-v1) | May/June 2026; 9,6M samples, 6,99 GB | English legal specialization | **B.** CC BY 4.0; нужен только если Murmur получает legal-domain цель. |
| [Nemotron CC v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | около 6,6T токенов web/synthetic/math/code/SFT | large all-in-one pretraining mix | **C.** NVIDIA training-data agreement, запрет на свободное перераспространение данных и возможные teacher-model obligations. Исключить из open track. |

Важно: метка «ready for commercial use» в dataset card не заменяет чтение полного соглашения. Для Murmur нужен отдельный `license_policy` и список разрешённых лицензий; gated NVIDIA data не должны случайно попасть в общий экспорт.

## 3. Лучшие наборы для post-training, не для базового pretraining

Сначала обучается base model на обычном causal text. Затем создаётся отдельный checkpoint и выполняется SFT. Смешивание диалоговых шаблонов, verifier traces и agent trajectories в базовый corpus затрудняет оценку архитектуры и может испортить обычное продолжение текста.

### 3.1 Общий instruction tuning и русский

| Набор | Размер/дата | Лицензия | Решение |
|---|---|---|---|
| [T-Wix](https://huggingface.co/datasets/t-tech/T-Wix) | около 499,6K русских samples: 468,6K general + 31K reasoning; EACL 2026 | ODC-By; third-party outputs могут иметь отдельные terms | **SFT, условно.** Лучший крупный готовый русский mix из рассмотренных. Проверить лицензии каждого source subset и teacher outputs. |
| [Dolci Instruct SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) | 2,152,112 samples, 67+ языков; OLMo 3 post-training | ODC-By; research/educational use | **SFT.** Сильный открытый общий multilingual mix. Русскую долю нужно измерить до использования. |
| [Nemotron Post-Training Dataset v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | 1M–10M rows, multilingual, 2025 | CC BY 4.0, gated acknowledgement | **SFT, условно.** Ценный общий mix, но сначала проверить provenance и условия доступа. |

Небольшие русские community-наборы без подробного provenance не должны становиться основой SFT. Их можно использовать только после ручной выборочной оценки и проверки лицензии.

### 3.2 Reasoning и математика

| Набор | Размер/дата | Лицензия | Решение |
|---|---|---|---|
| [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | 1,2M math/code/science prompts с reasoning traces; June 2025 | проверить dataset card и исходные source licenses; проект публикует pipeline | **SFT.** Сильная проверенная reasoning recipe, но слишком длинные traces нужно фильтровать для 28M модели. |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 220K задач, 2–4 DeepSeek-R1 traces; verified answers; Apache 2.0 | Apache 2.0, плюс проверить условия teacher output | **SFT.** Использовать `default`, а не `extended`: card сообщает лучшую SFT performance у default. |
| [Nemotron Math Proofs v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v2) | May 2026; 82 737 traces / 5 752 problems / около 5B токенов | CC BY 4.0 | **SFT, позднее.** Proof, verification и meta-verification; слишком специализирован и длинен для первой маленькой модели. |

### 3.3 Кодовые и агентные траектории

| Набор | Размер/дата | Лицензия | Решение |
|---|---|---|---|
| [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | June 2026; 207 489 OpenHands/SWE-agent trajectories | CC BY 4.0 + permissive repo licenses | **SFT.** Один из лучших свежих открытых источников для software-engineering agents. Не нужен базовой модели. |
| [Nemotron SFT SWE v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | June 2026; 237 970 samples, 11,7 GB | CC BY 4.0 + Apache/MIT/BSD source licenses | **SFT.** Сильная альтернатива Open-SWE-Traces; нужна дедупликация между ними. |
| [OpenThoughts Agent SFT 100K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K) | June 2026; 100K multi-turn terminal/code trajectories | Apache 2.0 | **SFT.** Самый свежий компактный агентный кандидат; подходит только после общего instruction tuning. |

Для Murmur Smoke agentic SFT пока преждевременен: контекст 512 и 27,9M параметров слишком малы для длинных repository trajectories. Эти данные имеют смысл после увеличения контекста и подтверждения базовой языковой способности.

## 4. Что сознательно не включать в первый запуск

1. **Eval и benchmark test splits:** MMLU, ARC, GSM8K, HumanEval, MBPP, SWE-bench, RuBLiMP и другие тесты нельзя подмешивать в train. Они хранятся отдельно и участвуют в decontamination.
2. **Случайные community mega-mixes:** объединённая лицензия верхнего репозитория не отменяет лицензии исходных данных и teacher-моделей.
3. **Книги неизвестного происхождения, пиратские библиотеки, lyrics, paywalled news:** высокий copyright и memorization risk.
4. **Telegram, форумы и социальные сети без PII-фильтра:** риск персональных данных, токсичности и низкого signal-to-noise.
5. **Сырые The Stack / GitHub dumps без license allowlist:** наличие публичного URL не означает разрешение на обучение или перераспространение.
6. **Synthetic data без указанного teacher model и промптов:** невозможно оценить качество, ограничения лицензии и степень model collapse.
7. **Nemotron CC v2 / Code v2 в open-default track:** технически ценные, но gated agreements и teacher-license obligations делают pipeline менее переносимым.

## 5. Лицензионная интерпретация

Этот документ не является юридической консультацией. Практически важны следующие различия:

- **ODC-By и CC0 у web-корпуса часто лицензируют базу/упаковку, а не авторские права на каждый исходный текст.** FineWeb/FineMath дополнительно подчиняются Common Crawl Terms; HPLT прямо указывает, что не владеет исходными текстами.
- **CC BY требует атрибуции.** В manifest нужно сохранять dataset, revision, URL/provenance и license metadata.
- **Код наследует лицензию файла или репозитория.** Даже permissive лицензии могут требовать сохранения copyright notice и текста лицензии.
- **Лицензия синтетического датасета и лицензия teacher model — не одно и то же.** Некоторые NVIDIA cards прямо предупреждают о возможных Qwen, DeepSeek или Phi redistribution requirements.
- **Publicly accessible не значит public domain.** Это особенно важно для web, GitHub и форумов.

Минимальная policy для проекта:

```text
allow_default = [public-domain, CC0, CC-BY, Apache-2.0, MIT, BSD-2, BSD-3,
                 ODC-By datasets with recorded source terms]
conditional   = [CC-BY-SA, ODC-By web content, mixed per-file code licenses]
deny_default  = [unknown, no-license, NC, ND, gated model-training agreement,
                 source terms not recorded]
```

## 6. Точный план формирования артефактов Murmur

### 6.1 `corpus.txt` для токенизатора

Токенизатор должен видеть тот же доменный баланс, что и модель. Для первого tokenizer corpus:

| Источник | Доля символов/байтов |
|---|---:|
| FineWeb-Edu | 35% |
| FineWeb2-HQ Russian | 25% |
| Common Pile Stack V2 Edu Filtered | 20% |
| FineMath-4+ | 12% |
| peS2o Filtered | 8% |

Практический объём: **2–5 GB чистого текста** достаточно для обучения SentencePiece BPE 32K с byte fallback. Документы перемешиваются, но их границы сохраняются пустой строкой. Нельзя строить tokenizer только на английском: это ухудшит token fertility русского и кода.

Перед фиксацией токенизатора измерить:

- tokens/character отдельно для русского, английского, Python, JavaScript и LaTeX;
- долю byte-fallback tokens;
- среднюю и p95 длину документа;
- покрытие кириллицы, Unicode punctuation и code symbols.

### 6.2 `train.jsonl`

Каноническая строка до packing:

```json
{"text":"...","source":"fineweb2_hq_ru","source_id":"...","url":"...","license":"odc-by-1.0","language":"ru","quality_score":0.91}
```

Для текущего `prepare_data.py` достаточно поля `text`, но остальные поля должны оставаться в сыром manifest или sidecar provenance-файле. Нельзя терять происхождение после экспорта.

Этапы обработки:

1. Загрузить источники потоково и закрепить `revision`/commit SHA.
2. Нормализовать Unicode в NFC, line endings и недопустимые control characters; не разрушать Markdown/LaTeX/indentation кода.
3. Отфильтровать пустые, слишком короткие, boilerplate, явный spam, secrets и PII.
4. Провести exact hash dedup внутри каждого источника.
5. Провести cross-source near-dedup, сохранив более качественный документ.
6. Удалить n-gram overlaps с evaluation corpus.
7. Сделать split **до token packing**, группируя по domain/URL/repository, чтобы один источник не попал одновременно в train и val.
8. Выгрузить JSONL, затем использовать существующий Murmur packer.

### 6.3 `val.jsonl`

Validation должен быть фиксированным и стратифицированным по тем же пяти доменам. Рекомендуется 20–50 млн токенов для длинных прогонов или минимум 10 000 документов для smoke/prototype разработки.

Дополнительно создать отдельные validation slices:

- `val_en.jsonl`;
- `val_ru.jsonl`;
- `val_code.jsonl`;
- `val_math.jsonl`;
- `val_science.jsonl`.

Это позволит увидеть, какой домен деградирует, даже когда общий validation loss улучшается.

## 7. Streaming API и автоматическая загрузка

Все пять основных источников можно читать через Hugging Face `datasets` без полного скачивания. Официальная документация streaming: [Hugging Face Datasets](https://huggingface.co/docs/datasets/stream).

```python
from datasets import load_dataset

sources = {
    "en": load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
        revision="PINNED_COMMIT_SHA",
    ),
    "ru": load_dataset(
        "epfml/FineWeb2-HQ",
        "rus_Cyrl",
        split="train",
        streaming=True,
        revision="PINNED_COMMIT_SHA",
    ),
    "code": load_dataset(
        "common-pile/stackv2_edu_filtered",
        split="train",
        streaming=True,
        revision="PINNED_COMMIT_SHA",
    ),
    "math": load_dataset(
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        split="train",
        streaming=True,
        revision="PINNED_COMMIT_SHA",
    ),
    "science": load_dataset(
        "common-pile/peS2o_filtered",
        split="train",
        streaming=True,
        revision="PINNED_COMMIT_SHA",
    ),
}
```

`revision` нельзя оставлять равным `main` для настоящего эксперимента. Подготовительный модуль должен записывать:

- repo ID, config и split;
- точный commit SHA;
- дату загрузки;
- лицензию и URL card/license;
- число просмотренных, принятых и отклонённых документов;
- число символов и токенов после фильтрации;
- SHA-256 итоговых shards.

Текущий Murmur downloader должен быть отдельным модулем перед `prepare_data.py`: Hugging Face streaming → фильтрация/микширование → локальные `train.jsonl`, `val.jsonl`, `corpus.txt` → существующий tokenizer/packer. Так Kaggle notebook остаётся воспроизводимым и не зависит от ручной подготовки файлов.

## 8. Рекомендуемая последовательность экспериментов

### Experiment D0 — pipeline smoke

- 10M токенов, текущий config;
- по 2M документов не требуется: экспортировать только нужное число токенов;
- цель: проверить download, tokenizer, packing, loss, checkpoint и resume;
- результат не использовать для сравнения качества архитектур.

### Experiment D1 — первая содержательная base model

- 100–300M токенов;
- смесь 35/25/20/12/8;
- один фиксированный tokenizer и один frozen validation set;
- сравнить fixed GQA и recurrent GQA при одинаковом числе уникальных параметров и токенов.

### Experiment D2 — data-source ablation

- заменить FineWeb2-HQ RU на top WDS bins HPLT 3.0 RU;
- заменить FineWeb-Edu на Dolma 3/Bolmo/DCLM;
- заменить Common Pile code на Stack-Edu;
- остальные условия не менять.

### Experiment D3 — длинный Smoke run

- 0,5–1B токенов только после стабильного D1;
- использовать quality-weighted sampling и domain-specific validation;
- не менять смесь или tokenizer посреди запуска.

### Experiment D4 — post-training

- сохранить чистый base checkpoint;
- общий SFT: T-Wix RU + контролируемая multilingual часть Dolci;
- reasoning SFT: небольшая отфильтрованная часть OpenThoughts3/OpenR1;
- agentic data отложить до модели с большим контекстом и параметрами.

## 9. Финальный рейтинг

### Немедленно использовать

1. FineWeb2-HQ `rus_Cyrl` — лучший русский source для ограниченного compute.
2. FineWeb-Edu — лучший стартовый English educational web source.
3. Common Pile Stack V2 Edu Filtered — лучший практический code source с actual text и license metadata.
4. FineMath-4+ — лучший компактный math pretraining source.
5. Common Pile peS2o Filtered — лучший science source с открыто-лицензионным фильтром.

### Использовать во втором цикле

6. HPLT 3.0 top WDS Russian — независимое расширение русского, после собственной дедупликации.
7. Dolma 3 Mix / Bolmo Mix — контрольные готовые English technical mixtures.
8. Common Pile Comma components — книги, энциклопедии, PubMed, OER и другие специализированные добавки.
9. DCLM Baseline — сильный web ablation.
10. Stack-Edu — quality code ablation после настройки hydration/license handling.

### Только после юридической проверки или для отдельных исследований

11. Nemotron CC Math, CC Code, Code v2, CC v2 и Specialized v1.2.
12. The Stack v2 full.
13. HPLT raw lower-quality bins.

### Только post-training

14. T-Wix и Dolci Instruct.
15. OpenThoughts3, OpenR1-Math-220k, Nemotron Math Proofs v2.
16. Open-SWE-Traces, Nemotron SFT SWE v3, OpenThoughts Agent SFT 100K.

## 10. Решение для проекта

Первая версия Murmur должна обучаться не «на всём Hugging Face», а на закреплённом **Murmur Base Mix v0.1** из пяти основных источников. Такой mix:

- соответствует целям русский + английский + код + математика/наука;
- помещается в потоковый Kaggle pipeline;
- позволяет точно воспроизвести эксперимент;
- имеет понятную лицензионную карту;
- даёт отдельные доменные validation losses;
- допускает честные архитектурные ablations без смены данных.

Следующий кодовый шаг — реализовать `science/scripts/build_hf_corpus.py` и декларативный `science/configs/data/base_mix_v0_1.toml`, которые автоматически создают `corpus.txt`, `train.jsonl`, `val.jsonl`, provenance manifest и отчёт о фильтрации.
