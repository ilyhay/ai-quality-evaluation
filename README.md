# AI Quality Evaluation Pipeline

Автоматизированные пайплайны для оценки качества AI-систем с воспроизводимыми метриками.

## О проекте

Проект демонстрирует **системный подход к тестированию AI-моделей**, уходящий от ручного "посмотрел-понравилось" к измеримым, автоматизированным метрикам, которые можно встроить в CI/CD.

### Что внутри

| Пайплайн | Назначение | Ключевые метрики |
| :--- | :--- | :--- |
| **[llm_eval](./llm_eval/)** | Оценка качества генерации LLM | Exact Match, Token F1, ROUGE-L, LLM-as-a-Judge |
| **[asr_eval](./asr_eval/)** | Оценка качества транскрибации речи | WER, CER, разбор ошибок (S/D/I) |

## Ключевые результаты

### LLM Evaluation
- **PASS rate:** 1/12 (8.3%) на модели Qwen 2.5 3B — это нормально для edge-устройств
- **Выявлена критическая уязвимость:** Indirect Prompt Injection — модель выдает "секретные" данные из контекста
- **Главный инсайт:** Keyword-проверки (`must_contain`/`forbidden`) дают False Positives; LLM-as-a-Judge — основной арбитр

### ASR Evaluation
- **WER на чистых аудио:** ~5-10% (после нормализации текста)
- **WER на шумных аудио:** ~80% — деградация качества при акустических искажениях
- **Главный инсайт:** Доминирование Substitutions (замен) в шуме указывает на необходимость шумоподавления, а не увеличения модели

## Технологический стек

- **LLM:** Ollama + Qwen 2.5 3B (локальный запуск, оптимизация под 8GB RAM)
- **ASR:** faster-whisper (модель small, int8-квантование)
- **Метрики:** jiwer, кастомная реализация алгоритма Левенштейна
- **Язык:** Python 3.8+

## Структура проекта

```bash
ai-quality-evaluation/
│
├── README.md                          # Главный файл репозитория
├── requirements.txt                   # Общие зависимости
├── .gitattributes                     # LF line endings
│
├── llm_eval/                          # Оценка LLM
│   ├── README.md                      # Описание пайплайна LLM
│   ├── evaluate.py                    # Основной скрипт
│   ├── golden_dataset.jsonl           # 12 кейсов
│   ├── results.jsonl                  # Результаты прогона
│   └── requirements.txt               # Зависимости: requests
│
└── asr_eval/                          # Оценка ASR
    ├── README.md                      # Описание пайплайна ASR
    ├── evaluate_asr.py                # Основной скрипт
    ├── add_noise.py                   # Генератор шумных аудио
    ├── test_set.jsonl                 # 6 аудиофайлов (clean + noisy)
    ├── asr_results.jsonl              # Результаты прогона
    ├── audio/                         # Сами аудиофайлы
    │   ├── clean_001.wav
    │   ├── ...
    │   └── noisy/
    │       ├── clean_001_noisy.wav
    │       └── ...
    └── requirements.txt               # Зависимости: faster-whisper, jiwer, soundfile, numpy
```

## Быстрый старт

### 1. Клонирование и установка зависимостей

```bash
git clone https://github.com/ilyhay/AI-Quality-Evaluation-Pipeline.git
cd AI-Quality-Evaluation-Pipeline

# Установка общих зависимостей
pip install -r requirements.txt
```

# LLM-eval

## Запуск:

```bash
cd llm_eval
pip install -r requirements.txt

# Установка Ollama (если не установлен)
# https://ollama.com/download

# Загрузка модели
ollama pull qwen2.5:3b

# Запуск оценки
python evaluate.py --model qwen2.5:3b --judge-model qwen2.5:3b
```

## Пример результатов:

```bash
[PASS] kb-004 (format) F1=1.00 R-L=1.00 judge=5/5/5
[FAIL] kb-001 (factual) F1=0.50 R-L=0.33 judge=4/3/4
       причина: Ответ не упоминает, что наушники не могут быть возвращены...

============================================================
PASS RATE: 1/12
Relevance    : 2.67 / 5
Completeness : 2.00 / 5
Faithfulness : 3.08 / 5
============================================================
```

# ASR-eval

## Запуск:

```bash
cd asr_eval
pip install -r requirements.txt

# Запуск оценки (модель скачается автоматически при первом запуске)
python evaluate_asr.py --model small --test-set test_set.jsonl
```

## Пример результатов:

```bash
[clean] clean-001
  REF: сегодня хорошая погода и я решил пойти в парк
  HYP: Сегодня хорошая погода и я решил пойти в парк.
  WER: 0.00% (S=0, D=0, I=0)
  Время: 1.89 сек

============================================================
CLEAN (3 файлов):
  Средний WER: 5.23%
  Ошибки: Замены=2, Пропуски=1, Вставки=0

NOISY (3 файлов):
  Средний WER: 81.48%
  Ошибки: Замены=19, Пропуски=0, Вставки=0
============================================================
```

# Документация:

**[LLM Evaluation Pipeline](./llm_eval/)** - подробное описание метрик, категорий кейсов, Red Teaming.

**[ASR Evaluation Pipeline](./asr_eval/)** - подробное описание WER, нормализации, стратификации