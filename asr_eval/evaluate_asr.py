"""
ASR Evaluation Pipeline: транскрибация аудио + расчет WER.
Метрики: WER, CER, разбор ошибок (Substitutions, Deletions, Insertions).
"""
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel
import re
from jiwer import wer, cer

# Windows-совместимый вывод кириллицы
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Загрузка модели (один раз на весь прогон)
# ---------------------------------------------------------------------------
def load_model(model_size: str = "small"):
    """
    model_size: tiny, base, small, medium, large-v3
    Для 8GB RAM и CPU оптимально: small (460 МБ, баланс скорости/качества)
    """
    print(f"Загрузка модели Whisper ({model_size})...")
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device="cpu",           # используем CPU (нет GPU)
        compute_type="int8",    # квантование int8 — экономит RAM и ускоряет
    )
    print(f"Модель загружена за {time.time() - t0:.1f} сек.\n")
    return model


# ---------------------------------------------------------------------------
# 2. Транскрибация одного файла
# ---------------------------------------------------------------------------
def transcribe(model, audio_path: Path) -> str:
    segments, info = model.transcribe(
        str(audio_path),
        language="ru",
        beam_size=5,
        vad_filter=True,  # фильтрует тишину и паузы
    )
    text = " ".join(segment.text.strip() for segment in segments)
    # нормализация: убираем лишние пробелы
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 3. Загрузка тестового набора
# ---------------------------------------------------------------------------
def load_test_set(path: Path) -> list[dict]:
    """
    Формат test_set.jsonl:
    {"id": "clean-001", "category": "clean", "audio": "audio/clean_001.wav", "reference": "текст эталона"}
    """
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# 4. Подсчет ошибок через process_words
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Приводит текст к единому формату для честного сравнения:
    1. Нижний регистр
    2. Удаление всех знаков препинания
    """
    text = text.lower()
    # Удаляем все, кроме букв, цифр и пробелов
    text = re.sub(r'[^\w\s]', '', text)
    # Убираем лишние пробелы
    return " ".join(text.split())


def count_errors(reference: str, hypothesis: str) -> dict:
    """
    Расчёт S/D/I через выравнивание Левенштейна по словам.
    ВАЖНО: Текст предварительно нормализуется!
    """
    # Нормализуем оба текста перед сравнением
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    n, m = len(ref_words), len(hyp_words)

    # Матрица редактирующих расстояний
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # замена
                    dp[i - 1][j],      # пропуск
                    dp[i][j - 1],      # вставка
                )

    # Обратный проход: восстанавливаем тип каждой операции
    i, j = n, m
    subs = dels = ins = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return {
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "hits": hits,
    }


# ---------------------------------------------------------------------------
# 5. Основной прогон
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="tiny/base/small/medium")
    ap.add_argument("--test-set", default="test_set.jsonl")
    args = ap.parse_args()

    model = load_model(args.model)
    cases = load_test_set(Path(args.test_set))

    results = []
    print(f"Прогон {len(cases)} аудиофайлов...\n")

    for case in cases:
        audio_path = Path(case["audio"])
        if not audio_path.exists():
            print(f"[SKIP] {case['id']}: файл {audio_path} не найден")
            continue

        t0 = time.time()
        hypothesis = transcribe(model, audio_path)
        latency = time.time() - t0
        reference = case["reference"]
        
        # Расчет метрик НА НОРМАЛИЗОВАННОМ ТЕКСТЕ
        norm_ref = normalize_text(reference)
        norm_hyp = normalize_text(hypothesis)
        
        w = wer(norm_ref, norm_hyp)
        c = cer(norm_ref, norm_hyp)
        errors = count_errors(norm_ref, norm_hyp)

        record = {
            "id": case["id"],
            "category": case["category"],
            "audio": str(audio_path),
            "reference": reference,
            "hypothesis": hypothesis,
            "metrics": {
                "WER": round(w * 100, 2),
                "CER": round(c * 100, 2),
                **errors,
            },
            "latency_s": round(latency, 2),
        }
        results.append(record)

        # Вывод в консоль
        print(f"[{case['category']}] {case['id']}")
        print(f"  REF: {reference}")
        print(f"  HYP: {hypothesis}")
        print(f"  WER: {record['metrics']['WER']:.2f}% "
              f"(S={errors['substitutions']}, D={errors['deletions']}, I={errors['insertions']})")
        print(f"  Время: {latency:.2f} сек\n")

    # ---------------------------------------------------------------------------
    # 6. Агрегированный отчет по категориям
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("АГРЕГИРОВАННЫЙ ОТЧЕТ")
    print("=" * 60)

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        subset = [r for r in results if r["category"] == cat]
        avg_wer = sum(r["metrics"]["WER"] for r in subset) / len(subset)
        total_s = sum(r["metrics"]["substitutions"] for r in subset)
        total_d = sum(r["metrics"]["deletions"] for r in subset)
        total_i = sum(r["metrics"]["insertions"] for r in subset)
        print(f"\n{cat.upper()} ({len(subset)} файлов):")
        print(f"  Средний WER: {avg_wer:.2f}%")
        print(f"  Ошибки: Замены={total_s}, Пропуски={total_d}, Вставки={total_i}")

    # Общий WER
    total_wer = sum(r["metrics"]["WER"] for r in results) / len(results)
    print(f"\n{'=' * 60}")
    print(f"ОБЩИЙ WER: {total_wer:.2f}%")
    print(f"{'=' * 60}")

    # Сохранение полного отчета
    out = Path("asr_results.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nПолный отчет сохранен: {out.resolve()}")


if __name__ == "__main__":
    main()