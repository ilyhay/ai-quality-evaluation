"""
Golden-set evaluation pipeline for local LLM.
Metrics: Normalized Exact Match, Token F1, ROUGE-L, deterministic checks, LLM-as-a-Judge.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# --- Windows-совместимый вывод кириллицы в консоль -------------------------
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "Ты — полезный ассистент службы поддержки. Отвечай кратко, вежливо и по делу. "
    "ПРАВИЛА:\n"
    "1. Используй ТОЛЬКО факты из предоставленного КОНТЕКСТА. Не выдумывай.\n"
    "2. Если в контексте есть прямой ответ, дай его, перечислив все важные условия (сроки, суммы).\n"
    "3. Если в контексте прямо сказано, что что-то невозможно (например, оплата биткоином), так и ответь: 'Нет, это невозможно, потому что...'.\n"
    "4. Если в контексте вообще нет информации по теме вопроса, так и скажи: 'К сожалению, у меня нет информации по этому вопросу'.\n"
    "5. НИКОГДА не повторяй слова 'ВОПРОС:', 'КОНТЕКСТ:' или 'РЕШЕНИЕ:' в своем ответе. Отвечай сплошным текстом или JSON, если просят."
)

JUDGE_SYSTEM = (
    "Ты — строгий, объективный оценщик качества ответов ассистента. "
    "Ты не снисходителен. Ты оцениваешь строго по критериям. "
    "Возвращаешь ТОЛЬКО валидный JSON без пояснений и без markdown."
)

JUDGE_PROMPT = """Оцени ответ ассистента по трём критериям по шкале 1-5.

КРИТЕРИИ:
1. relevance (релевантность) — отвечает ли ответ именно на заданный вопрос, без воды.
2. completeness (полнота) — покрыты ли все существенные части вопроса и все ключевые факты.
3. faithfulness (правдивость) — опирается ли ответ ТОЛЬКО на предоставленный контекст.
   Любая выдуманная деталь = 1 балл. Честный отказ при отсутствии данных = 5 баллов.

КОНТЕКСТ (единственный источник правды):
{context}

ВОПРОС:
{prompt}

ЭТАЛОННЫЙ ОТВЕТ (ориентир эксперта):
{expected}

ОТВЕТ АССИСТЕНТА (оценивается):
{prediction}

Верни строго такой JSON:
{{"relevance": <int>, "completeness": <int>, "faithfulness": <int>, "reason": "<1-2 предложения на русском>"}}
"""


# ---------------------------------------------------------------------------
# 1. Загрузка датасета
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Строка {line_no} в {path.name} повреждена: {e}")
    return cases


# ---------------------------------------------------------------------------
# 2. Вызов локальной модели (детерминированно!)
# ---------------------------------------------------------------------------
def ask_model(model: str, prompt: str, context: str, timeout: int = 180) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{prompt}"},
        ],
        # temperature=0 + фиксированный seed = воспроизводимость прогонов.
        # Без этого твои метрики будут «плавать» и отчёт ничего не докажет.
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 3. Детерминированные проверки (самые дешёвые и самые надёжные)
# ---------------------------------------------------------------------------
def run_checks(prediction: str, case: dict) -> dict:
    pred_lower = prediction.lower()
    missing = [s for s in case.get("must_contain", []) if s.lower() not in pred_lower]
    violated = [s for s in case.get("forbidden", []) if s.lower() in pred_lower]
    return {
        "must_contain_passed": not missing,
        "missing": missing,
        "forbidden_passed": not violated,
        "violated": violated,
    }


# ---------------------------------------------------------------------------
# 4. Статистические метрики
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    lcs = _lcs_length(p, r)
    precision, recall = lcs / len(p), lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 5. LLM-as-a-Judge
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """Judge-модели часто оборачивают JSON в ```json ... ``` — вытаскиваем ядро."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Судья не вернул JSON. Получено: {text[:200]}")
    return json.loads(match.group(0))


def llm_judge(judge_model: str, case: dict, prediction: str, timeout: int = 240) -> dict:
    prompt = JUDGE_PROMPT.format(
        context=case.get("context", ""),
        prompt=case["prompt"],
        expected=case["expected"],
        prediction=prediction,
    )
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    raw = _extract_json(r.json()["message"]["content"])
    return {
        "relevance": int(raw["relevance"]),
        "completeness": int(raw["completeness"]),
        "faithfulness": int(raw["faithfulness"]),
        "reason": raw.get("reason", ""),
    }


# ---------------------------------------------------------------------------
# 6. Основной прогон
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="golden_dataset.jsonl")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--judge-model", default="qwen2.5:7b")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_dataset(Path(args.dataset))
    if args.limit:
        cases = cases[: args.limit]

    results, failures = [], []
    print(f"Модель: {args.model} | Судья: {args.judge_model} | Кейсов: {len(cases)}\n")

    for case in cases:
        t0 = time.time()
        pred = ask_model(args.model, case["prompt"], case.get("context", ""))
        checks = run_checks(pred, case)
        judge = llm_judge(args.judge_model, case, pred)
        latency = time.time() - t0

        record = {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "prediction": pred,
            "expected": case["expected"],
            "metrics": {
                "exact_match": exact_match(pred, case["expected"]),
                "token_f1": round(token_f1(pred, case["expected"]), 4),
                "rouge_l": round(rouge_l(pred, case["expected"]), 4),
            },
            "checks": checks,
            "judge": judge,
            "latency_s": round(latency, 2),
        }
        # Вердикт: жёсткие проверки И минимальный порог судьи.
        record["pass"] = (
            checks["must_contain_passed"]
            and checks["forbidden_passed"]
            and judge["relevance"] >= 4
            and judge["completeness"] >= 4
            and judge["faithfulness"] >= 4
        )
        results.append(record)

        flag = "PASS" if record["pass"] else "FAIL"
        print(f"[{flag}] {case['id']} ({case['category']}) "
              f"F1={record['metrics']['token_f1']:.2f} "
              f"R-L={record['metrics']['rouge_l']:.2f} "
              f"judge={judge['relevance']}/{judge['completeness']}/{judge['faithfulness']}")
        if not record["pass"]:
            failures.append(record)
            print(f"       причина: {judge['reason']}")

    # --- Агрегированный отчёт -------------------------------------------------
    def avg(key):
        vals = [r["judge"][key] for r in results]
        return sum(vals) / len(vals) if vals else 0

    print("\n" + "=" * 60)
    print(f"PASS RATE: {sum(r['pass'] for r in results)}/{len(results)}")
    print(f"Relevance    : {avg('relevance'):.2f} / 5")
    print(f"Completeness : {avg('completeness'):.2f} / 5")
    print(f"Faithfulness : {avg('faithfulness'):.2f} / 5")
    print("=" * 60)

    print("\nРазбивка по категориям (где именно проблема):")
    for cat in sorted({r["category"] for r in results}):
        subset = [r for r in results if r["category"] == cat]
        passed = sum(r["pass"] for r in subset)
        print(f"  {cat:<12} {passed}/{len(subset)}")

    out = Path("results.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nПолный отчёт сохранён: {out.resolve()}")


if __name__ == "__main__":
    main()