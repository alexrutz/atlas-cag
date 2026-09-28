"""Measure answer quality of a running Atlas instance on your own questions.

    uv run python scripts/benchmark.py cases.json [--url http://127.0.0.1:8000] [--runs 3]

cases.json is a list of cases; documents are referenced by their names in the library and each
"expect" entry is a regular expression that must match the final answer:

    [{"question": "What is the stipend and what was Q3 revenue?",
      "documents": ["hr_policy.md", "q3_report.txt"],
      "expect": ["450", "48[.,]2"]}]

Run it once per configuration (model, ATLAS_* settings, prompts) and compare the scores.
Answers are sampled, so use several runs per case.
"""

import argparse
import json
import os
import re
import sys
import time

import httpx


def ask(client: httpx.Client, question: str, doc_ids: list[str]) -> tuple[str, dict]:
    answer, stats = "", {}
    with client.stream("POST", "/api/query", json={"question": question, "document_ids": doc_ids}) as r:
        if r.status_code != 200:
            return f"HTTP {r.status_code}: {r.read().decode(errors='replace')}", {}
        for line in r.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "done":
                    answer, stats = event["answer"], event["stats"]
                elif event["type"] == "error":
                    answer = "ERROR: " + event["message"]
    return answer, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--api-key", default=os.environ.get("ATLAS_API_KEY"))
    ap.add_argument("--verbose", action="store_true", help="print every answer")
    args = ap.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=900) as client:
        library = {d["name"]: d for d in client.get("/api/documents").raise_for_status().json()}
        missing = {n for c in cases for n in c["documents"] if n not in library or not library[n]["queryable"]}
        if missing:
            print(f"not in the library or not ready: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

        hits = total = 0
        started = time.time()
        for case in cases:
            ids = [library[n]["id"] for n in case["documents"]]
            case_hits = 0
            for run in range(args.runs):
                answer, stats = ask(client, case["question"], ids)
                found = [bool(re.search(p, answer, re.I)) for p in case["expect"]]
                case_hits += sum(found)
                if args.verbose:
                    print(f"--- run {run + 1} ({stats.get('total_ms', 0) / 1000:.1f}s) {found}\n{answer}\n")
            n = args.runs * len(case["expect"])
            hits += case_hits
            total += n
            print(f"{case_hits:3}/{n:<3} {case['question']}")
        print(f"\n{hits}/{total} expected facts found ({100 * hits / max(total, 1):.0f}%) in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
