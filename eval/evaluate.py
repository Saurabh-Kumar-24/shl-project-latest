"""Evaluation script for the SHL Assessment Recommender.

Parses C1-C10 conversation traces, simulates them against the API,
and measures Recall@10, schema compliance, and behavior probes.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"

EXPECTED: dict[str, list[str]] = {
    "C1": [
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
        "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
    ],
    "C2": [
        "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
        "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
        "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C3": [
        "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
        "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
        "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
        "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
    ],
    "C4": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
        "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
        "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C5": [
        "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
        "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
        "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
    ],
    "C6": [
        "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
        "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
    ],
    "C7": [
        "https://www.shl.com/products/product-catalog/view/hipaa-security/",
        "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
        "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C8": [
        "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
        "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
        "https://www.shl.com/products/product-catalog/view/ms-word-new/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C9": [
        "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
        "https://www.shl.com/products/product-catalog/view/spring-new/",
        "https://www.shl.com/products/product-catalog/view/sql-new/",
        "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
        "https://www.shl.com/products/product-catalog/view/docker-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C10": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
    ],
}


@dataclass
class TraceResult:
    trace_id: str
    recall: float = 0.0
    schema_ok: bool = True
    errors: list[str] = field(default_factory=list)


def parse_trace(path: Path) -> list[dict[str, str]]:
    """Extract user messages from a conversation trace markdown file."""
    text = path.read_text(encoding="utf-8")
    user_msgs: list[dict[str, str]] = []
    pattern = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\n|\n\*\*|\Z)", re.DOTALL)
    for match in pattern.finditer(text):
        content = match.group(1).strip()
        user_msgs.append({"role": "user", "content": content})
    return user_msgs


def simulate_conversation(trace_id: str, user_msgs: list[dict[str, str]]) -> TraceResult:
    """Run a conversation against the API, building up messages turn by turn."""
    result = TraceResult(trace_id=trace_id)
    messages: list[dict[str, str]] = []
    last_response: dict | None = None
    all_rec_urls: set[str] = set()

    for msg in user_msgs:
        if last_response and last_response.get("reply"):
            messages.append({"role": "assistant", "content": last_response["reply"]})
        messages.append(msg)

        try:
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"messages": messages},
                timeout=30,
            )
            if resp.status_code != 200:
                result.errors.append(f"HTTP {resp.status_code}: {resp.text[:200]}")
                result.schema_ok = False
                continue
            last_response = resp.json()
        except Exception as e:
            result.errors.append(f"Request failed: {e}")
            result.schema_ok = False
            continue

        if not _validate_schema(last_response):
            result.schema_ok = False
            result.errors.append("Schema violation in response")

        for r in last_response.get("recommendations", []):
            all_rec_urls.add(r["url"])

    expected_urls = set(EXPECTED.get(trace_id, []))
    if expected_urls:
        hits = len(all_rec_urls & expected_urls)
        result.recall = hits / len(expected_urls)

    return result


def _validate_schema(resp: dict) -> bool:
    if "reply" not in resp or not isinstance(resp["reply"], str):
        return False
    if "recommendations" not in resp or not isinstance(resp["recommendations"], list):
        return False
    if "end_of_conversation" not in resp or not isinstance(resp["end_of_conversation"], bool):
        return False
    for r in resp["recommendations"]:
        if not all(k in r for k in ("name", "url", "test_type")):
            return False
    return True


def main() -> None:
    traces_dir = Path(__file__).resolve().parent.parent / "GenAI_SampleConversations"
    if not traces_dir.exists():
        print(f"Traces directory not found: {traces_dir}")
        sys.exit(1)

    results: list[TraceResult] = []
    for i in range(1, 11):
        trace_id = f"C{i}"
        path = traces_dir / f"{trace_id}.md"
        if not path.exists():
            print(f"Skipping {trace_id}: file not found")
            continue

        print(f"Simulating {trace_id}...")
        user_msgs = parse_trace(path)
        result = simulate_conversation(trace_id, user_msgs)
        results.append(result)
        print(
            f"  Recall@10: {result.recall:.2f} | "
            f"Schema OK: {result.schema_ok} | "
            f"Errors: {len(result.errors)}"
        )

    avg_recall = sum(r.recall for r in results) / len(results) if results else 0
    schema_pass = sum(1 for r in results if r.schema_ok)
    print(f"\n--- Summary ---")
    print(f"Average Recall@10: {avg_recall:.2f}")
    print(f"Schema compliance: {schema_pass}/{len(results)}")

    for r in results:
        if r.errors:
            print(f"\n{r.trace_id} errors:")
            for e in r.errors:
                print(f"  - {e}")


if __name__ == "__main__":
    main()
