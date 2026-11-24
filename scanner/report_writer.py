import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def safe_report_text(value):
    text = str(value or "")
    return text.replace("```", "``\u200b`")


def save_report(results, success_count, scored_count):
    project_folder = Path(__file__).resolve().parent.parent
    reports_folder = project_folder / "reports"
    reports_folder.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = reports_folder / f"scan-{timestamp}.json"
    markdown_file = reports_folder / f"scan-{timestamp}.md"

    total_cases = len(results)
    inconclusive_count = total_cases - scored_count

    categories = defaultdict(list)
    for result in results:
        category = result.get("category", "uncategorized")
        categories[category].append(result)

    category_summary = {}

    for category, category_results in categories.items():
        scored_results = [
            result
            for result in category_results
            if result.get("verdict") in {"SUCCESS", "FAILURE"}
        ]

        category_successes = sum(
            result["verdict"] == "SUCCESS"
            for result in scored_results
        )

        category_summary[category] = {
            "total_cases": len(category_results),
            "scored_cases": len(scored_results),
            "successes": category_successes,
            "failures": len(scored_results) - category_successes,
            "inconclusive": len(category_results) - len(scored_results),
            "attack_success_rate": (
                category_successes / len(scored_results)
                if scored_results
                else None
            ),
        }

    summary = {
        "total_cases": total_cases,
        "scored_attacks": scored_count,
        "successful_attacks": success_count,
        "failed_attacks": scored_count - success_count,
        "inconclusive_cases": inconclusive_count,
        "attack_success_rate": (
            success_count / scored_count
            if scored_count
            else None
        ),
        "per_category": category_summary,
    }

    report_data = {
        "created_at": datetime.now().astimezone().isoformat(),
        "summary": summary,
        "results": results,
    }

    report_file.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    overall_rate = summary["attack_success_rate"]
    overall_text = (
        f"{overall_rate:.1%}"
        if overall_rate is not None
        else "N/A"
    )

    lines = [
        "# Prompt Injection Scan Report",
        "",
        f"- Created: {report_data['created_at']}",
        f"- Total cases: {total_cases}",
        f"- Scored attacks: {scored_count}",
        f"- Successes: {success_count}",
        f"- Failures: {scored_count - success_count}",
        f"- Inconclusive: {inconclusive_count}",
        f"- Attack success rate: {overall_text}",
        "",
        "## Results by category",
        "",
        "| Category | Successes | Failures | Inconclusive | ASR |",
        "|---|---:|---:|---:|---:|",
    ]

    for category, values in category_summary.items():
        rate = values["attack_success_rate"]
        rate_text = f"{rate:.1%}" if rate is not None else "N/A"
        lines.append(
            f"| {category} | {values['successes']} | "
            f"{values['failures']} | {values['inconclusive']} | "
            f"{rate_text} |"
        )

    lines.extend(["", "## Individual cases", ""])

    for result in results:
        lines.extend([
            f"### {safe_report_text(result.get('attack_id', 'unknown'))}",
            "",
            f"- Category: {safe_report_text(result.get('category'))}",
            f"- Verdict: **{safe_report_text(result.get('verdict'))}**",
            f"- Judge: {safe_report_text(result.get('judge_method'))}",
            f"- Reason: {safe_report_text(result.get('reason'))}",
            "",
            "**Prompt**",
            "",
            "```text",
            safe_report_text(result.get("prompt")),
            "```",
            "",
            "**Target response**",
            "",
            "```text",
            safe_report_text(result.get("response")),
            "```",
            "",
        ])

    markdown_file.write_text("\n".join(lines), encoding="utf-8")

    return report_file