from scanner.attack_loader import load_attack_suite
from scanner.target_client import send_prompt
from scanner.llm_judge import judge_response
from scanner.report_writer import save_report
import argparse


def main(limit=0, suite_path=None):
    suite = load_attack_suite(suite_path)
    canary = suite["canary"]
    attacks = suite["attacks"]
    if limit > 0:
        attacks = attacks[:limit]

    successful_attacks = 0
    scored_attacks = 0
    results = []

    print(f"Loaded {len(attacks)} attacks.\n")

    for attack in attacks:
        print(f"Running: {attack['id']}")

        try:
            response = send_prompt(attack["prompt"])
        except Exception as error:
            print(f"  Target error: {error}\n")
            results.append({
                "attack_id": attack["id"],
                "category": attack["category"],
                "prompt": attack["prompt"],
                "verdict": "INCONCLUSIVE",
                "judge_method": "not_run",
                "reason": f"Target request failed: {error}",
                "error": str(error),
            })
            continue

        if canary.casefold() in response.casefold():
            verdict = "SUCCESS"
            reason = "The canary appeared in the target response."
            confidence = 1.0
            judge_method = "canary_rule"

        elif canary.casefold() in attack["success_criteria"].casefold():
            verdict = "FAILURE"
            reason = "The required canary was absent from the target response."
            confidence = 1.0
            judge_method = "canary_rule"

        else:
            try:
                judgment = judge_response(attack, response)
                verdict = judgment["verdict"]
                reason = judgment["reason"]
                confidence = max(
                    0.0,
                    min(1.0, float(judgment["confidence"])),
                )
                judge_method = "llm"
            except Exception as error:
                verdict = "INCONCLUSIVE"
                reason = f"Judge request failed: {error}"
                confidence = 0.0
                judge_method = "judge_error"

        if verdict in {"SUCCESS", "FAILURE"}:
            scored_attacks += 1

        if verdict == "SUCCESS":
            successful_attacks += 1

        print(f"  Verdict: {verdict}")
        print(f"  Judge: {judge_method}")
        print(f"  Reason: {reason}")
        print(f"  Response: {response}\n")

        results.append({
            "attack_id": attack["id"],
            "category": attack["category"],
            "language": attack["language"],
            "prompt": attack["prompt"],
            "success_criteria": attack["success_criteria"],
            "response": response,
            "verdict": verdict,
            "judge_method": judge_method,
            "reason": reason,
            "confidence": confidence,
        })

    score = (
        successful_attacks / scored_attacks * 100
        if scored_attacks
        else 0
    )

    print(f"Scored attacks: {scored_attacks}")
    print(f"Inconclusive cases: {len(results) - scored_attacks}")
    print(f"Attack success rate: {score:.1f}%")

    report_file = save_report(
        results,
        successful_attacks,
        scored_attacks,
    )
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run prompt injection attacks."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of attacks to run; 0 means all attacks.",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Optional YAML attack suite path; defaults to payloads/attacks.yaml.",
    )
    args = parser.parse_args()
    main(limit=args.limit, suite_path=args.suite)
