"""Reusable scan engine shared by the CLI and Streamlit UI."""
from scanner.attack_loader import load_attack_suite
from scanner.llm_judge import judge_response
from scanner.report_writer import save_report
from scanner.target_client import send_prompt


def run_scan(
    suite_path=None,
    limit=0,
    target_config=None,
    responder=None,
    save=True,
    canary_override=None,
    on_case_complete=None,
):
    suite = load_attack_suite(suite_path)
    attacks = suite["attacks"][:limit] if limit and limit > 0 else suite["attacks"]
    results = []
    for index, attack in enumerate(attacks, start=1):
        try:
            response = responder(attack["prompt"]) if responder else send_prompt(attack["prompt"], target_config)
        except Exception as error:
            results.append({
                "attack_id": attack["id"], "category": attack["category"],
                "language": attack.get("language", "en"), "prompt": attack["prompt"],
                "success_criteria": attack["success_criteria"], "response": "",
                "verdict": "INCONCLUSIVE", "judge_method": "not_run",
                "reason": f"Target request failed: {error}", "confidence": 0.0,
                "error": str(error),
            })
            if on_case_complete:
                on_case_complete(index, len(attacks), attack, results[-1])
            continue
        canary = canary_override or suite["canary"]
        if canary.casefold() in response.casefold():
            verdict, reason, confidence, judge_method = "SUCCESS", "The canary appeared in the target response.", 1.0, "canary_rule"
        elif canary.casefold() in attack["success_criteria"].replace(suite["canary"], canary).casefold():
            verdict, reason, confidence, judge_method = "FAILURE", "The required canary was absent from the target response.", 1.0, "canary_rule"
        else:
            try:
                judgment = judge_response(attack, response)
                verdict = judgment["verdict"]
                reason = judgment["reason"]
                confidence = max(0.0, min(1.0, float(judgment["confidence"])))
                judge_method = "llm"
            except Exception as error:
                verdict, reason, confidence, judge_method = "INCONCLUSIVE", f"Judge request failed: {error}", 0.0, "judge_error"
        results.append({
            "attack_id": attack["id"], "category": attack["category"],
            "language": attack.get("language", "en"), "prompt": attack["prompt"],
            "success_criteria": attack["success_criteria"], "response": response,
            "verdict": verdict, "judge_method": judge_method,
            "reason": reason, "confidence": confidence,
        })
        if on_case_complete:
            on_case_complete(index, len(attacks), attack, results[-1])
    scored = sum(item["verdict"] in {"SUCCESS", "FAILURE"} for item in results)
    successes = sum(item["verdict"] == "SUCCESS" for item in results)
    report_path = save_report(results, successes, scored) if save else None
    return {
        "results": results,
        "summary": {
            "total_cases": len(results), "scored_attacks": scored,
            "successful_attacks": successes, "failed_attacks": scored - successes,
            "inconclusive_cases": len(results) - scored,
            "attack_success_rate": successes / scored if scored else None,
        },
        "report_path": str(report_path) if report_path else None,
    }
