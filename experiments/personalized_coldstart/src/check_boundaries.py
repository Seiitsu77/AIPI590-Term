"""Check that no script can cross the train / validation / test boundary.

Two kinds of check. Static: read the source and confirm that only the intended files can
reach ratings, labels and the test roles. Dynamic: ask for test labels the way a
development script would and confirm the request is refused.

Run before the freeze and again before the final test. Writes
artifacts/boundary_checks.json and exits non-zero if anything fails.

Usage:
    python src/check_boundaries.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from common import ARTIFACTS

SRC = Path(__file__).resolve().parent
ME = Path(__file__).name

# Which files may do what. Everything not listed may not.
MAY_OPEN_RATINGS = {"common.py", "data_access.py", "make_split_manifest.py"}
MAY_IMPORT_LABELS = {"train_rankers.py", "final_test.py"}
MAY_REQUEST_TEST_LABELS = {"final_test.py"}
MUST_NOT_NAME_TEST_ROLES = {"train_rankers.py", "train_als.py"}


def sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(SRC.glob("*.py")) if path.name != ME}


def code_only(text: str) -> str:
    """Source with docstrings and comments removed, so prose about a rule cannot trip the rule."""
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    code = {name: code_only(text) for name, text in sources().items()}
    checks: dict[str, dict] = {}

    def record(name: str, offenders: list[str], rule: str) -> None:
        checks[name] = {"passed": not offenders, "rule": rule, "offenders": offenders}

    record("ratings_file", [n for n, c in code.items() if re.search(r"\bRATINGS\b|rating_complete", c) and n not in MAY_OPEN_RATINGS],
           f"only {sorted(MAY_OPEN_RATINGS)} may name the ratings file")  # fmt: skip
    record("labels_import", [n for n, c in code.items() if re.search(r"from data_access import[^\n]*\blabels\b", c) and n not in MAY_IMPORT_LABELS],
           f"only {sorted(MAY_IMPORT_LABELS)} may import labels()")  # fmt: skip
    record("test_label_request", [n for n, c in code.items() if "final_test=True" in c and n not in MAY_REQUEST_TEST_LABELS],
           f"only {sorted(MAY_REQUEST_TEST_LABELS)} may pass final_test=True")  # fmt: skip
    record("test_roles_in_development", [n for n in MUST_NOT_NAME_TEST_ROLES if "cold_test" in code.get(n, "")],
           f"{sorted(MUST_NOT_NAME_TEST_ROLES)} must not name a test role")  # fmt: skip
    # The MAL catalog has an age-rating column called "Rating"; that is content metadata,
    # so the pattern looks for interaction files by name, not for the word.
    record("interaction_file_reads", [n for n, c in code.items() if re.search(r"(animelist|ratings?)\.csv", c, flags=re.IGNORECASE) and n not in MAY_OPEN_RATINGS],
           "no other file names a user-interaction CSV")  # fmt: skip

    # Dynamic: a development-style request for test labels must be refused.
    from data_access import final_test_permitted, labels

    refused = {}
    for role in ("cold_test", "cold_test_late"):
        try:
            labels(role)
            refused[role] = False
        except PermissionError:
            refused[role] = True
    checks["test_labels_refused_without_final_test_flag"] = {"passed": all(refused.values()), "rule": "labels(test role) raises PermissionError unless called by the final test", "detail": refused}
    permitted, reason = final_test_permitted()
    checks["final_test_gate"] = {"passed": True, "rule": "informational: whether the final test would currently be allowed", "permitted": permitted, "reason": reason}

    passed = all(check["passed"] for check in checks.values())
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "boundary_checks.json").write_text(json.dumps({"all_passed": passed, "checks": checks}, indent=2), encoding="utf-8")
    for name, check in checks.items():
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {name}: {check['rule']}" + (f"  -> {check.get('offenders') or check.get('detail') or check.get('reason')}" if not check["passed"] or name == "final_test_gate" else ""))
    if not passed:
        raise SystemExit("boundary checks failed")


if __name__ == "__main__":
    main()
