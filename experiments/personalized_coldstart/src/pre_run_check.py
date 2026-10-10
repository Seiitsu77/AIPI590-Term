"""Checks run immediately before the final test. Reads no cold label.

Everything the final test depends on is compared with what was frozen: the commits, the
manifest, every input and artifact digest, the library versions, the time-cutoff record
and the boundary checks. Any mismatch makes this script exit non-zero, and the final
test is then not run.

It is kept outside the experiment folder while it runs, so that the folder stays exactly
at the frozen commit; a copy is committed afterwards together with its output.

Usage (from the experiment folder):
    python <path>/pre_run_check.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from common import ALT_DATA_INPUTS, ARTIFACTS, MAL_CATALOG, MAL_CATALOG_SHA256, MANIFEST, RATINGS, RATINGS_SHA256, SERVING_CATALOG, SERVING_CATALOG_SHA256, sha256_file  # noqa: E402
from data_access import FINAL_TEST_LOCK, MODEL_SELECTION, final_test_permitted  # noqa: E402

FROZEN_COMMITS = {"protocol": "37defa8", "training_code": "836016b", "model_selection": "45e2905"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False).stdout.strip()


def main() -> None:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    manifest = json.loads((MANIFEST / "split_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(MODEL_SELECTION.read_text(encoding="utf-8"))
    features = json.loads((ARTIFACTS / "feature_checks.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ git
    head = git("rev-parse", "HEAD")
    record("head_is_the_model_selection_commit", head.startswith(FROZEN_COMMITS["model_selection"]), head)
    record("frozen_commits_are_ancestors_of_head", all(subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", c, "HEAD"]).returncode == 0 for c in FROZEN_COMMITS.values()), FROZEN_COMMITS)
    record("experiment_folder_has_no_uncommitted_change", git("status", "--porcelain", "--", ".") == "", git("status", "--porcelain", "--", "."))
    changed = git("diff", "--name-only", FROZEN_COMMITS["protocol"], "HEAD", "--", ".").splitlines()
    record("only_gitattributes_and_model_selection_changed_since_the_freeze", sorted(Path(p).name for p in changed) == [".gitattributes", "MODEL_SELECTION.json"], changed)
    record("models_were_trained_from_the_training_code_commit", selection["code"]["commit"].startswith(FROZEN_COMMITS["training_code"]) and selection["code"]["experiment_folder_clean"], selection["code"]["commit"])

    # ------------------------------------------------------------- manifest
    record("manifest_is_frozen", manifest["status"] == "frozen", manifest["status"])
    record("manifest_leakage_checks_passed", manifest["leakage_checks"]["all_passed"], {k: v for k, v in manifest["leakage_checks"].items() if k != "all_passed"})
    manifest_sha = sha256_file(MANIFEST / "split_manifest.json")
    record("manifest_digest_equals_the_one_the_models_were_trained_with", manifest_sha == selection["inputs"]["manifest_sha256"], manifest_sha)
    verify = subprocess.run([sys.executable, str(ROOT / "src" / "make_split_manifest.py"), "--verify"], capture_output=True, text=True, cwd=ROOT)
    record("manifest_rebuilt_from_the_rule_matches", verify.returncode == 0 and '"sections_differing": []' in verify.stdout, verify.stdout[-200:].strip())

    # --------------------------------------------------------------- inputs
    pinned = {"ratings": (RATINGS, RATINGS_SHA256), "mal_catalog": (MAL_CATALOG, MAL_CATALOG_SHA256), "serving_catalog": (SERVING_CATALOG, SERVING_CATALOG_SHA256)}
    record("raw_input_digests", all(sha256_file(path) == digest == manifest["inputs"][name]["sha256"] for name, (path, digest) in pinned.items()), {name: digest[:16] for name, (_, digest) in pinned.items()})
    record("alt_data_input_digests", all(sha256_file(path) == manifest["alt_data_inputs"][name]["sha256"] for name, path in ALT_DATA_INPUTS.items()), sorted(ALT_DATA_INPUTS))
    derived = {"day_zero.csv": ARTIFACTS / "day_zero.csv", "wikipedia_article_audit.csv": MANIFEST / "wikipedia_article_audit.csv", "wikipedia_article_audit.json": MANIFEST / "wikipedia_article_audit.json"}
    record("derived_file_digests", all(sha256_file(path) == manifest["derived_files"][name] for name, path in derived.items()), sorted(derived))
    record("item_roles_and_user_list_digests", all((MANIFEST / name).exists() and sha256_file(MANIFEST / name) == digest for name, digest in manifest["files"].items()), sorted(manifest["files"]))

    # ------------------------------------------------------------ artifacts
    artifact_names = [name for name in selection["inputs"] if name != "manifest_sha256"]
    record("model_and_feature_artifact_digests", all(sha256_file(ARTIFACTS / name) == selection["inputs"][name] for name in artifact_names), {name: selection["inputs"][name][:16] for name in artifact_names})
    als = json.loads(str(np.load(ARTIFACTS / "als_experiment.npz")["metadata_json"]))
    record("als_is_the_experimental_model_trained_on_warm_titles", als["artifact_role"] == "experiment" and als["latest_premiere_year"] <= 2018 and als["manifest_sha256"] == manifest_sha, {"role": als["artifact_role"], "latest_premiere_year": als["latest_premiere_year"], "positives": als["positives"]})
    record("als_digest_in_feature_checks_matches", features["als_sha256"] == selection["inputs"]["als_experiment.npz"], features["als_sha256"][:16])
    record("feature_digests_in_feature_checks_match", all(features["artifacts"][name] == selection["inputs"][name] for name in features["artifacts"]), sorted(features["artifacts"]))

    # ------------------------------------------------------------ isolation
    time = features["time_discipline"]
    record("no_attention_data_later_than_the_cutoff", time["latest_anilist_japan_day_read"] <= time["cutoff_day"] == -2 and time["latest_wikipedia_utc_day_read"] <= -2, time)
    record("content_mapper_trained_on_warm_titles_only", features["content_mapper"]["trained_on_roles"] == ["warm"], features["content_mapper"]["training_titles"])
    record("user_vectors_built_from_warm_positives_only", features["user_vectors"]["built_from"] == "positives on warm titles only", features["user_vectors"])
    items = pd.read_parquet(ARTIFACTS / "item_features.parquet")
    roles = pd.read_csv(MANIFEST / "item_roles.csv").set_index("anime_id")["role"]
    record("feature_rows_carry_the_manifest_roles", bool((roles.loc[items["anime_id"]].to_numpy() == items["role"].to_numpy()).all()), items["role"].value_counts().to_dict())
    record("frozen_models_use_only_registered_features", {name: len(m["features"]) for name, m in selection["frozen_models"].items()} == {"M0b": 7, "M1": 6, "M2": 13}, {name: m["C"] for name, m in selection["frozen_models"].items()})
    boundary = subprocess.run([sys.executable, str(ROOT / "src" / "check_boundaries.py")], capture_output=True, text=True, cwd=ROOT)
    record("boundary_checks", boundary.returncode == 0, [line[:60] for line in boundary.stdout.splitlines()])

    # ---------------------------------------------------- state of the test
    permitted, reason = final_test_permitted()
    record("final_test_gate_open_and_not_yet_run", permitted and not FINAL_TEST_LOCK.exists() and not (ROOT / "results").exists() and not (ARTIFACTS / "FINAL_TEST_STARTED.json").exists(), reason)

    # ---------------------------------------------------------- environment
    now = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    record("library_versions_equal_those_used_for_training", now == selection["environment"], now)

    passed = all(check["passed"] for check in checks.values())
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "head": head, "all_passed": passed, "checks": checks}
    (ARTIFACTS / "pre_run_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, check in checks.items():
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {name}")
    print("ALL PASSED" if passed else "STOP: at least one check failed")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
