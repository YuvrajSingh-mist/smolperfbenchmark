#!/usr/bin/env python3
"""Keep Hugging Face dataset cards aligned with the repo's license policy.

smolperfbenchmark is dual licensed:

  * code / harness (scripts, generators, device folders) -> Apache-2.0
  * results, artifacts, charts, leaderboard data         -> CC BY 4.0

Every HF dataset repo published from this harness contains results only, so its
card must declare CC BY 4.0. Cards are uploaded from outside the repo, which is
how they previously drifted to ``apache-2.0``; run this script as a pre-flight
step before uploading, or point it at already published cards.

Usage:

    # Verify local cards (exits 1 if any are wrong); CI / pre-upload friendly
    python tools/hf_cards.py check artifacts/llamacpp/*/README.md

    # Fix local cards in place
    python tools/hf_cards.py apply artifacts/llamacpp/*/README.md

    # Fetch published cards, fix them, and push them back
    HF_TOKEN=... python tools/hf_cards.py sync-hf YuvrajSingh9886/jetson-non-reasoning-benchmark-25w

Requires ``huggingface_hub`` for the ``sync-hf`` subcommand only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LICENSE_ID = "cc-by-4.0"
LICENSE_LINE_RE = re.compile(r"^license: ?.*$", flags=re.M)
CITATION = """@misc{singh2026smolperfbenchmark,
      title={smolperfbenchmark: On-Device LLM Leaderboard},
      author={Yuvraj Singh},
      year={2026},
      howpublished={\\url{https://github.com/YuvrajSingh-mist/smolperfbenchmark}},
}"""

LICENSE_SECTION = f"""
## License & citation

- **Benchmark results and artifacts** (aiperf exports, server logs, `tegrastats` logs, generated reports): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse and adaptation allowed, including commercially, provided you credit **Yuvraj Singh**, link the license, and indicate changes.
- **Harness code** that produced these results: [Apache-2.0](https://github.com/YuvrajSingh-mist/smolperfbenchmark/blob/master/LICENSE).

If you use these results, please cite:

```bibtex
{CITATION}
```
"""


def problems(text: str) -> list[str]:
    """Return the list of license problems in a dataset card."""
    issues = []
    match = LICENSE_LINE_RE.search(text)
    if not match:
        issues.append("front matter has no 'license:' field")
    elif match.group(0) != f"license: {LICENSE_ID}":
        issues.append(f"license is '{match.group(0).split(':', 1)[1].strip()}', expected '{LICENSE_ID}'")
    if "## License & citation" not in text:
        issues.append("missing '## License & citation' section")
    return issues


def fix(text: str) -> str:
    """Set the CC BY 4.0 front matter and append the license section if absent."""
    if LICENSE_LINE_RE.search(text):
        text = LICENSE_LINE_RE.sub(f"license: {LICENSE_ID}", text, count=1)
    else:
        # No front matter block: prepend a minimal one.
        text = f"---\nlicense: {LICENSE_ID}\n---\n\n{text}"
    if "## License & citation" not in text:
        text = text.rstrip("\n") + "\n" + LICENSE_SECTION
    return text


def _run_check(paths: list[Path]) -> int:
    failed = False
    for path in paths:
        issues = problems(path.read_text(encoding="utf-8"))
        if issues:
            failed = True
            print(f"FAIL {path}")
            for issue in issues:
                print(f"       - {issue}")
        else:
            print(f"ok   {path}")
    return 1 if failed else 0


def _run_apply(paths: list[Path]) -> int:
    for path in paths:
        before = path.read_text(encoding="utf-8")
        after = fix(before)
        if after == before:
            print(f"unchanged {path}")
            continue
        path.write_text(after, encoding="utf-8")
        print(f"fixed     {path}")
    return 0


def _run_sync_hf(repo_ids: list[str]) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return print("huggingface_hub is required for sync-hf: pip install 'huggingface_hub[cli]'") or 1

    api = HfApi()
    rc = 0
    for repo_id in repo_ids:
        try:
            path = api.hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL {repo_id}: could not fetch README.md ({type(exc).__name__})")
            rc = 1
            continue
        text = Path(path).read_text(encoding="utf-8")
        fixed = fix(text)
        if fixed == text:
            print(f"ok   {repo_id} (already {LICENSE_ID})")
            continue
        api.upload_file(
            path_or_fileobj=str(_staged(repo_id, fixed)),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Set dataset card license to {LICENSE_ID} and add citation.",
        )
        info = api.dataset_info(repo_id)
        print(f"fixed {repo_id} -> license={(info.cardData or {}).get('license')}")
    return rc


def _staged(repo_id: str, text: str) -> Path:
    tmp = Path("/tmp") / f"hfcard-{repo_id.replace('/', '__')}.md"
    tmp.write_text(text, encoding="utf-8")
    return tmp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="verify local cards declare CC BY 4.0")
    check.add_argument("paths", nargs="+", type=Path)

    apply_ = sub.add_parser("apply", help="fix local cards in place")
    apply_.add_argument("paths", nargs="+", type=Path)

    sync = sub.add_parser("sync-hf", help="fix published HF dataset cards (needs HF_TOKEN)")
    sync.add_argument("repos", nargs="+")

    args = parser.parse_args()
    if args.cmd == "check":
        return _run_check(args.paths)
    if args.cmd == "apply":
        return _run_apply(args.paths)
    return _run_sync_hf(args.repos)


if __name__ == "__main__":
    sys.exit(main())
