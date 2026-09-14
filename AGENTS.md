# AGENTS.md

There is **no root cargo/npm/pytest suite** — treat each device folder on its own. The clone-root `pyproject.toml` is only the **uv** pin for aiperf + chart libs (`uv sync`).

## Git authorship

Never add `Co-authored-by: Cursor` (or any Cursor/agent co-author trailer) to commits. Commits are authored as the human repo owner only. Do not present the agent as an author or co-author in commit messages, PRs, or repo metadata.

## The one easy mistake

`benchmark-tablets/pipette-clients/` is a **vendored, separately-cloned copy** of an external Liquid AI Rust workspace (it has its own `.git`, and `git ls-files benchmark-tablets/` returns nothing). It is **not** part of this repo. Do not edit it as smolperfbenchmark source, and do not assume its `Cargo.toml` belongs to the root project. The real repo content is the 4 benchmark folders listed in the root `README.md` (Jetson, Mac Mini M4, Raspberry Pi 5, Android/mobile). Note that `benchmark-tablets/`, `AGENTS.md`, and `session-ses_fae9.md` are currently **untracked** in git.

## Leaderboard site (private)

- **Edit** `leaderboard-site/` (gitignored clone of private `smolperfbenchmark-leaderboard`). Push `main` → Vercel production (`https://smolperfbenchmark.vercel.app/`; legacy `https://smolbenchmark.vercel.app/` dual-serves).
- **Ignore** the separate public `smolbenchmark` Pages stub for site work — it only redirects github.io → Vercel.
- Do not develop the site from the Pages stub; normal GitHub users cannot change the private source.

## How to verify work

There is no test suite. Results are **generated, never hand-authored** — and there are TWO distinct generation paths, don't conflate them:

1. **The `report.md` inside an artifact dir is written by the `.sh` script itself** at the end of a run, via an embedded Python heredoc (written to `/tmp/blog_report*.py` or `/tmp/moe_report.py`). Exceptions: `benchmark_all_bonsai.sh` (Jetson bonsai) generates **no report**; Android copies its folder's `generate_report.py` into the artifact dir and runs it.
2. **Charts and `RESULTS.md` come from the folder's `generate_*.py`**, run manually:
   - `generate_combined_charts.py` — takes **no args**; it has a hardcoded `RUNS` dict you must edit to point at the new artifact dir. Writes `artifacts/charts/*.png`. Needs pandas/seaborn/matplotlib.
   - `generate_report.py` (Jetson, mobile) — takes the artifact `base_dir` as an arg; writes `RESULTS.md` (Jetson) / `report.md` (mobile; mobile's also takes thermal/timing/ctx/framework/device positional args).
   - Mac MoE extras: `generate_appendix_i2_charts.py`, `generate_hw_utilization_charts.py` (no args, hardcoded dirs). Mac non-reasoning and Pi folders have **no generators**.

The `.sh` scripts are not unit-testable — they need real hardware, `sudo`, and multi-hour runs. Use `--dry-run` to validate script CLI/filtering without running.

## Toolchain (venv & CLIs — paths differ per device!)

- **Mac Mini + Android** use **uv** at the clone root (`uv sync` → `<repo>/.venv`). Scripts search `$SMOL_VENV`, then `.venv`, then `venv`, then the old `~/Desktop/smolbenchmark/venv` fallback. Host setup was tested on **MacBook Air M1 (2020)** and **Mac Mini M4 (2025), 16 GB**. Mac Mini MLX also needs `uv sync --extra mac`.
- **Jetson + Pi** run `uv sync` in the clone on the board (same `pyproject.toml`). Scripts walk up to `pyproject.toml` and activate `.venv`, then fall back to `$HOME/venv` / `$HOME/aiperf-env`.
- `aiperf` is the load generator. The **PyPI `aiperf` is a yanked placeholder**. The pin is in `pyproject.toml` / `uv.lock`: **0.11.0** at git `44addf0c545ff4a865c177881ca9814484ec97b4` (`aiperf --version` → `0.11.0`). Requires Python ≥ 3.11; Homebrew Python is broken on macOS (libexpat), so `uv` must provide the interpreter.
- HF CLI is now `hf`, not `huggingface-cli`. `hf auth login` is required for gated models (Gemma-3, Llama-3.2).

## Git-ignored / non-committed

`**/artifacts/`, `*.gguf`, `*.bin`, `*.safetensors`, `models/`, `gguf-models/`, `venv/`, `.venv/` are gitignored. `uv.lock` **is** committed. Reports reference `artifacts/charts/*.png` that will **not exist** in a fresh clone — they are regenerated, not checked in.

## Benchmark script conventions (shared across folders)

- Flags: `--reqs` (default 20), `--only <substring>` (case-insensitive), `--resume <dir>`, `--dry-run`, `--skip-smoke`. Plus: `--backend` (`llamacpp|ollama|both` on Jetson/Pi; `llamacpp|mlxlm` on Mac; `cpu|vulkan|both` on Android), `--power-mode 0..3` (Jetson: 15W/25W/MAXN/7W), and folder-specific ones (`--skip-download`, `--no-lock-clocks` bonsai; `--skip-build` Jetson MoE; `--no-power`, `--stream` Mac; `--only-combo` Pi; `--prompt-lengths`, `--gen-lengths`, `--host-port` Android).
- Per-combo artifacts live at `<base>/<backend>/<model>/gen<G>/ctx<P>/profile_export_aiperf.json`. `--resume <dir>` skips combos that already have that file (Jetson v1 `bench-non-reasoning.sh` is stricter — reruns timed-out/errored cells, requires ≥18/20 valid requests).
- **Failure handling**: scripts are `set -euo pipefail`; a single failed aiperf combo is **logged and skipped** (Jetson/Pi/Android), while the Mac script aborts on aiperf failure. `--resume` is the recovery/iteration path — pass it the same artifact dir and the same hardware-mode flags.
- **All scripts auto-relaunch into `tmux`** if not already inside one (Jetson, Mac, Pi, Android) — attach with `tmux attach -t <session-name>`; don't wrap them in tmux yourself.
- **bash 3.2 caveat**: macOS default `/bin/bash` is 3.2 (no associative arrays). The Android script uses `#!/opt/homebrew/bin/bash` — invoke it with `/opt/homebrew/bin/bash`, not `bash`. The **Mac scripts also use `declare -A`**, so run them with bash 4+ too.
- Power telemetry needs privileges: `sudo powermetrics` (macOS, skip with `--no-power`); `sudo tegrastats` + `nvpmodel` + `jetson_clocks` (Jetson). Pi scripts **only check** throttling via `vcgencmd get_throttled` / `measure_temp` — the `force_turbo` clock-lock is a manual `config.txt` step documented in the README, not done by the script.
- Backends in play: llama.cpp (everywhere), Ollama (Jetson non-reasoning + bonsai, Pi), MLX-LM (Mac only), llama.cpp **RPC cluster** (Jetson multi-node MoE + its single-node copy). MoE folders: Jetson MoE = llama.cpp CUDA only; Mac MoE = llama.cpp Metal + MLX-LM.
- **Known stale docs**: Jetson `non-reasoning-models/README.md` references a non-existent `generate_appendix_ab_charts.py` (real generators: `generate_combined_charts.py` + `generate_report.py`). `benchmark-raspberrypi5/BLOG.md` is a misplaced copy of the Jetson blog — don't use it as Pi documentation.

## tok/J convention — do not cross-compare

The headline metric is `output tok/J` (tokens per joule), but it is **computed differently by hardware**, and each README spells out its own formula:
- Jetson (non-reasoning v1/v2, MoE, MoE-RPC) + Pi: `tps.avg / avg_power_W` — output-token throughput avg over whole-combo average power.
- Bonsai + Mac (non-reasoning & MoE): `OSL_p50 / (decode_power_W × p50_decode_s)` — decode-phase energy only.
- Mobile: the current `generate_report.py` computes **no tok/J** (only TTFT / ITL / tok/s + thermal).

Never compare tok/J across reports using different formulas. Read the per-folder README's Metrics section before quoting numbers.

## Per-device prerequisite sources

Detailed setup (cross-compile NDK + BLIS for Android, CUDA SM_87 for Jetson, Ollama v0.24.0 pin for Pi, WiFi-ADB for real battery power) lives in each benchmark folder's README, not here.
