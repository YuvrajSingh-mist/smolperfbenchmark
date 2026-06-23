# Tiny LLM Benchmark — Raspberry Pi 5

Benchmarks non-reasoning small LLMs on Raspberry Pi 5 using llama.cpp and Ollama.


## Prerequisites

### llama.cpp (required for `--backend llamacpp` or `--backend both`)

See [llama.cpp build with BLIS](#llamacpp-build-with-blis) below. The benchmark expects the binary at `~/llama.cpp/build/bin/llama-server`, or override with:
```bash
LLAMACPP_BIN=/path/to/llama-server bash benchmark_all_cpu.sh
```

### Ollama (required for `--backend ollama` or `--backend both`)

**Ollama v0.24.0 is required.** The benchmark script auto-installs this version if a different version (or none) is detected, so in most cases you don't need to install it manually. If you do want to install it yourself:

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.24.0 sudo -E sh
```

> **Why v0.24.0?** Versions 0.30.x introduced a `llama-quantize` validation regression that rejects the GGUFs for `gemma3-4b`, `lfm2.5-350m`, and `lfm2.5-1.2b` with the error `failed to validate GGUF with llama-quantize without compatibility patches`. v0.24.0 imports all 9 benchmark GGUFs successfully.

Ollama imports the same local GGUF files — no separate model download needed.

#### Benchmark-critical Ollama settings

The script starts its own `ollama serve` instance with these environment variables:

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | 1 | **Critical.** aiperf dispatches all requests near-simultaneously. Without this, Ollama processes multiple requests concurrently, collapsing measured throughput from ~24 tok/s to ~0.5 tok/s. |
| `OLLAMA_FLASH_ATTENTION` | 1 | Enables flash attention in the runner. |
| `OLLAMA_MAX_LOADED_MODELS` | 1 | Prevents a second model loading between benchmarks and stealing RAM. |

These are set inline when the script starts Ollama — no manual configuration needed.

### aiperf (load generator, required)

```bash
python3 -m venv ~/aiperf-env
source ~/aiperf-env/bin/activate
pip install aiperf
```

The script tries `~/venv` first, then `~/aiperf-env`.

### HuggingFace CLI (for model auto-download)

```bash
pip install huggingface_hub
huggingface-cli login   # required for gated models (Llama-3.2, Gemma 3)
```

Models are downloaded automatically on first run if the GGUF is missing from `~/gguf-models/`.

---

## Reproducibility: clock locking

The Pi 5 defaults to an `ondemand` CPU governor that scales between 1.5–2.4 GHz.
Without locking clocks, thermal throttling and frequency variation make benchmark results unreproducible across runs.

The Pi 5 soft-throttles at 80 °C (−100 MHz/°C) and hard-throttles at 85 °C even with `force_turbo=1`.
Check throttle history before and after a run:
```bash
vcgencmd get_throttled   # 0x0 = clean; anything else = throttling has occurred since last reboot
vcgencmd measure_temp
```

### 1. Lock the clock frequency (runtime)

Pin all 4 cores to 2400 MHz immediately without a reboot:
```bash
echo 2400000 | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo 2400000 | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Persist across reboots

**`/boot/firmware/config.txt`** — add under `[all]`:
```ini
# Lock CPU to 2400 MHz for benchmark reproducibility
arm_freq=2400
force_turbo=1
```

`force_turbo=1` pins the CPU at `arm_freq` and disables dynamic scaling. It does **not** void the warranty or set the OTP bit — that only happens if `over_voltage_*` is also set.

**`/etc/tmpfiles.d/cpu-performance-governor.conf`** — create this file:
```
w /sys/devices/system/cpu/cpufreq/policy0/scaling_governor - - - - performance
```

This uses systemd-tmpfiles (the Pi OS mechanism) to apply the governor early in boot. The Pi 5 exposes all 4 cores under a single `policy0`.

### Verify

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # performance
vcgencmd measure_clock arm                                   # ~2400000000
vcgencmd get_throttled                                       # 0x0 after clean reboot
```

---

## llama.cpp build with BLIS

BLIS (FLAME) replaces the default OpenBLAS backend and is tuned for ARM Cortex-A76.

### Install BLIS from source

```bash
git clone https://github.com/flame/blis.git
cd blis
./configure --prefix=/usr/local --enable-shared aarch64
make -j4
sudo make install
```

### Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=FLAME \
  -DGGML_NATIVE=ON
cmake --build build --config Release -j4
```

- `GGML_BLAS_VENDOR=FLAME` — selects BLIS over OpenBLAS
- `GGML_NATIVE=ON` — compiles with `-march=native`, enabling NEON/SVE for the A76 cores

---

## Thread environment variables

The benchmark script exports four variables before launching `llama-server`:

```bash
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export LLAMA_ARG_THREADS=4
```

| Variable | Value | Why |
|---|---|---|
| `BLIS_NUM_THREADS` | 1 | Prevents BLIS from spawning its own thread pool on top of llama.cpp's ggml pool |
| `OPENBLAS_NUM_THREADS` | 1 | Same for OpenBLAS (present as a transitive dependency in some builds) |
| `OMP_NUM_THREADS` | 1 | Prevents OpenMP from over-subscribing the 4 cores with additional threads |
| `LLAMA_ARG_THREADS` | 4 | Gives llama.cpp's ggml thread pool all 4 Pi 5 cores |

Without the first three, external BLAS/OMP libraries each spawn their own thread pools on top of ggml's, saturating the 4-core CPU and degrading throughput.

---


## Running the benchmark

```bash
bash benchmark-raspberrypi5/benchmark_all_cpu.sh                   # llamacpp, all models
bash benchmark-raspberrypi5/benchmark_all_cpu.sh --backend ollama  # ollama only
bash benchmark-raspberrypi5/benchmark_all_cpu.sh --backend both    # llamacpp then ollama
bash benchmark-raspberrypi5/benchmark_all_cpu.sh --only smollm2    # single model filter
bash benchmark-raspberrypi5/benchmark_all_cpu.sh --reqs 5          # fewer requests (quick test)
bash benchmark-raspberrypi5/benchmark_all_cpu.sh --resume DIR      # resume interrupted run
```

Results are written to `artifacts/blog-all-YYYYMMDD-HHMM/`.

---

## Power measurement

Power is sampled from the Pi 5 PMIC via `vcgencmd pmic_read_adc` every 500 ms. The value logged is the sum of `current × voltage` across all 12 measurable rails (dominated by `VDD_CORE`). The 5 V input rail (`EXT5V`) has no current sensor, so this is the **output-side rail sum** — it undercounts true board power by roughly the switching-regulator losses (~10–15%). All runs use the same formula, so relative tok/J comparisons between models are valid; treat absolute watt values as approximate.

### Per-combo logging

The PMIC logger runs **per aiperf combo** (one `ctx × gen` pair), writing to:

```
artifacts/.../llamacpp/<model>/gen<G>/ctx<C>/power.log
```

This means:
- Each combo owns its own power file — no global timing correlation needed.
- If a run crashes mid-sweep, completed combos keep their power data intact.
- `--resume` re-runs only the missing combos and overwrites only those combos' power files.

### Report generation fallback

The report generator tries the per-combo `power.log` first. If absent, it falls back to `tegrastats.log` in the same directory (artifact dirs from before the rename), then to the run-wide `power.log` + `model_timing.log` timing windows for pre-per-combo artifact dirs — so old and new runs can be compared without manual intervention.
