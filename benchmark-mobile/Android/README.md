# Android Mobile Benchmark — OnePlus 10R (Dimensity 8100-Max)

End-to-end setup and results for benchmarking small LLMs on Android via USB + ADB.
Mirrors the structure of the Jetson benchmarks — artifacts, report format, and chart scripts are compatible.

---

## Device

| Field | Value |
|---|---|
| Model | OnePlus 10R (CPH2423) |
| SoC | MediaTek Dimensity 8100-Max (mt6895) |
| Big cores | 4× Cortex-A78 @ 2850 MHz (CPU part 0xd41) |
| Little cores | 4× Cortex-A55 @ 2000 MHz (CPU part 0xd05) |
| GPU | Mali-G68 MC4 (Vulkan) |
| RAM | 7411 MB |
| Android | 15 (ColorOS) |
| CPU ISA | ARMv8.2-A + dotprod (`asimddp`) + fp16 (`asimdhp`) — **no i8mm, no SVE** |

Confirmed via `/proc/cpuinfo`:
```
Features: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp
          cpuid asimdrdm lrcpc dcpop asimddp
```

---

## Host Requirements

| Tool | Install |
|---|---|
| macOS (tested on Mac mini M-series) | — |
| Android NDK 29 | `brew install --cask android-ndk` |
| ADB | `brew install android-platform-tools` |
| uv (Python manager) | `brew install uv` |
| cmake, make | `brew install cmake` |

---

## Setup — Step by Step

### 1. Enable USB Debugging on Phone

1. Settings → About Device → tap **Version** 7 times
2. Settings → Additional Settings → Developer Options → enable **USB Debugging**
3. Plug in phone via USB-C data cable (charge-only cables won't work), unlock screen
4. Tap **Allow** on the "Allow USB debugging?" prompt

```bash
adb devices
# RS7XKZDI8HTOJNYL    device   ← must show "device", not "unauthorized"
```

If it shows `unauthorized`: unlock phone, look for the popup again. If popup doesn't appear: Developer Options → Revoke USB debugging authorizations, then reconnect.

---

### 2. Set Up NDK Toolchain

NDK 29 installs to `/opt/homebrew/Caskroom/android-ndk/29/`.

```bash
export NDK="/opt/homebrew/Caskroom/android-ndk/29/AndroidNDK14206865.app/Contents/NDK"
export TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/darwin-x86_64"
export API=29
export CC="$TOOLCHAIN/bin/aarch64-linux-android${API}-clang"
export AR="$TOOLCHAIN/bin/llvm-ar"
export RANLIB="$TOOLCHAIN/bin/llvm-ranlib"
export CFLAGS="-O3 -mcpu=cortex-a78"
```

Verify:
```bash
$CC --version
# Android (r29) clang version 21.0.0 — Target: aarch64-unknown-linux-android29
```

> **Note:** Use `API=29` not `API=28` — NDK 29 ships no `android-28` clang wrapper.

---

### 3. Build BLIS for Android ARM64

BLIS provides optimized BLAS routines linked into llama.cpp for matrix ops.

```bash
cd benchmark-mobile/
git clone https://github.com/flame/blis
cd blis

./configure \
    --disable-shared \
    --enable-static \
    --enable-cblas \
    -t pthreads \
    --prefix=$HOME/blis-android \
    cortexa57

make -j$(sysctl -n hw.ncpu)
make install
# Output: ~/blis-android/lib/libblis.a
```

**Lessons learned (what NOT to do):**

| Mistake | Fix |
|---|---|
| Pass `--host=aarch64-linux-android` | BLIS is not autoconf — no `--host` flag. Export `CC`/`AR`/`RANLIB` instead |
| Pass `CC="$CC"` inline to `./configure` | Only works if vars are already exported; export first |
| Use `--enable-shared` | NDK's lld doesn't understand macOS `-install_name` flag — use `--disable-shared` |
| Use `-t openmp,pthreads` | Invalid syntax — pick one. Use `pthreads` (NDK OpenMP is unreliable on Android) |
| Use `-t openmp` | NDK OpenMP support is broken — use `pthreads` |
| Use config `cortexa78` | BLIS has no cortexa78 config — use `cortexa57` (same ARMv8a NEON kernels) |
| Use `CFLAGS=-mcpu=cortex-a76` | OnePlus 10R big cores are Cortex-**A78** (CPU part 0xd41), not A76 |

---

### 4. Build llama-server (CPU)

```bash
export NDK="/opt/homebrew/Caskforce/android-ndk/29/AndroidNDK14206865.app/Contents/NDK"

cd benchmark-mobile/llama.cpp

cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+fp16" \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=FLAME \
    -DBLAS_LIBRARIES=$HOME/blis-android/lib/libblis.a \
    -DBLAS_INCLUDE_DIRS=$HOME/blis-android/include/blis \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build-android --target llama-server -j$(sysctl -n hw.ncpu)
```

**Why `GGML_CPU_ARM_ARCH="armv8.2-a+dotprod+fp16"` is critical:**
Without it, cmake auto-detection fails silently — `HAVE_FP16_VECTOR_ARITHMETIC` and `HAVE_DOTPROD` both fail, leaving the build without the optimized NEON kernels. The flag explicitly enables:
- `+dotprod` → `asimddp` kernel paths (confirmed on device)
- `+fp16` → `asimdhp` kernel paths (confirmed on device)
- Do **not** add `+i8mm` — not present on this SoC

---

### 5. Push Binary to Phone

```bash
adb push build-android/bin/llama-server /data/local/tmp/llama-server
adb shell chmod +x /data/local/tmp/llama-server

# Verify
adb shell /data/local/tmp/llama-server --version
# version: 9665 (e3a74b299) built with Clang 21.0.0 for Android aarch64
```

---

### 6. Build llama-server (Vulkan — Mali-G68)

> Only needed for `--backend vulkan` or `--backend both`.

```bash
cmake -B build-android-vulkan \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+fp16" \
    -DGGML_VULKAN=ON \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build-android-vulkan --target llama-server -j$(sysctl -n hw.ncpu)

adb push build-android-vulkan/bin/llama-server /data/local/tmp/llama-server-vulkan
adb shell chmod +x /data/local/tmp/llama-server-vulkan
```

---

### 7. WiFi ADB — Required for Real Power Measurements

When USB is connected, the battery sits at 100% float charge and `current_now` stays zero — you get no power data. The fix: switch ADB to WiFi, then unplug the USB cable. The phone discharges freely during inference and `battery.csv` records real watts.

**The benchmark script handles the phone side automatically** (tcpip mode + connect) in preflight. But the Mac must already be on the same WiFi network as the phone — the script cannot join a WiFi network itself.

#### Step 0 — Connect the Mac to the same WiFi as the phone (one-time)

Check if the Mac is already on the right network:
```bash
ipconfig getifaddr en1   # en1 is usually the Wi-Fi interface on Mac mini
# Should return 192.168.x.x — same subnet as the phone
```

If it returns `169.254.x.x` (link-local) or nothing, the Mac is not on WiFi:
```bash
# Find your WiFi interface name first
networksetup -listallhardwareports | grep -A1 "Wi-Fi"
# Device: en1

# Connect to your network
networksetup -setairportnetwork en1 "YourSSID" "YourPassword"
sleep 8
ipconfig getifaddr en1   # should now show 192.168.x.x
```

> **Note:** The Mac and phone must be on the **same subnet**. If the router has AP/client isolation enabled, devices can't reach each other — disable it in the router settings.

#### Step 1 — One-time phone authorization (first WiFi ADB connection only)

With USB still plugged in:
```bash
adb tcpip 5555
PHONE_IP=$(adb shell ip route show dev wlan0 | awk '{print $9}')
echo "Phone IP: $PHONE_IP"
adb connect "${PHONE_IP}:5555"
# → "failed to authenticate" on first run — that's expected
```

**Tap Allow** on the "Allow USB debugging?" popup on the phone (it re-prompts for WiFi transport). Then:
```bash
adb connect "${PHONE_IP}:5555"
# → "already connected" or "connected to 192.168.x.x:5555"
adb -s "${PHONE_IP}:5555" shell echo ok
# → ok
```

The authorization is saved. Future connections don't need the popup.

#### Step 2 — Run the benchmark with WiFi ADB

```bash
PHONE_IP=$(adb shell ip route show dev wlan0 | awk '{print $9}')

ANDROID_SERIAL="${PHONE_IP}:5555" \
AIPERF_BIN=~/Desktop/smolbenchmark/venv/bin/aiperf \
HF_CLI=~/Desktop/smolbenchmark/venv/bin/hf \
/opt/homebrew/bin/bash benchmark-non-reasoning.sh --reqs 20
```

The script will detect `ANDROID_SERIAL` already contains a dot (WiFi IP) and skip the USB preflight. Unplug USB once the script starts — the phone will begin discharging and `battery.csv` will record real watts.

**Why WiFi overhead doesn't affect results:** llama-server runs entirely on-device. aiperf sends a few KB of JSON per request. At 1–5ms WiFi latency vs 1,000–55,000ms TTFT, the overhead is < 0.5% — measurement noise.

**To restore USB ADB** after the run:
```bash
adb -s "${PHONE_IP}:5555" usb   # switch back to USB mode, then replug cable
```

---

### 9. Set Up Python Environment

> **Important:** `pip install aiperf` from PyPI installs a **deliberately non-functional placeholder** (v0.1.0). The real tool is at [github.com/ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf) and requires Python ≥ 3.11.
>
> Homebrew Python 3.12/3.13/3.14 are broken on macOS 15 due to a `libexpat` dylib symbol mismatch. Use `uv` which bundles its own Python and avoids the issue.

```bash
# uv installs its own Python — no system libexpat dependency
brew install uv
uv venv venv --python 3.12    # run from smolbenchmark/ root
source venv/bin/activate
uv pip install "git+https://github.com/ai-dynamo/aiperf.git" \
               'huggingface_hub[hf_transfer]'

# Verify
aiperf --version   # 0.11.0
hf --version       # 1.19.0
```

> **Note:** The HF CLI is now `hf`, not `huggingface-cli` (deprecated).

---

### 10. Script Fixes Required (macOS)

The benchmark script needed these fixes to run on macOS:

**Fix 1 — Shebang: use bash 5 (macOS ships bash 3.2 which lacks associative arrays)**
```
#!/opt/homebrew/bin/bash
```
Run with: `TMUX=bypass /opt/homebrew/bin/bash benchmark-non-reasoning.sh ...`
(calling as `bash script.sh` ignores the shebang)

**Fix 2 — `date --iso-8601=seconds` is Linux-only**
```bash
# Fixed (POSIX):
date -u +%Y-%m-%dT%H:%M:%SZ
```

**Fix 3 — Empty associative array + `set -u`**
```bash
# Fixed: declare -A FOUND_ZONES=()
```

**Fix 4 — Model download failures exit the script**
```bash
# Added || true so a failed download skips instead of aborting:
ensure_model_on_device ... || true
```

**Fix 5 — aiperf binary not on PATH in background jobs**
```bash
# Script now calls "$AIPERF_BIN" profile instead of bare aiperf
```

---

### 11. How the Script Works

```
Phase 0 — Preflight
    Validates ADB device, llama-server binary, aiperf binary.
    Collects device info (SoC, RAM, CPU freq) → device_info.txt

Phase 1 — Thermal zone discovery
    Primary:  dumpsys thermalservice (HAL) — works on ColorOS
              Discovers: CPU, GPU, BATTERY, SKIN, POWER_AMPLIFIER, NPU
    Fallback: /sys/class/thermal sysfs (standard Android)
    Saves zone map → thermal_zones.txt

Phase 2 — Model staging
    Downloads each GGUF from HuggingFace to host /tmp/smolbench_models/
    Pushes to device /data/local/tmp/models/<family>/
    Skips models already on device; skips gated models on auth failure

Phase 3 — No-op (pollers are per-combo, not global)

Phase 4 — Model loop  [for each model × 9 combos]
    - Start llama-server on device, forward port 8080 via ADB
    - Smoke test (32/128/256-token prompts)
    - For each combo (prompt=64/256/512, gen=64/128/256):
        - mkdir artifacts/android/cpu-<date>/cpu/<Model>/gen<N>/ctx<M>/
        - Start per-combo pollers → thermal.csv, cpufreq.csv, battery.csv
        - Run aiperf: 20 requests, measures TTFT, ITL, output tok/s
        - Stop pollers
        - Cooldown 10s (30s between models)
    - Resume support: --resume <artifact-dir> skips completed combos

Phase 5 — Final poller cleanup + battery snapshot

Phase 6 — Generate Markdown report from artifact tree
```

### 12. Run Benchmarks

```bash
cd benchmark-mobile/Android

# CPU only (default, 20 requests per combo)
TMUX=bypass \
AIPERF_BIN=~/Desktop/smolbenchmark/venv/bin/aiperf \
HF_CLI=~/Desktop/smolbenchmark/venv/bin/hf \
/opt/homebrew/bin/bash benchmark-non-reasoning.sh --reqs 20

# Resume interrupted run
TMUX=bypass \
AIPERF_BIN=~/Desktop/smolbenchmark/venv/bin/aiperf \
HF_CLI=~/Desktop/smolbenchmark/venv/bin/hf \
/opt/homebrew/bin/bash benchmark-non-reasoning.sh \
    --resume artifacts/android/cpu-<YYYYMMDD-HHMM> --reqs 20

# Quick test — single model, 5 requests
TMUX=bypass \
AIPERF_BIN=~/Desktop/smolbenchmark/venv/bin/aiperf \
HF_CLI=~/Desktop/smolbenchmark/venv/bin/hf \
/opt/homebrew/bin/bash benchmark-non-reasoning.sh --only qwen2.5-0.5b --reqs 5

# Vulkan (requires llama-server-vulkan on device)
TMUX=bypass \
AIPERF_BIN=~/Desktop/smolbenchmark/venv/bin/aiperf \
HF_CLI=~/Desktop/smolbenchmark/venv/bin/hf \
/opt/homebrew/bin/bash benchmark-non-reasoning.sh --backend vulkan

# Dry run (no device needed)
TMUX=bypass /opt/homebrew/bin/bash benchmark-non-reasoning.sh --dry-run
```

**Models requiring HuggingFace login** (gated — accept terms at hf.co first):
```bash
hf auth login   # paste token from hf.co/settings/tokens
```
- `google/gemma-3-1b-it`, `google/gemma-3-4b-it`
- `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`

---

## Known Device Quirks

| Issue | Detail |
|---|---|
| `/sys/class/thermal/*/type` permission denied | ColorOS 15 blocks sysfs thermal access. Script uses `dumpsys thermalservice` (HAL) instead — real zone names and values. |
| `date --iso-8601` fails | Android's `date` is BSD-based; use `date -u +%Y-%m-%dT%H:%M:%SZ` |
| `seq` not available in adb shell | Use `$(( i + 1 ))` arithmetic instead |

---

## Benchmark Results

> Benchmarks running — results will be filled in here.

### SmolLM2-135M · Q4_K_M · CPU (Cortex-A78/A55)

| prompt | gen | TTFT avg (ms) | ITL avg (ms) | Output tok/s |
|--------|-----|---------------|--------------|-------------|
| 64 | 64 | 8,393 | 141.6 | 7.1 |
| 64 | 128 | — | — | — |
| 64 | 256 | — | — | — |
| 256 | 64 | — | — | — |
| 256 | 128 | — | — | — |
| 256 | 256 | — | — | — |
| 512 | 64 | — | — | — |
| 512 | 128 | — | — | — |
| 512 | 256 | — | — | — |

> Full results to be added once benchmark run completes.
> Artifacts: `artifacts/android/cpu-20260616-1531/`

---

## File Locations

| Artifact | Path |
|---|---|
| BLIS static library | `~/blis-android/lib/libblis.a` |
| llama-server CPU binary (host) | `benchmark-mobile/llama.cpp/build-android/bin/llama-server` |
| llama-server on device | `/data/local/tmp/llama-server` |
| GGUF models on device | `/data/local/tmp/models/<family>/` |
| Python venv | `~/Desktop/smolbenchmark/venv/` (Python 3.12 via uv) |
| Benchmark script | `benchmark-mobile/Android/benchmark-non-reasoning.sh` |
| Artifacts | `benchmark-mobile/Android/artifacts/android/` |
