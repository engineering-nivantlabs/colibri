# Nivant GPU box: CPU RAM + Radeon VRAM

This profile uses the existing Vulkan expert tier: hot routed experts in VRAM,
warm experts in system RAM, and cold experts read from the model on disk. CPU
and GPU execute their share of experts concurrently. Exclusive caching gives
up redundant CPU copies when RAM is tight. This is application-level tiering,
not coherent hardware unified memory or a promise of VRAM bandwidth from RAM.

## Hardware validated

- Linux x86-64, Ryzen 9 9900X, 12 cores / 24 threads.
- Roughly 30 GiB usable host RAM and Radeon RX 7900 XTX, 24 GiB VRAM
  (Vulkan driver's identification; ROCm's board label disagrees).
- Mesa RADV Vulkan backend, native GCC build; no CUDA/ROCm compiler needed.
- Existing Qwen3-Coder 30B Q4_K_M GGUF, 48 layers, 128 experts/layer.

## Build and use

```sh
make -C c qwen36 VK=1 -j4
python3 c/tests/test_gpu_box.py
printf '<|im_start|>user\nWhat is a GPU? Answer briefly.\n<|im_end|>\n<|im_start|>assistant\n' > /tmp/colibri-prompt.txt
python3 c/tools/gpu_box.py \
  --model "$HOME/models/qwen3-coder-30b-q4km.gguf" \
  --prompt /tmp/colibri-prompt.txt
# Inspect the printed plan, then add --run to execute.
```

The launcher defaults to 12 CPU threads, a 2048-token context, and 32 generated
tokens. It leaves 6 GiB of currently available RAM outside its planning target,
caps that target at 12 decimal GB, asks Vulkan to reserve 3 GiB beyond expert
storage, and caps its mapped host pool at 2 GiB. The primary device defaults to
Colibri's discrete-GPU preference; the small integrated GPU stays unused.

The engine chooses RAM expert slots from `RAM_GB` with `--cap 0` (default).
An explicit `--cap N` bypasses this sizing and means slots **per layer**, not GB.
These are estimates, not enforced RSS/cgroup limits. They do not guarantee a
model fits; context, model geometry, scratch allocations and other processes
matter. The launcher refuses to start if less than 4 GiB remains after its host
reserve. It does not stop services, clear swap, download models or start a
network server. Its `--run` invokes the local engine and exits after generation.

`--cpu` disables GPU execution for a baseline. Compare the same prompt, thread
count, token count and explicit `--cap` to hold cache capacity constant. Cache
warmth, concurrent workloads and compilation affect timings; record them.

Inherited Vulkan/device/fault settings are cleared so an old experiment cannot
silently select the iGPU, enable a second device, or defeat the memory reserves.

## Initial smoke test

The initial 8-slot/layer profile generated 16 tokens successfully, used 7.71
decimal GB peak RSS, and routed 65.5% of expert invocations to the GPU. It gave
up 88 redundant RAM copies. However, 754 disk loads and 58 seconds for generation
made this unsuitable as a performance recommendation (0.28 tokens/sec reported).
The broader checks were compiling concurrently, so this is not an isolated
benchmark. The default now uses RAM-budget sizing and a smaller host pool;
the subsequent measurements below document this profile.

## Subsequent validation (2026-11-05)

Same 19-token prompt, 16 generated tokens, 12 CPU threads, context 2048:

| Mode | Reported speed | TTFT | Peak host RSS |
| --- | ---: | ---: | ---: |
| Vulkan, RAM auto sizing, host pool 2 GiB | 1.43 tok/s | 4.145 s | 3.26 GB |
| CPU, RAM auto sizing | 2.53 tok/s | 3.930 s | 7.26 GB |

Both runs auto-sized the RAM cache rather than fixing all residency. These are
smoke measurements, not controlled throughput comparisons. The Vulkan run had
14.58 GiB available to its expert tier, versus 8.51 GiB in the first run because
other workloads' memory use changed. The earlier and later numbers cannot
isolate the effect of the launcher changes. Compilation overlapped portions
of validation. CPU and GPU continuations differed in their final words; no
token-exact equivalence or model-quality comparison is claimed.

The later Vulkan run served 94.6% of routed expert invocations on the device,
used a 1.98 GiB host pool, and read 19 cold experts from disk. CPU execution is
the faster observed choice for this short prompt; Vulkan reduces host-memory
use and offers additional model capacity. More capacity does not imply speed.

Passed: native Vulkan build, GGUF header/tensor/tokenizer compatibility check,
three launcher tests, upstream `test_qwen36_cache_route`, `test_qwen36_ctx`,
`test_qwen36_dense_batch`, and CPU/GPU text generation. Full `make check` was
started but deliberately stopped during its lengthy C compilation; it did not
complete and is not reported as passing. Existing upstream test compilation
emitted unused-variable warnings; the Vulkan engine build completed cleanly.

No kernel allocator changes or new managed-memory backend are claimed. Upstream
already implements the GPU/RAM/disk scheduling this launcher configures.
