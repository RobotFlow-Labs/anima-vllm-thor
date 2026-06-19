#!/usr/bin/env bash
# reclaim_mem.sh — reclaim GPU/unified memory on Jetson AGX Thor.
#
# WHY THIS EXISTS
#   Thor is a Tegra SoC with UNIFIED memory (CPU + integrated GPU share one pool of
#   LPDDR5X). Unlike a discrete NVIDIA GPU, the Tegra `nvgpu` driver has NO gpu-reset.
#   When vLLM's container stops and its CUDA context is not torn down cleanly (SIGKILL,
#   or an orphaned EngineCore subprocess), the driver keeps the GPU carveout PINNED with
#   no owning process. Then:
#     - there is nothing to `kill` (no process holds it),
#     - `echo 3 > /proc/sys/vm/drop_caches` can't help (it's driver-pinned, not cache).
#   The ONLY guaranteed reclaim is a reboot, which re-initializes nvgpu + the carveout.
#
# WHAT THIS DOES (least-destructive first)
#   1. report memory
#   2. stop the vLLM engine GRACEFULLY (SIGTERM + long timeout) so CUDA can tear down
#      — this is the path that AVOIDS the leak; do it before every model swap
#   3. reap any orphaned CUDA holders (EngineCore / vllm / pt_main_thread)
#   4. drop page cache (harmless; helps the *CPU-side* number)
#   5. re-measure. If enough was reclaimed -> DONE, no reboot.
#   6. else, only with --reboot, reboot (autoserve restores the last model on boot).
#
# USAGE
#   ./reclaim_mem.sh                 # graceful reclaim, report, never reboots
#   ./reclaim_mem.sh --reboot        # graceful first; reboot if still leaked
#   ./reclaim_mem.sh --need 40       # "enough" = >= 40 GB free (default 50)
#
# Run ON the Thor box (or via: ssh thor 'bash -s' < ops/reclaim_mem.sh -- --reboot).
set -uo pipefail

NEED_GB=50
DO_REBOOT=0
CONTAINER="anima-vllm"
while [ $# -gt 0 ]; do
  case "$1" in
    --reboot) DO_REBOOT=1 ;;
    --need)   NEED_GB="$2"; shift ;;
    --container) CONTAINER="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done

free_gb(){ free -g | awk '/Mem:/{print $7}'; }
say(){ echo "[reclaim] $*"; }

say "memory before: $(free_gb) GB free"

# 1+2. graceful stop — give CUDA up to 60s to tear the context down cleanly
if sudo docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  say "stopping $CONTAINER gracefully (docker stop -t 60)..."
  sudo docker stop -t 60 "$CONTAINER" >/dev/null 2>&1 || true
  sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
else
  say "$CONTAINER not running"
fi

# 3. reap orphaned CUDA holders (these keep the context alive if the parent died)
for pat in EngineCore "vllm serve" pt_main_thread "from multiprocessing"; do
  pids=$(pgrep -f "$pat" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    say "reaping orphaned '$pat' pids: $pids"
    # shellcheck disable=SC2086
    sudo kill -TERM $pids 2>/dev/null || true
  fi
done
sleep 5
for pat in EngineCore "vllm serve" pt_main_thread; do
  pids=$(pgrep -f "$pat" 2>/dev/null || true)
  # shellcheck disable=SC2086
  [ -n "$pids" ] && { say "force-killing lingering '$pat': $pids"; sudo kill -9 $pids 2>/dev/null || true; }
done

# 4. drop page cache (CPU-side only; does NOT touch driver-pinned GPU memory)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
sleep 3

AVAIL=$(free_gb)
say "memory after graceful reclaim: ${AVAIL} GB free (target >= ${NEED_GB} GB)"

if [ "${AVAIL:-0}" -ge "$NEED_GB" ] 2>/dev/null; then
  say "RECLAIMED enough without a reboot. Done."
  exit 0
fi

say "STILL LEAKED: ${AVAIL} GB < ${NEED_GB} GB — driver-pinned GPU memory, only a reboot frees it."
if [ "$DO_REBOOT" = 1 ]; then
  say "rebooting now (autoserve will restore the last model on boot)..."
  sudo reboot
else
  say "re-run with --reboot to reclaim it (or 'ssh thor sudo reboot'). NOT rebooting (no --reboot)."
  exit 1
fi
