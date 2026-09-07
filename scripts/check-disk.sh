#!/usr/bin/env bash
# Disk guard for roofline.c.
#
# This project downloads model checkpoints measured in gigabytes and writes
# GGUF exports beside them. Running out of disk mid-export leaves a truncated
# file that still parses far enough to look plausible, so the cheap check is
# worth doing before the expensive thing, not after.
#
#   scripts/check-disk.sh            warn under WARN_GIB, fail under FAIL_GIB
#   scripts/check-disk.sh 3          also require 3 GiB of headroom for a task
#
# Exit: 0 ok (possibly with a warning), 1 not enough space.

set -uo pipefail

FAIL_GIB=${ROOFLINE_DISK_FAIL_GIB:-5}
WARN_GIB=${ROOFLINE_DISK_WARN_GIB:-15}
NEED_GIB=${1:-0}

avail_kib=$(df -Pk . | awk 'NR==2 {print $4}')
avail_gib=$(( avail_kib / 1024 / 1024 ))
pct=$(df -Pk . | awk 'NR==2 {gsub(/%/,"",$5); print $5}')

need=$FAIL_GIB
[ "$NEED_GIB" -gt "$need" ] && need=$NEED_GIB

if [ "$avail_gib" -lt "$need" ]; then
  printf '\033[31mDISK FAIL\033[0m  %s GiB free (%s%% used) — need at least %s GiB\n' \
    "$avail_gib" "$pct" "$need" >&2
  printf '  free space before continuing; a truncated GGUF export still parses\n' >&2
  printf '  far enough to look plausible.\n' >&2
  exit 1
fi

if [ "$avail_gib" -lt "$WARN_GIB" ]; then
  printf '\033[33mDISK WARN\033[0m  %s GiB free (%s%% used)\n' "$avail_gib" "$pct" >&2
  printf '  torch install ~0.6 GiB · BioCLIP-2 checkpoint ~1.6 GiB · f16 export ~0.6 GiB\n' >&2
  exit 0
fi

printf '\033[2mdisk ok: %s GiB free (%s%% used)\033[0m\n' "$avail_gib" "$pct"
