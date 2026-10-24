#!/usr/bin/env bash
set -euo pipefail

# Restart CANable/candleLight adapters exposed by Linux as gs_usb SocketCAN.
# Usage:
#   scripts/canable_restart.sh
#   scripts/canable_restart.sh can0
#   scripts/canable_restart.sh --bitrate 1000000 can0

BITRATE=1000000
LOOPBACK=off
TX_QLEN=2000
CHECK_DRIVER=1
IFS_LIST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bitrate)
      BITRATE="$2"
      shift 2
      ;;
    --loopback)
      LOOPBACK="$2"
      shift 2
      ;;
    --txqueuelen)
      TX_QLEN="$2"
      shift 2
      ;;
    --no-driver-check)
      CHECK_DRIVER=0
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Restart CANable/candleLight gs_usb SocketCAN interfaces.

Options:
  --bitrate <num>       CAN bitrate (default: 1000000)
  --loopback <on|off>   loopback mode (default: off)
  --txqueuelen <num>    TX queue length (default: 2000)
  --no-driver-check     skip gs_usb verification
  -h, --help            show help

Examples:
  scripts/canable_restart.sh
  scripts/canable_restart.sh can0
  scripts/canable_restart.sh --bitrate 1000000 can0
EOF
      exit 0
      ;;
    *)
      IFS_LIST+=("$1")
      shift
      ;;
  esac
done

if [[ ${#IFS_LIST[@]} -eq 0 ]]; then
  IFS_LIST=(can0)
fi

restart_one() {
  local ifn="$1"
  if ! ip link show "$ifn" >/dev/null 2>&1; then
    echo "[canable_restart] skip ${ifn}: interface not found"
    return 0
  fi

  local before
  before="$(ip -details link show "$ifn")"
  if [[ "$CHECK_DRIVER" -eq 1 ]] && ! grep -q "gs_usb" <<<"$before"; then
    echo "[canable_restart] error: ${ifn} does not look like gs_usb/candleLight" >&2
    echo "$before" >&2
    return 1
  fi

  echo "[canable_restart] restarting ${ifn} bitrate=${BITRATE} loopback=${LOOPBACK} txqueuelen=${TX_QLEN}"
  sudo ip link set "$ifn" down 2>/dev/null || true
  sudo ip link set "$ifn" txqueuelen "$TX_QLEN"
  sudo ip link set "$ifn" type can bitrate "$BITRATE" loopback "$LOOPBACK"
  sudo ip link set "$ifn" up
  ip -details link show "$ifn"
}

for ifn in "${IFS_LIST[@]}"; do
  restart_one "$ifn"
done
