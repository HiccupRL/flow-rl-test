#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ALGO="${ALGO:-fpo}"
export EXP_NAME="${EXP_NAME:-isaaclab-fpo-baseline}"

exec bash "${SCRIPT_DIR}/isaaclab-onpolicy-baseline.sh" "$@"
