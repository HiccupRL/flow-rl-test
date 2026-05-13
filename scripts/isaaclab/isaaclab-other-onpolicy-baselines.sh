#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ALGOS="${ALGOS:-dppo fpo fpopp genpo}"
exec bash "${SCRIPT_DIR}/isaaclab-all-onpolicy-baselines.sh" "$@"
