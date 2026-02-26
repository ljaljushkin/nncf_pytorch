#!/bin/bash
set -euo pipefail
# ── Compare FQ_STRETCHED_LORA vs FQ_LORA with identical hyperparameters ──
#
# Usage:
#   ./run_compare_formats.sh                          # uses built-in default config
#   ./run_compare_formats.sh --configs run_configs.txt    # uses configs from file
#   ./run_compare_formats.sh --debug                  # debug mode (separate DB)
#
# This script runs every configuration twice — once with FQ_STRETCHED_LORA
# (ParetoQ-style) and once with FQ_LORA (standard) — so that results are
# directly comparable in the same MLflow experiment.
# ─────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=()

# Forward all arguments; inject --compression_format ourselves.
for arg in "$@"; do
    # Strip any user-provided --compression_format (we control it).
    if [[ "$arg" == "--compression_format" ]]; then
        shift 2 2>/dev/null || true
        continue
    fi
    EXTRA_ARGS+=("$arg")
done

echo "═══════════════════════════════════════════════════════════════"
echo "  Pass 1/2: FQ_STRETCHED_LORA (ParetoQ)"
echo "═══════════════════════════════════════════════════════════════"
"$SCRIPT_DIR/run_grid_search.sh" "${EXTRA_ARGS[@]}" --compression_format FQ_STRETCHED_LORA

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Pass 2/2: FQ_LORA (standard)"
echo "═══════════════════════════════════════════════════════════════"
"$SCRIPT_DIR/run_grid_search.sh" "${EXTRA_ARGS[@]}" --compression_format FQ_LORA

echo ""
echo "Done. Both formats logged to the same MLflow experiment for comparison."
