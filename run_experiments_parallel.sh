#!/bin/bash
# Parallel Experiment Runner for MARL-Swarm Scalability Experiments
#
# This script helps run multiple experiments in parallel to maximize GPU utilization
#
# Usage:
#   bash run_experiments_parallel.sh [NUM_PARALLEL] [CONFIG_FILES...]
#
# Examples:
#   # Run 3 experiments in parallel from matrix_quick.json
#   bash run_experiments_parallel.sh 3 matrix_quick
#
#   # Run all single-variable experiments with 2 parallel processes
#   bash run_experiments_parallel.sh 2 embed_scaling depth_scaling
#
#   # Run all configs with 4 parallel processes
#   bash run_experiments_parallel.sh 4 all

set -e

# Default values
NUM_PARALLEL=${1:-2}
shift || true

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}MARL-Swarm Parallel Experiment Runner${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""
echo -e "${YELLOW}Running with $NUM_PARALLEL parallel processes${NC}"
echo ""

# Create log directory
LOGDIR="evaluation/results/logs"
mkdir -p "$LOGDIR"

# Function to run experiment
run_experiment() {
    local config_file=$1
    local exp_name=$(basename "$config_file" .json)
    local logfile="$LOGDIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

    echo -e "${YELLOW}Starting: $exp_name${NC}"
    echo "Log: $logfile"

    if python evaluation/run_scalability_experiment.py --config "$config_file" > "$logfile" 2>&1; then
        echo -e "${GREEN}✓ Completed: $exp_name${NC}"
    else
        echo -e "${RED}✗ Failed: $exp_name (see $logfile)${NC}"
    fi
}

# Export function for parallel execution
export -f run_experiment
export LOGDIR GREEN RED YELLOW NC

# Determine which configs to run
if [ $# -eq 0 ] || [ "$1" == "all" ]; then
    # Run all configs
    CONFIGS=(
        "training/configs/embed_scaling.json"
        "training/configs/depth_scaling.json"
        "training/configs/transferability_small_to_large.json"
        "training/configs/transferability_large_to_small.json"
        "training/configs/edge_cases.json"
        "training/configs/matrix_embed_x_swarmsize.json"
        "training/configs/matrix_depth_x_swarmsize.json"
    )
elif [ "$1" == "matrix_quick" ]; then
    CONFIGS=("training/configs/matrix_quick.json")
elif [ "$1" == "single" ]; then
    # All single-variable experiments
    CONFIGS=(
        "training/configs/embed_scaling.json"
        "training/configs/depth_scaling.json"
        "training/configs/transferability_small_to_large.json"
        "training/configs/transferability_large_to_small.json"
        "training/configs/edge_cases.json"
    )
elif [ "$1" == "matrix" ]; then
    # All matrix experiments
    CONFIGS=(
        "training/configs/matrix_embed_x_swarmsize.json"
        "training/configs/matrix_depth_x_swarmsize.json"
    )
else
    # Custom list of config names
    CONFIGS=()
    for arg in "$@"; do
        CONFIGS+=("training/configs/${arg}.json")
    done
fi

echo -e "${YELLOW}Configs to run:${NC}"
for config in "${CONFIGS[@]}"; do
    echo "  - $config"
done
echo ""

# Run experiments in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
    echo -e "${GREEN}Using GNU parallel${NC}"
    printf '%s\n' "${CONFIGS[@]}" | parallel -j "$NUM_PARALLEL" run_experiment {}
elif command -v xargs &> /dev/null; then
    echo -e "${GREEN}Using xargs${NC}"
    printf '%s\n' "${CONFIGS[@]}" | xargs -I {} -P "$NUM_PARALLEL" bash -c 'run_experiment "$@"' _ {}
else
    echo -e "${YELLOW}Neither 'parallel' nor 'xargs' found. Running sequentially...${NC}"
    for config in "${CONFIGS[@]}"; do
        run_experiment "$config"
    done
fi

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""
echo "Logs available in: $LOGDIR"
