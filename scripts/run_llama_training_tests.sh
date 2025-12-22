#!/bin/bash

# Script to run multiple llama_test commands with different configurations
# Usage: ./scripts/run_llama_tests.sh
#
# This script tests llama_test with various matrix sizes (N) and batch counts (B):
# - N values: 1024, 2048, 4096, 8192, 16384
# - B values: 8, 16, 32
# Total: 15 test configurations

# Define commands to run in an array
declare -a commands=(
    # # B=8
    # "./llama_training_test 1024 8"
    # "./llama_training_test 2048 8"
    # "./llama_training_test 4096 8"
    # "./llama_training_test 8192 8"
    # "./llama_training_test 16384 8"

    # # B=16
    # "./llama_training_test 1024 16"
    # "./llama_training_test 2048 16"
    # "./llama_training_test 4096 16"
    # "./llama_training_test 8192 16"
    # "./llama_training_test 16384 16"

    # # B=32
    # "./llama_training_test 1024 32"
    # "./llama_training_test 2048 32"
    # "./llama_training_test 4096 32"
    # "./llama_training_test 8192 32"
    # "./llama_training_test 16384 32"

    # # B=64
    # "./llama_training_test 1024 64"
    # "./llama_training_test 2048 64"
    # "./llama_training_test 4096 64"
    # "./llama_training_test 8192 64"
    # "./llama_training_test 16384 64"


    "./llama_training_test 1024 32"
    "./llama_training_test 2048 32"
    "./llama_training_test 4096 32"
    "./llama_training_test 8192 32"
    "./llama_training_test 16384 32"
    "./llama_training_test 32768 32"
    "./llama_training_test 65536 32"
    "./llama_training_test 131072 32"




    "./llama_training_test 1024 64"
    "./llama_training_test 2048 64"
    "./llama_training_test 4096 64"
    "./llama_training_test 8192 64"
    "./llama_training_test 16384 64"
    "./llama_training_test 32768 64"
    "./llama_training_test 65536 64"
    "./llama_training_test 131072 64"


    "./llama_training_test 1024 128"
    "./llama_training_test 2048 128"
    "./llama_training_test 4096 128"
    "./llama_training_test 8192 128"
    "./llama_training_test 16384 128"
    "./llama_training_test 32768 128"
    "./llama_training_test 65536 128"
    "./llama_training_test 131072 128"

)

# Check if at least one command is provided
if [ ${#commands[@]} -eq 0 ]; then
    echo "Error: No commands defined in the array."
    echo "Edit the 'commands' array in this script to add commands."
    exit 1
fi

# Counter for tracking progress
total_commands=${#commands[@]}
completed_commands=0

echo "=========================================="
echo "Running $total_commands llama_training_test commands synchronously"
echo "=========================================="
echo

# Run each command sequentially
for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    command_num=$((i + 1))

    echo "[${command_num}/${total_commands}] Executing: $cmd"
    echo "-------------------------------------------"

    # Execute the command
    if eval "$cmd"; then
        ((completed_commands++))
        echo "[${command_num}/${total_commands}] ✓ Command completed successfully"
    else
        exit_code=$?
        echo "[${command_num}/${total_commands}] ✗ Command failed with exit code $exit_code"
    fi
    echo
done

echo "=========================================="
echo "All $completed_commands commands completed successfully!"
echo "=========================================="
