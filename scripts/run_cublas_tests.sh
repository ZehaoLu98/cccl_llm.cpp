#!/bin/bash

# Script to run multiple cublas_test commands with different configurations
# Usage: ./scripts/run_cublas_tests.sh
#
# This script tests cublas_test with various matrix sizes (N) and batch counts (B):
# - N values: 1024, 2048, 4096, 8192, 16384
# - B values: 8, 16, 32
# Total: 15 test configurations

# Define commands to run in an array
declare -a commands=(
    # # B=8
    # "./cublas_test 1024 8"
    # "./cublas_test 2048 8"
    # "./cublas_test 4096 8"
    # "./cublas_test 8192 8"
    # "./cublas_test 16384 8"

    # # B=16
    # "./cublas_test 1024 16"
    # "./cublas_test 2048 16"
    # "./cublas_test 4096 16"
    # "./cublas_test 8192 16"
    # "./cublas_test 16384 16"

    # # B=32
    # "./cublas_test 1024 32"
    # "./cublas_test 2048 32"
    # "./cublas_test 4096 32"
    # "./cublas_test 8192 32"
    # "./cublas_test 16384 32"

    # # B=64
    # "./cublas_test 1024 64"
    # "./cublas_test 2048 64"
    # "./cublas_test 4096 64"
    # "./cublas_test 8192 64"
    # "./cublas_test 16384 64"


    "./cublas_test 1024 2"
    "./cublas_test 1024 4"
    "./cublas_test 1024 8"
    "./cublas_test 1024 16"
    "./cublas_test 1024 32"
    "./cublas_test 1024 64"
    "./cublas_test 1024 128"
    "./cublas_test 1024 256"
    "./cublas_test 1024 512"

    "./cublas_test 2560 2"
    "./cublas_test 2560 4"
    "./cublas_test 2560 8"
    "./cublas_test 2560 16"
    "./cublas_test 2560 32"
    "./cublas_test 2560 64"
    "./cublas_test 2560 128"
    "./cublas_test 2560 256"
    "./cublas_test 2560 512"


    "./cublas_test 4096 2"
    "./cublas_test 4096 4"
    "./cublas_test 4096 8"
    "./cublas_test 4096 16"
    "./cublas_test 4096 32"
    "./cublas_test 4096 64"
    "./cublas_test 4096 128"
    "./cublas_test 4096 256"
    "./cublas_test 4096 512"
    "./cublas_test 4096 1024"
    "./cublas_test 4096 2048"

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
echo "Running $total_commands cublas_test commands synchronously"
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
