#!/bin/bash

# Script to run multiple train_gpt2cu commands synchronously
# Usage: ./launch_multiple_runs.sh
# Edit the 'commands' array below to add/modify commands to run

# Define commands to run in an array
declare -a commands=(
    "./train_gpt2cu 2 -b 4 -t 64"
    "./train_gpt2cu 2 -b 4 -t 128"
    # Add more commands below as needed
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
echo "Running $total_commands train_gpt2cu commands synchronously"
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
