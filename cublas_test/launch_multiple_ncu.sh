#!/bin/bash

# This script launches multiple instances of NVIDIA's Nsight Compute (ncu)
# to profile different GPU applications simultaneously.
sudo ncu --set full --replay-mode application --nvtx --nvtx-include "BatchedGEMM" --nvtx-include "NaiveGEMM"  -o n_1024_k_16 ./cublas_test 1024 16
