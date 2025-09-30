#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="lab01"
#SBATCH --nodes=1
#SBATCH --output="lab01.%j.%N.out"
#SBATCH -t 00:15:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


# Read input path from argument
INPUT_PATH="$1"
if [[ -z "$INPUT_PATH" ]]; then
  echo "Error: No input path provided."
  exit 1
fi


# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8


# Run the benchmark
./bin/spmm_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_spmm_benchmark.json
./blocked_mm_bench  --benchmark_report_aggregates_only=true --benchmark_format=json > results_blocked_mm_bench.json
./bin/sptrsv_benchmark "$INPUT_PATH" --benchmark_report_aggregates_only=true --benchmark_format=json > results_sptrsv_benchmark.json
