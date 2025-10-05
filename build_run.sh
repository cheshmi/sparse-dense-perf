#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 24:00:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


# Check for password argument
if [ -z "$1" ]; then
  echo "Usage: sbatch run.sh <sudo_password>"
  exit 1
fi

SUDO_PASS="$1"

# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8


# Run the benchmark
./bin/spmm_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_spmm_benchmark.json
./blocked_mm_bench  --benchmark_report_aggregates_only=true --benchmark_format=json > results_blocked_mm_bench.json


#Run the benchmark with profiling on nsys
nsys profile -t cuda \
             --output ./profiling_nsys_spmm_benchmark \
             ./bin/spmm_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_spmm_benchmark_nsys.json

nsys profile -t cuda \
             --output ./profiling_nsys_blocked_mm_bench \
             ./blocked_mm_bench  --benchmark_report_aggregates_only=true --benchmark_format=json > results_blocked_mm_bench_nsys.json


# Run the benchmark with profiling on nsight compute
echo "$SUDO_PASS" | sudo -S ncu --target-processes all --set full -o ./profiling_out_put_spmm_benchmark  ./bin/spmm_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_spmm_benchmark_with_profiling.json
echo "$SUDO_PASS" | sudo -S ncu --target-processes all --set full -o ./profiling_out_put_blocked_mm_bench ./blocked_mm_bench  --benchmark_report_aggregates_only=true --benchmark_format=json > results_blocked_mm_bench_with_profiling.json