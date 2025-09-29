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

# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8


# Run the benchmark
./bin/spmm_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_spmm_benchmark.json
./bin/sptrsv_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_sptrsv_benchmark.json
