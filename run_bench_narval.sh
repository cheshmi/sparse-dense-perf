#!/bin/bash

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=hosseinalbakri3@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 00:30:00
#SBATCH --mem=50G  # Request 32 GB of memory

module load StdEnv/2020
module load intel/2022.2.1
echo " +-+-+-+-+ ========> ${MKLROOT}"
export MKL_DIR=$MKLROOT
echo " +-+-+-+-+ ========> ${MKL_DIR}"
module load cmake
module load gcc
module load python
module load cuda/12.2


# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

python3 -m venv venv
source venv/bin/activate
pip install matplotlib

# Run the benchmark
./bin/matrix_benchmark
./bin/matrix_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_A100.json
