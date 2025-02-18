#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=hosseinalbakri3@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 00:30:00

module load MistEnv/2021a
module load cmake/3.27.8
module load gcc
module load anaconda3/2021.05
module load cuda/11.8.0

# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

# Run the benchmark
./bin/matrix_benchmark
./bin/matrix_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results_V100.json
