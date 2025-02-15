#!/bin/bash

# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8


# Run the benchmark
./bin/matrix_benchmark
./bin/matrix_benchmark  --benchmark_report_aggregates_only=true --benchmark_format=json > results.json
