# sparse-dense-perf

A simple benchmark to compare the performance of sparse and dense matrix multiplication using cuSparse and cuBLAS.


## Build
If paths are set correctly, the following commands should build the project.
```aiignore
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Otherwise, you can set the paths manually like:
```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_PREFIX_PATH=/usr/local/cuda/lib64 ..
make 
```



