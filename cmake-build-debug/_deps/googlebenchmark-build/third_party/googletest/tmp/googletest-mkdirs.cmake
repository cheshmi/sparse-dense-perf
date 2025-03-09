# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/src"
  "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/build"
  "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest"
  "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/tmp"
  "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/stamp"
  "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/download"
  "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/albakrih/Desktop/llm/sparse-dense-perf/cmake-build-debug/_deps/googlebenchmark-build/third_party/googletest/stamp${cfgdir}") # cfgdir has leading slash
endif()
