# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/albakrih/Desktop/llm/sparse-dense-perf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/albakrih/Desktop/llm/sparse-dense-perf/build

# Include any dependencies generated for this target.
include _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/flags.make

_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o: _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/flags.make
_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o: _deps/googlebenchmark-src/test/benchmark_min_time_flag_time_test.cc
_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o: _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/albakrih/Desktop/llm/sparse-dense-perf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o"
	cd /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o -MF CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o.d -o CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o -c /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-src/test/benchmark_min_time_flag_time_test.cc

_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.i"
	cd /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-src/test/benchmark_min_time_flag_time_test.cc > CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.i

_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.s"
	cd /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-src/test/benchmark_min_time_flag_time_test.cc -o CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.s

# Object files for target benchmark_min_time_flag_time_test
benchmark_min_time_flag_time_test_OBJECTS = \
"CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o"

# External object files for target benchmark_min_time_flag_time_test
benchmark_min_time_flag_time_test_EXTERNAL_OBJECTS =

_deps/googlebenchmark-build/test/benchmark_min_time_flag_time_test: _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/benchmark_min_time_flag_time_test.cc.o
_deps/googlebenchmark-build/test/benchmark_min_time_flag_time_test: _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/build.make
_deps/googlebenchmark-build/test/benchmark_min_time_flag_time_test: _deps/googlebenchmark-build/src/libbenchmark.a
_deps/googlebenchmark-build/test/benchmark_min_time_flag_time_test: _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/albakrih/Desktop/llm/sparse-dense-perf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable benchmark_min_time_flag_time_test"
	cd /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_min_time_flag_time_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/build: _deps/googlebenchmark-build/test/benchmark_min_time_flag_time_test
.PHONY : _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/build

_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/clean:
	cd /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-build/test && $(CMAKE_COMMAND) -P CMakeFiles/benchmark_min_time_flag_time_test.dir/cmake_clean.cmake
.PHONY : _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/clean

_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/depend:
	cd /home/albakrih/Desktop/llm/sparse-dense-perf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/albakrih/Desktop/llm/sparse-dense-perf /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-src/test /home/albakrih/Desktop/llm/sparse-dense-perf/build /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-build/test /home/albakrih/Desktop/llm/sparse-dense-perf/build/_deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/googlebenchmark-build/test/CMakeFiles/benchmark_min_time_flag_time_test.dir/depend

