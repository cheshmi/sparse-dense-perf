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
include CMakeFiles/matrix_benchmark.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/matrix_benchmark.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/matrix_benchmark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrix_benchmark.dir/flags.make

CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o: CMakeFiles/matrix_benchmark.dir/flags.make
CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o: ../matrix_benchmark.cpp
CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o: CMakeFiles/matrix_benchmark.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/albakrih/Desktop/llm/sparse-dense-perf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o -MF CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o.d -o CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o -c /home/albakrih/Desktop/llm/sparse-dense-perf/matrix_benchmark.cpp

CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/albakrih/Desktop/llm/sparse-dense-perf/matrix_benchmark.cpp > CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.i

CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/albakrih/Desktop/llm/sparse-dense-perf/matrix_benchmark.cpp -o CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.s

# Object files for target matrix_benchmark
matrix_benchmark_OBJECTS = \
"CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o"

# External object files for target matrix_benchmark
matrix_benchmark_EXTERNAL_OBJECTS =

bin/matrix_benchmark: CMakeFiles/matrix_benchmark.dir/matrix_benchmark.cpp.o
bin/matrix_benchmark: CMakeFiles/matrix_benchmark.dir/build.make
bin/matrix_benchmark: _deps/googlebenchmark-build/src/libbenchmark.a
bin/matrix_benchmark: CMakeFiles/matrix_benchmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/albakrih/Desktop/llm/sparse-dense-perf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/matrix_benchmark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_benchmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrix_benchmark.dir/build: bin/matrix_benchmark
.PHONY : CMakeFiles/matrix_benchmark.dir/build

CMakeFiles/matrix_benchmark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrix_benchmark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrix_benchmark.dir/clean

CMakeFiles/matrix_benchmark.dir/depend:
	cd /home/albakrih/Desktop/llm/sparse-dense-perf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/albakrih/Desktop/llm/sparse-dense-perf /home/albakrih/Desktop/llm/sparse-dense-perf /home/albakrih/Desktop/llm/sparse-dense-perf/build /home/albakrih/Desktop/llm/sparse-dense-perf/build /home/albakrih/Desktop/llm/sparse-dense-perf/build/CMakeFiles/matrix_benchmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrix_benchmark.dir/depend

