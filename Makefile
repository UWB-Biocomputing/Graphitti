# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.17.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.17.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/tori/Projects/SummerOfBrain

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/tori/Projects/SummerOfBrain

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/local/Cellar/cmake/3.17.3/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/local/Cellar/cmake/3.17.3/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/local/Cellar/cmake/3.17.3/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/local/Cellar/cmake/3.17.3/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/local/Cellar/cmake/3.17.3/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/local/Cellar/cmake/3.17.3/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/Cellar/cmake/3.17.3/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/local/Cellar/cmake/3.17.3/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/tori/Projects/SummerOfBrain/CMakeFiles /Users/tori/Projects/SummerOfBrain/CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/tori/Projects/SummerOfBrain/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named SummerOfBrain

# Build rule for target.
SummerOfBrain: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 SummerOfBrain
.PHONY : SummerOfBrain

# fast build rule for target.
SummerOfBrain/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/build
.PHONY : SummerOfBrain/fast

#=============================================================================
# Target rules for targets named gmock

# Build rule for target.
gmock: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 gmock
.PHONY : gmock

# fast build rule for target.
gmock/fast:
	$(MAKE) $(MAKESILENT) -f Testing/lib/googletest-master/googlemock/CMakeFiles/gmock.dir/build.make Testing/lib/googletest-master/googlemock/CMakeFiles/gmock.dir/build
.PHONY : gmock/fast

#=============================================================================
# Target rules for targets named gmock_main

# Build rule for target.
gmock_main: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 gmock_main
.PHONY : gmock_main

# fast build rule for target.
gmock_main/fast:
	$(MAKE) $(MAKESILENT) -f Testing/lib/googletest-master/googlemock/CMakeFiles/gmock_main.dir/build.make Testing/lib/googletest-master/googlemock/CMakeFiles/gmock_main.dir/build
.PHONY : gmock_main/fast

#=============================================================================
# Target rules for targets named gtest_main

# Build rule for target.
gtest_main: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 gtest_main
.PHONY : gtest_main

# fast build rule for target.
gtest_main/fast:
	$(MAKE) $(MAKESILENT) -f Testing/lib/googletest-master/googletest/CMakeFiles/gtest_main.dir/build.make Testing/lib/googletest-master/googletest/CMakeFiles/gtest_main.dir/build
.PHONY : gtest_main/fast

#=============================================================================
# Target rules for targets named gtest

# Build rule for target.
gtest: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 gtest
.PHONY : gtest

# fast build rule for target.
gtest/fast:
	$(MAKE) $(MAKESILENT) -f Testing/lib/googletest-master/googletest/CMakeFiles/gtest.dir/build.make Testing/lib/googletest-master/googletest/CMakeFiles/gtest.dir/build
.PHONY : gtest/fast

ChainOfResponsibility/Foo.o: ChainOfResponsibility/Foo.cpp.o

.PHONY : ChainOfResponsibility/Foo.o

# target to build an object file
ChainOfResponsibility/Foo.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/ChainOfResponsibility/Foo.cpp.o
.PHONY : ChainOfResponsibility/Foo.cpp.o

ChainOfResponsibility/Foo.i: ChainOfResponsibility/Foo.cpp.i

.PHONY : ChainOfResponsibility/Foo.i

# target to preprocess a source file
ChainOfResponsibility/Foo.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/ChainOfResponsibility/Foo.cpp.i
.PHONY : ChainOfResponsibility/Foo.cpp.i

ChainOfResponsibility/Foo.s: ChainOfResponsibility/Foo.cpp.s

.PHONY : ChainOfResponsibility/Foo.s

# target to generate assembly for a file
ChainOfResponsibility/Foo.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/ChainOfResponsibility/Foo.cpp.s
.PHONY : ChainOfResponsibility/Foo.cpp.s

ChainOfResponsibility/GenericFunctionNode.o: ChainOfResponsibility/GenericFunctionNode.cpp.o

.PHONY : ChainOfResponsibility/GenericFunctionNode.o

# target to build an object file
ChainOfResponsibility/GenericFunctionNode.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/ChainOfResponsibility/GenericFunctionNode.cpp.o
.PHONY : ChainOfResponsibility/GenericFunctionNode.cpp.o

ChainOfResponsibility/GenericFunctionNode.i: ChainOfResponsibility/GenericFunctionNode.cpp.i

.PHONY : ChainOfResponsibility/GenericFunctionNode.i

# target to preprocess a source file
ChainOfResponsibility/GenericFunctionNode.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/ChainOfResponsibility/GenericFunctionNode.cpp.i
.PHONY : ChainOfResponsibility/GenericFunctionNode.cpp.i

ChainOfResponsibility/GenericFunctionNode.s: ChainOfResponsibility/GenericFunctionNode.cpp.s

.PHONY : ChainOfResponsibility/GenericFunctionNode.s

# target to generate assembly for a file
ChainOfResponsibility/GenericFunctionNode.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/ChainOfResponsibility/GenericFunctionNode.cpp.s
.PHONY : ChainOfResponsibility/GenericFunctionNode.cpp.s

Core/OperationManager.o: Core/OperationManager.cpp.o

.PHONY : Core/OperationManager.o

# target to build an object file
Core/OperationManager.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Core/OperationManager.cpp.o
.PHONY : Core/OperationManager.cpp.o

Core/OperationManager.i: Core/OperationManager.cpp.i

.PHONY : Core/OperationManager.i

# target to preprocess a source file
Core/OperationManager.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Core/OperationManager.cpp.i
.PHONY : Core/OperationManager.cpp.i

Core/OperationManager.s: Core/OperationManager.cpp.s

.PHONY : Core/OperationManager.s

# target to generate assembly for a file
Core/OperationManager.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Core/OperationManager.cpp.s
.PHONY : Core/OperationManager.cpp.s

Testing/ChainOfResponsibility/FunctionNodeTests.o: Testing/ChainOfResponsibility/FunctionNodeTests.cpp.o

.PHONY : Testing/ChainOfResponsibility/FunctionNodeTests.o

# target to build an object file
Testing/ChainOfResponsibility/FunctionNodeTests.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/ChainOfResponsibility/FunctionNodeTests.cpp.o
.PHONY : Testing/ChainOfResponsibility/FunctionNodeTests.cpp.o

Testing/ChainOfResponsibility/FunctionNodeTests.i: Testing/ChainOfResponsibility/FunctionNodeTests.cpp.i

.PHONY : Testing/ChainOfResponsibility/FunctionNodeTests.i

# target to preprocess a source file
Testing/ChainOfResponsibility/FunctionNodeTests.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/ChainOfResponsibility/FunctionNodeTests.cpp.i
.PHONY : Testing/ChainOfResponsibility/FunctionNodeTests.cpp.i

Testing/ChainOfResponsibility/FunctionNodeTests.s: Testing/ChainOfResponsibility/FunctionNodeTests.cpp.s

.PHONY : Testing/ChainOfResponsibility/FunctionNodeTests.s

# target to generate assembly for a file
Testing/ChainOfResponsibility/FunctionNodeTests.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/ChainOfResponsibility/FunctionNodeTests.cpp.s
.PHONY : Testing/ChainOfResponsibility/FunctionNodeTests.cpp.s

Testing/ChainOfResponsibility/OperationManagerTests.o: Testing/ChainOfResponsibility/OperationManagerTests.cpp.o

.PHONY : Testing/ChainOfResponsibility/OperationManagerTests.o

# target to build an object file
Testing/ChainOfResponsibility/OperationManagerTests.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/ChainOfResponsibility/OperationManagerTests.cpp.o
.PHONY : Testing/ChainOfResponsibility/OperationManagerTests.cpp.o

Testing/ChainOfResponsibility/OperationManagerTests.i: Testing/ChainOfResponsibility/OperationManagerTests.cpp.i

.PHONY : Testing/ChainOfResponsibility/OperationManagerTests.i

# target to preprocess a source file
Testing/ChainOfResponsibility/OperationManagerTests.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/ChainOfResponsibility/OperationManagerTests.cpp.i
.PHONY : Testing/ChainOfResponsibility/OperationManagerTests.cpp.i

Testing/ChainOfResponsibility/OperationManagerTests.s: Testing/ChainOfResponsibility/OperationManagerTests.cpp.s

.PHONY : Testing/ChainOfResponsibility/OperationManagerTests.s

# target to generate assembly for a file
Testing/ChainOfResponsibility/OperationManagerTests.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/ChainOfResponsibility/OperationManagerTests.cpp.s
.PHONY : Testing/ChainOfResponsibility/OperationManagerTests.cpp.s

Testing/RunTests.o: Testing/RunTests.cpp.o

.PHONY : Testing/RunTests.o

# target to build an object file
Testing/RunTests.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/RunTests.cpp.o
.PHONY : Testing/RunTests.cpp.o

Testing/RunTests.i: Testing/RunTests.cpp.i

.PHONY : Testing/RunTests.i

# target to preprocess a source file
Testing/RunTests.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/RunTests.cpp.i
.PHONY : Testing/RunTests.cpp.i

Testing/RunTests.s: Testing/RunTests.cpp.s

.PHONY : Testing/RunTests.s

# target to generate assembly for a file
Testing/RunTests.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SummerOfBrain.dir/build.make CMakeFiles/SummerOfBrain.dir/Testing/RunTests.cpp.s
.PHONY : Testing/RunTests.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... install"
	@echo "... install/local"
	@echo "... install/strip"
	@echo "... list_install_components"
	@echo "... rebuild_cache"
	@echo "... SummerOfBrain"
	@echo "... gmock"
	@echo "... gmock_main"
	@echo "... gtest"
	@echo "... gtest_main"
	@echo "... ChainOfResponsibility/Foo.o"
	@echo "... ChainOfResponsibility/Foo.i"
	@echo "... ChainOfResponsibility/Foo.s"
	@echo "... ChainOfResponsibility/GenericFunctionNode.o"
	@echo "... ChainOfResponsibility/GenericFunctionNode.i"
	@echo "... ChainOfResponsibility/GenericFunctionNode.s"
	@echo "... Core/OperationManager.o"
	@echo "... Core/OperationManager.i"
	@echo "... Core/OperationManager.s"
	@echo "... Testing/ChainOfResponsibility/FunctionNodeTests.o"
	@echo "... Testing/ChainOfResponsibility/FunctionNodeTests.i"
	@echo "... Testing/ChainOfResponsibility/FunctionNodeTests.s"
	@echo "... Testing/ChainOfResponsibility/OperationManagerTests.o"
	@echo "... Testing/ChainOfResponsibility/OperationManagerTests.i"
	@echo "... Testing/ChainOfResponsibility/OperationManagerTests.s"
	@echo "... Testing/RunTests.o"
	@echo "... Testing/RunTests.i"
	@echo "... Testing/RunTests.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

