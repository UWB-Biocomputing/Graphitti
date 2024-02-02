## Build Instructions

###### This folder is where all the build files are generated so it doesn't clutter the source directories with generated files.


To build the simulation enter this 'build' directory and input the following command. 

`cmake ..` 

or if it is the first time building and you are using the Raiju virtual machine. This sets the compiler GCC compiler to the working version.

`cmake -D CMAKE_CXX_COMPILER=/opt/rh/devtoolset-8/root/usr/bin/g++ ..`

NOTE: If you're making a clean build (No CMakeCache.txt) of the GPU version, you must do the `cmake ..' command a second time.

This will generate a Makefile using the instructions in the CMakeLists.txt. Once the Makefile is created input the following  command.

`make -j`

This will create the executables `graphitti` and `tests`. `graphitti` is the simulator and `tests` are the unit tests.
To run the simulation input the following command with a path to a configuration file.

`./graphitti -c <configfilepath>` 

such as 

`./graphitti -c ../configfiles/test-tiny.xml`

To run tests input the following command.

`./tests`


