## Build Instructions

###### This folder is where all the build files are generated so it doesn't clutter the source directories with generated files.


To build the simulation enter this directory and input the following command. 

`cmake ..`

This will generate a Makefile using the instructions in the CMakeLists.txt. Once the Makefile is created input the following  command.

`make`

This will create the executables `braingrid` and `tests`. `braingrid` is the simulator and `tests` are the unit tests.
To run the simulation input the following command with a path to a configuration file.

`./braingrid -c <configfilepath>` 

such as 

`./braingrid -c ../configfiles/test-tiny.xml`

To run tests input the following command.

`./tests`


