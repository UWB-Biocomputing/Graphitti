# ToDo: Once Hdf5 is in project use following line instead
# Get all header and source files from current directory
#file(GLOB Recorders *.cpp *.h)

file(GLOB Recorders
        IRecorder.h
        XmlRecorder.h
        XmlRecorder.cpp
        XmlGrowthRecorder.h
        XmlGrowthRecorder.cpp
        RecorderFactory.h
        RecorderFactory.cpp)

# Create a library from the list of files gathered
add_library(Recorders ${Recorders})