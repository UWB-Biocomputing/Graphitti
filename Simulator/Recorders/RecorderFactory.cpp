/**
 * @file RecorderFactory.cpp
 *
 * @ingroup Simulator/Recorders
 * 
 * @brief A factory class for creating Recorder objects.
 */

#include "RecorderFactory.h"

#include "Hdf5GrowthRecorder.h"
#include "Xml911Recorder.h"
#include "XmlGrowthRecorder.h"
#include "XmlSTDPRecorder.h"

/// Constructor is private to keep a singleton instance of this class.
RecorderFactory::RecorderFactory() {
	// register recorder classes
	registerClass("XmlRecorder", &XmlRecorder::Create);
	registerClass("XmlGrowthRecorder", &XmlGrowthRecorder::Create);
	registerClass("XmlSTDPRecorder", &XmlSTDPRecorder::Create);
	registerClass("Xml911Recorder", &Xml911Recorder::Create);

#if defined(HDF5)
   registerClass("Hdf5Recorder", &Hdf5Recorder::Create);
   registerClass("Hdf5GrowthRecorder", &Hdf5GrowthRecorder::Create);
#endif
}

RecorderFactory::~RecorderFactory() { createFunctions.clear(); }

///  Register recorder class and its creation function to the factory.
///
///  @param  className  vertices class name.
///  @param  Pointer to the class creation function.
void RecorderFactory::registerClass(const std::string& className, CreateFunction function) {
	createFunctions[className] = function;
}


/// Creates concrete instance of the desired recorder class.
std::shared_ptr<IRecorder> RecorderFactory::createRecorder(const std::string& className) {
	recorderInstance = std::shared_ptr<IRecorder>(invokeCreateFunction(className));
	return recorderInstance;
}

/// Create an instance of the vertices class using the static ::Create() method.
///
/// The calling method uses this retrieval mechanism in
/// value assignment.
IRecorder* RecorderFactory::invokeCreateFunction(const std::string& className) {
	for (auto i = createFunctions.begin(); i != createFunctions.end(); ++i) if (className == i->first) return i->
		second();
	return nullptr;
}
