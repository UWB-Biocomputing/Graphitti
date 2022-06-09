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
RecorderFactory::RecorderFactory()
{
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

RecorderFactory::~RecorderFactory()
{
   createFunctions.clear();
}

///  Register recorder class and its creation function to the factory.
///
///  @param  className  recorder class name.
///  @param  function Pointer to the class creation function.
void RecorderFactory::registerClass(const string &className, CreateFunction function)
{
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired recorder class.
///
/// @param className Recorder class name.
/// @return Shared pointer to recorder instance if className is found in
///         createFunctions map, nullptr otherwise.
shared_ptr<IRecorder> RecorderFactory::createRecorder(const string &className)
{
   auto createRecorderIter = createFunctions.find(className);
   if (createRecorderIter != createFunctions.end()) {
      recorderInstance = shared_ptr<IRecorder>(createRecorderIter->second());
   } else {
      recorderInstance = nullptr;
   }

   return recorderInstance;
}
