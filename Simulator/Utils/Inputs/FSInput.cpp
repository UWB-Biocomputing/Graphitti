/**
 * @file FSInput.cpp
 * 
 * @ingroup Simulator/Utils/Inputs
 * 
 * @brief A factoy class that creates an instance of stimulus input object.
 */

#include "FSInput.h"
#include "HostSInputPoisson.h"
#include "HostSInputRegular.h"
#include "Simulator.h"
#if defined(USE_GPU)
   #include "GpuSInputPoisson.h"
   #include "GpuSInputRegular.h"
#endif
#include <tinyxml.h>

/// Create an instance of the stimulus input class based on the method
/// specified in the stimulus input file.
///
/// @param[in] psi                   Pointer to the simulation information
/// @return a pointer to a SInput object
ISInput *FSInput::CreateInstance()
{
   auto stimulusFileName = Simulator::getInstance().getStimulusFileName();
   if (stimulusFileName.empty()) {
      return nullptr;
   }
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   // load stimulus input file
   TiXmlDocument siDoc(stimulusFileName.c_str());
   if (!siDoc.LoadFile()) {
      LOG4CPLUS_ERROR(consoleLogger, ("Failed loading stimulus input file " << stimulusFileName << ":" << "\n\t"
           << siDoc.ErrorDesc() << endl));
      LOG4CPLUS_ERROR(consoleLogger, (" error: " << siDoc.ErrorRow() << ", " << siDoc.ErrorCol() << endl));
      return nullptr;
   }

   // load input parameters
   TiXmlElement *parms = nullptr;
   if ((parms = siDoc.FirstChildElement("InputParams")) == nullptr) {
      LOG4CPLUS_ERROR(consoleLogger, ("Could not find <InputParms> in stimulus input file " << stimulusFileName << endl));
      return nullptr;
   }

   // read input method
   TiXmlElement *temp = nullptr;
   string name;
   if ((temp = parms->FirstChildElement("IMethod")) != nullptr) {
      if (temp->QueryValueAttribute("name", &name) != TIXML_SUCCESS) {
         LOG4CPLUS_ERROR(consoleLogger, ("error IMethod:name" << endl));
         return nullptr;
      }
   } else {
      LOG4CPLUS_ERROR(consoleLogger, ("missing IMethod" << endl));
      return nullptr;
   }

   // create an instance
   ISInput *pInput = nullptr;   // pointer to a stimulus input object

   if (name == "SInputRegular") {
#if defined(USE_GPU)
      pInput = new GpuSInputRegular(parms);
#else
      pInput = new HostSInputRegular(parms);
#endif
   } else if (name == "SInputPoisson") {
#if defined(USE_GPU)
      pInput = new GpuSInputPoisson(parms);
#else
      pInput = new HostSInputPoisson(parms);
#endif
   } else {
      LOG4CPLUS_ERROR(consoleLogger, ("unsupported stimulus input method" << endl));
   }

   return pInput;
}
