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
#if defined(USE_GPU)
   #include "GpuSInputPoisson.h"
   #include "GpuSInputRegular.h"
#endif
#include <tinyxml.h>

/// constructor
FSInput::FSInput()
{
}

/// destructor
FSInput::~FSInput()
{
}

/// Create an instance of the stimulus input class based on the method
/// specified in the stimulus input file.
///
/// @param[in] psi                   Pointer to the simulation information
/// @return a pointer to a SInput object
ISInput *FSInput::CreateInstance()
{
   if (Simulator::getInstance().getStimulusFileName().empty()) {
      return nullptr;
   }

   // load stimulus input file
   TiXmlDocument siDoc(Simulator::getInstance().getStimulusFileName().c_str());
   if (!siDoc.LoadFile()) {
      cerr << "Failed loading stimulus input file "
           << Simulator::getInstance().getStimulusFileName() << ":"
           << "\n\t" << siDoc.ErrorDesc() << endl;
      cerr << " error: " << siDoc.ErrorRow() << ", " << siDoc.ErrorCol() << endl;
      return nullptr;
   }

   // load input parameters
   TiXmlElement *parms = nullptr;
   if ((parms = siDoc.FirstChildElement("InputParams")) == nullptr) {
      cerr << "Could not find <InputParms> in stimulus input file "
           << Simulator::getInstance().getStimulusFileName() << endl;
      return nullptr;
   }

   // read input method
   TiXmlElement *temp = nullptr;
   string name;
   if ((temp = parms->FirstChildElement("IMethod")) != nullptr) {
      if (temp->QueryValueAttribute("name", &name) != TIXML_SUCCESS) {
         cerr << "error IMethod:name" << endl;
         return nullptr;
      }
   } else {
      cerr << "missing IMethod" << endl;
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
      cerr << "unsupported stimulus input method" << endl;
   }

   return pInput;
}
