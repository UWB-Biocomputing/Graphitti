/**
 * @file SInputRegular.cpp
 *
 * @ingroup Simulator/Utils/Inputs
 * 
 * @brief A class that performs stimulus input (implementation Regular).
 */

#include "SInputRegular.h"
#include "tinyxml.h"

#define TIXML_USE_STL

void getValueList(const string &valString, vector<BGFLOAT> *pList);

/// constructor
///
/// @param[in] psi       Pointer to the simulation information
/// @param[in] parms     Pointer to xml parms element
SInputRegular::SInputRegular(TiXmlElement *parms) : values(nullptr)
{
   fSInput = false;

   // read duration, interval and sync
   TiXmlElement *temp = nullptr;
   string sync;
   if ((temp = parms->FirstChildElement("IntParams")) != nullptr) {
      if (temp->QueryFLOATAttribute("duration", &duration) != TIXML_SUCCESS) {
         cerr << "error IntParams:duration" << endl;
         return;
      }
      if (temp->QueryFLOATAttribute("interval", &interval) != TIXML_SUCCESS) {
         cerr << "error IntParams:interval" << endl;
         return;
      }
      if (temp->QueryValueAttribute("sync", &sync) != TIXML_SUCCESS) {
         cerr << "error IntParams:sync" << endl;
         return;
      }
   } else {
      cerr << "missing IntParams" << endl;
      return;
   }

   // initialize duration ,interval and cycle
   nStepsDuration = static_cast<int>(duration / Simulator::getInstance().getDeltaT() + 0.5);
   nStepsInterval = static_cast<int>(interval / Simulator::getInstance().getDeltaT() + 0.5);
   nStepsCycle = nStepsDuration + nStepsInterval;
   nStepsInCycle = 0;

   // read initial values
   if ((temp = parms->FirstChildElement("Values")) != nullptr) {
      TiXmlNode *pNode = nullptr;
      while ((pNode = temp->IterateChildren(pNode)) != nullptr) {
         if (strcmp(pNode->Value(), "I") == 0) {
            getValueList(pNode->ToElement()->GetText(), &initValues);
         } else {
            cerr << "error I" << endl;
            return;
         }
      }
   } else {
      cerr << "missing Values" << endl;
      return;
   }

   // we assume that initial values are in 10x10 matrix
   assert(initValues.size() == 100);

   // allocate memory for input values
   values.resize(Simulator::getInstance().getTotalVertices());

   // initialize values
   for (int i = 0; i < Simulator::getInstance().getHeight(); i++)
      for (int j = 0; j < Simulator::getInstance().getWidth(); j++)
         values[i * Simulator::getInstance().getWidth() + j] = initValues[(i % 10) * 10 + j % 10];

   initValues.clear();

   // allocate memory for shift values
   nShiftValues.resize(Simulator::getInstance().getTotalVertices());

   // initialize shift values
   memset(nShiftValues.data(), 0, sizeof(int) * Simulator::getInstance().getTotalVertices());

   if (sync == "no") {
      // asynchronous stimuli - fill nShiftValues array with values between 0 - nStepsCycle
      for (int i = 0; i < Simulator::getInstance().getHeight(); i++)
         for (int j = 0; j < Simulator::getInstance().getWidth(); j++)
            nShiftValues[i * Simulator::getInstance().getWidth() + j]
               = static_cast<int>(initRNG.inRange(0, nStepsCycle - 1));
   } else if (sync == "wave") {
      // wave stimuli - moving wave from left to right
      for (int i = 0; i < Simulator::getInstance().getHeight(); i++)
         for (int j = 0; j < Simulator::getInstance().getWidth(); j++)
            nShiftValues[i * Simulator::getInstance().getWidth() + j]
               = static_cast<int>((nStepsCycle / Simulator::getInstance().getWidth()) * j);
   }

   fSInput = true;
}

/// Initialize data.
///
/// @param[in] psi       Pointer to the simulation information.
void SInputRegular::init()
{
}

/// Terminate process.
///
/// @param[in] psi                Pointer to the simulation information.
void SInputRegular::term()
{
}

/// Helper function for input vaue list (copied from Core.cpp and modified for BGFLOAT)
void getValueList(const string &valString, vector<BGFLOAT> *pList)
{
   std::istringstream valStream(valString);
   BGFLOAT i;

   // Parse integers out of the string and add them to a list
   while (valStream.good()) {
      valStream >> i;
      pList->push_back(i);
   }
}
