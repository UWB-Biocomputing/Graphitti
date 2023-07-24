/**
 * @file XmlRecorder.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on xml file
 *
 * The XmlRecorder provides a mechanism for recording neuron's layout, spikes history,
 * and compile history information on xml file:
 *     -# the number of a single neuron,
 *     -# all vertex events for a single neuron.
 */

#pragma once
#include "Global.h"
#include "Model.h"
#include "Recorder.h"
#include <fstream>
#include <vector>

class XmlRecorder : public Recorder {
public:
   // constructor which opens the xml file to store results
   XmlRecorder();

   static Recorder *Create()
   {
      return new XmlRecorder();
   }

   /// Initialize data in the newly loadeded xml file
   virtual void init() override;

   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() override;

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() override;

   /// Get the current radii and rates vlaues
   virtual void getValues() override;

   /// Terminate process
   virtual void term() override;

   /// Compile history information in every epoch
   /// @param[in] vertices   The entire list of vertices.
   virtual void compileHistories(AllVertices &vertices) override;

   /// Writes simulation results to an output destination.
   /// @param  vertices the Vertex list to search from.
   virtual void saveSimData(const AllVertices &vertices) override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override;

   /// Store the neuron number and all the events of this single neuron
   void registerVariable(string varName, EventBuffer &recordVar) override;

   ///@{
   /** These methods are intended only for unit tests */
   // constructor only for unit test
   XmlRecorder(std::string fileName_)
   {
      resultFileName_ = fileName_;
   }

   // Getter method for neuronName_ (only included during unit tests)
   std::string getNeuronName() const
   {
      return neuronName_;
   }
   // Getter method for singleNeuronEvents_ (only included during unit tests)
   EventBuffer &getSingleNeuronEvents() const
   {
      return *singleNeuronEvents_;
   }
   // Getter method for singleNeuronHistory_ (only included during unit tests)
   std::vector<uint64_t> getHistory() const
   {
      return singleNeuronHistory_;
   }
   ///@}

protected:
   // variable neuronName_ records the number of a single neuron
   string neuronName_;

   // The address of the registered variable
   // As the simulator runs, the values will be updated
   // It can records all events of a single neuron in each epoch
   shared_ptr<EventBuffer> singleNeuronEvents_;

   // history of accumulated event for a single neuron
   std::vector<uint64_t> singleNeuronHistory_;

   // // create a structure contains the information of a variable
   // struct variableInfo{
   //    string variableName;
   //    EventBuffer* variableLocation;
   // };

   // // create table
   // std::vector<variableInfo> variableTable;


   // a file stream for xml output
   ofstream resultOut_;

   string toXML(string name, vector<uint64_t> singleNeuronBuffer_) const;

   //this method will be deleted
   void getStarterNeuronMatrix(VectorMatrix &matrix, const std::vector<bool> &starterMap);
};