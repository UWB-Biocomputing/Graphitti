/**
 * @file XmlRecorder.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief An implementation for recording spikes history on xml file
 *
 * The XmlRecorder provides a mechanism for recording neuron's layout, spikes history,
 * and compile history information on xml file:
 *     -# the neuron ID.
 *     -# time steps of events produced by each neuron,
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
   // @param numIndex   The index number in the variable list.
   std::string getNeuronName(int numIndex) const
   {
      return variableTable_[numIndex].variableName_;
   }

   // Getter method for a single variable address in the variableTable_
   // @param numIndex   The index number in the variable list.
   // (only included during unit tests)
   EventBuffer &getSingleNeuronEvents(int numIndex) const
   {
      return *(variableTable_[numIndex].variableLocation_);
   }

   // Getter method for neuronsHistory_ (only included during unit tests)
   std::vector<vector<uint64_t>> getHistory() const
   {
      return neuronsHistory_;
   }
   ///@}

protected:
   // create a struct contains a variable information
   struct variableInfo {
      // records the name of each variable
      string variableName_;

      // This pointer stores the address of the registered variable
      // As the simulator runs, the values will be updated
      // It can records all events of a single neuron in each epoch
      shared_ptr<EventBuffer> variableLocation_;

      //constructor
      variableInfo(string name, EventBuffer &location)
      {
         variableName_ = name;
         variableLocation_ = std::shared_ptr<EventBuffer>(&location, [](EventBuffer *) {
         });
      }
   };

   // A list of variables
   // the variableTable_ stores all the variables information that need to be recorded
   std::vector<variableInfo> variableTable_;

   // history of accumulated event for all neurons
   std::vector<vector<uint64_t>> neuronsHistory_;

   // a file stream for xml output
   ofstream resultOut_;

   string toXML(string name, vector<uint64_t> singleNeuronBuffer_) const;

   /// this method will be deleted
   void getStarterNeuronMatrix(VectorMatrix &matrix, const std::vector<bool> &starterMap);
};