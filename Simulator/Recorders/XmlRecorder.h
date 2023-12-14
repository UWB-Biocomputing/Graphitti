/**
 * @file XmlRecorder.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief Provides an implementation for recording Graph-based simulation
 *           variable history in an XML file.
 *
 * The XmlRecorder class facilitates the recording of various graph-based simulation 
 * variable information, including :
 *     -# the recorded variable name.
 *     -# the recorded variable basic data type.
 *     -# the recorded variable address.
 * 
 * Use case: Neuron network simulation information: neuron's layout, spikes history,
 * and compile history information on xml file:
 *     -# the neuron ID.
 *     -# time steps of events produced by each neuron.
 *     -# the history events information of all neurons.
 */

#pragma once
#include "Global.h"
#include "Model.h"
#include "Recorder.h"

/// a list of basic data types in different recorded variables
using multipleTypes = variant<uint64_t, double, string>;

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

   // TODO : remove it ?
   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() override;

   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() override;

   // TODO : remove it ?
   /// Get the current radii and rates vlaues
   virtual void getValues() override;

   /// Terminate process
   virtual void term() override;

   // TODO: No parameters needed (AllVertices &vertices)
   /// Compile/capture variable history information in every epoch
   virtual void compileHistories(AllVertices &vertices) override;

   // TODO: No parameters needed (AllVertices &vertices)
   /// Writes simulation results to an output destination.
   virtual void saveSimData(const AllVertices &vertices) override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override;

   /// Register a single instance of a class derived from RecordableBase.
   /// It stores the address of the registered variable and the related information
   /// of this recorded variable
   virtual void registerVariable(const string &varName, RecordableBase &recordVar) override;

   /// register a vector of instance of a class derived from RecordableBase.
   virtual void registerVariable(const string &varName,
                                 vector<RecordableBase *> &recordVars) override;

   ///@{
   /** These methods are intended only for unit tests */
   /// constructor only for unit test
   XmlRecorder(const string &fileName)
   {
      resultFileName_ = fileName;
   }

   /// Accessor method for variable name (only included during unit tests)
   /// @param numIndex   The index number in the variable list.
   const string &getVariableName(int numIndex) const
   {
      return variableTable_[numIndex].variableName_;
   }

   /// Accessor method for variable data type (only included during unit tests)
   /// @param numIndex   The index number in the variable list.
   const string &getDataType(int numIndex) const
   {
      return variableTable_[numIndex].dataType_;
   }

   /// Accessor method for a single variable address in the variableTable_
   /// @param numIndex   The index number in the variable list.
   /// (only included during unit tests)
   RecordableBase &getSingleVariable(int numIndex)
   {
      return variableTable_[numIndex].variableLocation_;
   }

   /// Accessor method for variablesHistory_ (only included during unit tests)
   vector<multipleTypes> &getHistory(int numIndex)
   {
      return (variableTable_[numIndex].variableHistory_);
   }

   /// get an output stream from toXml method
   string getToXML(const string &name, vector<multipleTypes> &singleVariableBuffer_,
                   const string &basicType)
   {
      string outputXML;
      outputXML = toXML(name, singleVariableBuffer_, basicType);
      return outputXML;
   }
   ///@}

protected:
   /// Represents information about a single recorded variable.
   /// The singleVariableInfo struct encapsulates details about a recorded variable, including its name,
   ///     basic data type, address (location), and the history of accumulated values over time.
   struct singleVariableInfo {
      /// the name of each variable
      string variableName_;

      /// the basic data type in the Recorded variable
      string dataType_;

      /// a reference to a RecordableBase variable
      /// As the simulator runs, the values in the RecordableBase object will be updated
      RecordableBase &variableLocation_;

      /// the history of accumulated values for a registered RecordableBase object variable
      vector<multipleTypes> variableHistory_;

      /// Constructor accepting the variable name and the address of recorded variable
      singleVariableInfo(const string &name, RecordableBase &location) :
         variableLocation_(location), variableName_(name)
      {
         dataType_ = location.getDataType();
      }
   };

   /// Represents a list of registered variables for recording.
   /// The variableTable_ vector stores information about all the variables
   ///      that need to be recorded, including their names, basic data types,
   ///      addresses (locations), and the history of accumulated values.
   vector<singleVariableInfo> variableTable_;

   /// a file stream for xml output
   ofstream resultOut_;

   /// string toXML(string name,  vector<multipleTypesuint64_t>const;
   string toXML(const string &name, vector<multipleTypes> &singleVariableBuffer_,
                const string &basicType) const;

   // TODO: this method will be deleted
   void getStarterNeuronMatrix(VectorMatrix &matrix, const vector<bool> &starterMap);
};