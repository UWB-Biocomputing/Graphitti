/**
 * @file XmlRecorder.h
 *
 * @ingroup Simulator/Recorders
 *
 * @brief Provides an implementation for recording Graph-based simulation
 *        variable history in an XML file. It can be reused to:
 *        1) Change variables to record in an existing model.
 *        2) Record variables for new integrated network simulations.
 * Supports updated types: Constant and Dynamic.
 * Supports recording 1-D variables whose base class is RecordableBase.
 *     - EventBuffer
 *     - VectorMatrix
 *     - RecordableVector for standard library vector
 *
 * The XmlRecorder class receives a list of registered variable entities passed from 
 * the variable owner classes and stores them into a variable table.
 * Each entity in this table includes:
 *     - The recorded variable name.
 *     - The recorded variable basic data type.
 *     - The recorded variable address.
 *     - The vector container to store the data.
 * 
 * Use case: Neural network simulation
 * Record neural network simulation variables' information in an XML file:
 *     - The neuron ID.
 *     - Time steps of events produced by each neuron.
 */

#pragma once
#include "Global.h"
#include "Model.h"
#include "Recorder.h"

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

   // TODO : remove it if the HDF5 Recorder class doesn't need this method
   /// Init radii and rates history matrices with default values
   virtual void initDefaultValues() override;

   // TODO : remove it if the HDF5 Recorder class doesn't need this method
   /// Init radii and rates history matrices with current radii and rates
   virtual void initValues() override;

   // TODO : remove it if the HDF5 Recorder class doesn't need this method
   /// Get the current radii and rates vlaues
   virtual void getValues() override;

   /// Terminate process
   virtual void term() override;


   /// Compile/capture variable history information in every epoch
   virtual void compileHistories() override;


   /// Writes simulation results to an output destination.
   virtual void saveSimData() override;

   ///  Prints out all parameters to logging file.
   ///  Registered to OperationManager as Operation::printParameters
   virtual void printParameters() override;

   /// Receives a recorded variable entity from the variable owner class
   /// Used when the return type from recordable variable is supported by Recorder
   /**
   * @brief Registers a single instance of a class derived from RecordableBase.
   * @param varName Name of the recorded variable.
   * @param recordVar Reference to the recorded variable.
   * @param variableType Updated frequency of the recorded variable.
   */
   virtual void registerVariable(const string &varName, RecordableBase &recordVar,
                                 UpdatedType variableType) override;

   /// Register a vector of instance of a class derived from RecordableBase.
   virtual void registerVariable(const string &varName, vector<RecordableBase *> &recordVars,
                                 UpdatedType variableType) override;

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

   /// get an output stream from toXml method (only included during unit tests)
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
   /// The singleVariableInfo struct encapsulates details about a recorded variable,
   /// including its name, basic data type, address (location),
   /// updated type and the history of accumulated values over time.
   struct singleVariableInfo {
      /// the name of each variable
      string variableName_;

      /// the basic data type in the Recorded variable
      string dataType_;

      /// the variable type: updated frequency
      UpdatedType variableType_;

      /// a reference to a RecordableBase variable
      /// As the simulator runs, the values in the RecordableBase object will be updated
      RecordableBase &variableLocation_;

      /// the history of accumulated values for a registered RecordableBase object variable
      vector<multipleTypes> variableHistory_;

      /// Constructor accepting the variable name, the address of recorded variable, the updated type
      singleVariableInfo(const string &name, RecordableBase &location, UpdatedType variableType) :
         variableLocation_(location), variableName_(name), variableType_(variableType)
      {
         dataType_ = location.getDataType();
      }

      /// @brief capture value to the vector in the table
      void captureData()
      {
         if (variableLocation_.getNumElements() > 0) {
            variableHistory_.reserve(variableHistory_.size() + variableLocation_.getNumElements());
            for (int index = 0; index < variableLocation_.getNumElements(); index++) {
               variableHistory_.push_back(variableLocation_.getElement(index));
            }
         }
      }
   };

   /// List of registered variables for recording
   vector<singleVariableInfo> variableTable_;

   /// a file stream for xml output
   ofstream resultOut_;

   /// Retrieves values of a vector of variant and outputs them to a xml file
   string toXML(const string &name, vector<multipleTypes> &singleVariableBuffer_,
                const string &basicType) const;

   // TODO: this method will be deleted if the HDF5Recorder doesn't need it
   void getStarterNeuronMatrix(VectorMatrix &matrix, const vector<bool> &starterMap);
};