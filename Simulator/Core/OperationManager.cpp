/**
 * @file OperationManager.cpp
 *
 * @brief Singleton instance method that registers and executes functions based on operation types.
 *
 * @ingroup Simulator/Core
 *
 * This class allows high level classes to perform high level operations that are defined by lower level classes.
 * Implementation of chain of responsibility design pattern.
 *
 * The implementation allows for multi-threaded use.
 */

#include "OperationManager.h"

#include <memory>
#include <list>
#include <string>

#include "GenericFunctionNode.h"

/// Get Instance method that returns a reference to this object.
OperationManager &OperationManager::getInstance() {
   static OperationManager instance;
   return instance;
}

/// Destructor.
OperationManager::~OperationManager() {}

/// Called by lower level classes constructors on creation to register their operations with their operation type.
/// This method can be overloaded to handle different function signatures.
/// Handles function signature: void ()
void OperationManager::registerOperation(const Operations::op &operation, const function<void()> &function) {
   try {
      functionList_.push_back(unique_ptr<IFunctionNode>(new GenericFunctionNode(operation, function)));
   }
   catch (exception e) {
      LOG4CPLUS_FATAL(logger_, string(e.what()) + ". Push back failed in OperationManager::registerOperation");
      throw runtime_error(string(e.what()) + " in OperationManager::registerOperation");
   }
}

/// Takes in a operation type and invokes all registered functions that are classified as that operation type.
void OperationManager::executeOperation(const Operations::op &operation) const {
   LOG4CPLUS_INFO(logger_, "Executing operation " + operationToString(operation));
   if (functionList_.size() > 0) {
      for (auto i = functionList_.begin(); i != functionList_.end(); ++i) {
         (*i)->invokeFunction(operation);
      }
   }
}

/// Takes in the operation enum and returns the enum as a string. Used for debugging purposes.
string OperationManager::operationToString(const Operations::op &operation) const {
   switch (operation) {
      case Operations::op::printParameters:
         return "printParameters";
      case Operations::op::loadParameters:
         return "loadParameters";
      case Operations::op::serialize:
         return "serialize";
      case Operations::op::deserialize:
         return "deserialize";
      case Operations::op::deallocateGPUMemory:
         return "deallocateGPUMemory";
      case Operations::op::restoreToDefault:
         return "restoreToDefault";
      case Operations::op::copyToGPU:
         return "copyToGPU";
      case Operations::op::copyFromGPU:
         return "copyFromGPU";
      default:
         return "Operation isn't in OperationManager::operationToString()";
   }
}
