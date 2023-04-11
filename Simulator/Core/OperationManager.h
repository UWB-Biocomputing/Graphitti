/**
 * @file OperationManager.h
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

#pragma once

#include "IFunctionNode.h"
#include "Operations.h"
#include <functional>
#include <list>
#include <log4cplus/loggingmacros.h>
#include <memory>

using namespace std;

class OperationManager {
public:
   /// Get Instance method that returns a reference to this object.
   static OperationManager &getInstance();

   /// Destructor
   ~OperationManager() = default;

   /// Called by lower level classes constructors on creation to register their operations with their operation type.
   /// This method can be overloaded to handle different function signatures.
   /// Handles function signature: void ()
   void registerOperation(const Operations::op &operation, const function<void()> &function);

   /// Takes in a operation type and invokes all registered functions that are classified as that operation type.
   void executeOperation(const Operations::op &operation) const;

   /// Takes in the operation enum and returns the enum as a string. Used for debugging purposes.
   string operationToString(const Operations::op &operation) const;

   /// Delete copy and move methods to avoid copy instances of the singleton
   OperationManager(const OperationManager &operationManager) = delete;
   OperationManager &operator=(const OperationManager &operationManager) = delete;

   OperationManager(OperationManager &&operationManager) = delete;
   OperationManager &operator=(OperationManager &&operationManager) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   OperationManager()
   {
      // Set logger_ to a reference to the rootLogger
      logger_ = (log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file")));
   }

   /// List of functions based containing a function and it's operation type.
   list<unique_ptr<IFunctionNode>> functionList_;

   /// Logger for log4plus
   log4cplus::Logger logger_;
};
