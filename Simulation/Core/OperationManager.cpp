/**
 * @file OperationManager.cpp
 *
 * @brief Singleton instance method that registers and executes functions based on operation types.
 *
 * @ingroup Core
 *
 * This class allows high level classes to perform high level operations that are defined by lower level classes.
 * Implementation of chain of responsibility design pattern.
 *
 * The implementation allows for multi-threaded use.
 */

#include "OperationManager.h"

#include <memory>
#include <list>

#include "Simulation/Core/FunctionNodes/GenericFunctionNode.h"

/// Get Instance method that returns a reference to this object.
OperationManager &OperationManager::getInstance() {
    static OperationManager instance;
    return instance;
}

/// Called by lower level classes constructors on creation to register their operations with their operation type
/// This method can be overloaded to handle different function signatures.
/// Handles function signature: void ()
void OperationManager::registerOperation(const Operations::op &operation, function<void()> function) {
    try {
        functionList_.push_back(unique_ptr<IFunctionNode>(new GenericFunctionNode(operation, function)));
    }
    catch (exception e) {
        throw runtime_error(string(e.what()) + " in OperationManager::registerOperation");
    }
}

/// Takes in a operation type and invokes all registered functions that are classified as that operation type.
void OperationManager::executeOperation(const Operations::op &operation) const {
    if (functionList_.size() > 0) {
        for (auto i = functionList_.begin(); i != functionList_.end(); ++i) {
            (*i)->invokeFunction(operation);
        }
    }
}

