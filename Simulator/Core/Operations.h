/**
 * @file Operations.h
 *
 * @brief This class is public reference to the operation types that the OperationManager can register and execute.
 * Stores operation types as an enum.
 *
 * @ingroup Simulator/Core
 */

#pragma once

class Operations {
public:
    /// Available operations the OperationManager can register and execute.
    enum op {
       printParameters,
       loadParameters,
       serialize,
       deserialize,
       deallocateGPUMemory, // Make sure deallocate memory isn't called until all GPU memory is copied back.
       restoreToDefault, // Not sure what this refers to.
       copyToGPU,
       copyFromGPU
    };
};