/**
 * @file Operations.h
 *
 * @brief This class is public reference to the operation types that the OperationManager can register and execute.
 * Stores operation types as an enum.
 *
 * @ingroup Core
 */

#pragma once

class Operations {
public:
    /// Available operations the OperationManager can register and execute.
    enum op {
        allocateMemory,
        deallocateMemory,
        restoreToDefault,
        copyToGPU,
        copyFromGPU
    };
};

