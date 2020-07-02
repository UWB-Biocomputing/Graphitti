//
// Created by chris on 6/26/2020.
//

#pragma once

/**
 * This class is public reference to the operation types the OperationManager can register and execute.
 */

class Operations {
public:
    // Available operations the OperationManager can register and execute
    enum op {
        allocateMemory,
        deallocateMemory,
        restoreToDefault,
        copyToGPU,
        copyFromGPU
    };
};

