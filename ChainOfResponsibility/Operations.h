#pragma once      

class Operations {
public:
    // Available operations the ChainObjectHandler can execute, these operations will be passed along the chain of low-level objects
    enum op {
        allocateMemory,
        deallocateMemory,
        restoreToDefault,
        copyToGPU,
        copyFromGPU
    };
};
