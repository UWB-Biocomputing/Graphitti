//
// Created by chris on 6/26/2020.
//

#ifndef SUMMEROFBRAIN_OPERATION_H
#define SUMMEROFBRAIN_OPERATION_H


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


#endif //SUMMEROFBRAIN_OPERATION_H
