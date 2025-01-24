/**
 * @file Operations.h
 *
 * @brief This class is public reference to the operation types that the OperationManager can register and execute.
 * Stores operation types as an enum.
 *
 * @ingroup Simulator/Core
 */

#pragma once

enum class Operations {
   /// Available operations the OperationManager can register and execute.
   printParameters,
   loadParameters,
   registerGraphProperties,
   setup,
   serialize,
   deserialize,
   deallocateGPUMemory,   // Make sure deallocate memory isn't called until all GPU memory is copied back.
   restoreToDefault,   // Not sure what this refers to.
   copyToGPU,
   copyFromGPU
};