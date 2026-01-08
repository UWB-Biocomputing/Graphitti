# AsyncPhilox_d Class

## Overview

`AsyncPhilox_d` is a GPU-based random number generator class that uses NVIDIA's [CURAND](https://docs.nvidia.com/cuda/curand/index.html) library and the **Philox** counter-based RNG engine. It is designed for high-throughput simulations and supports asynchronous random number generation via an internal CUDA stream.

The class provides a **double-buffered**, asynchronous mechanism to produce random floating-point numbers (normally distributed) directly on the GPU. This enables overlapping random number generation with compute or memory transfer tasks in other streams.

---

## Purpose

`AsyncPhilox_d` enables:

- **Per-thread RNG state initialization** on the GPU
- **Segmented random number generation** using `curand_normal`
- **Double-buffering** for non-blocking buffer filling
- **Asynchronous execution** using a CUDA stream created and managed internally

This design improves parallelism and simulation throughput by decoupling RNG generation from synchronous host and device execution.

---

## CURAND Philox Generator

Philox is a **counter-based** RNG suitable for parallel applications. It is:

- **Stateless** across launches (state only encodes seed, counter, and thread ID)
- **Efficient** on GPUs due to its low register and instruction overhead
- **Deterministic**, producing reproducible sequences across threads

NVIDIA’s CURAND provides the Philox generator via the `curandStatePhilox4_32_10_t` type, which is initialized on a per-thread basis using `curand_init`.

You can find more details in the [CURAND Device API Overview](https://docs.nvidia.com/cuda/curand/device-api-overview.html).


## CUDA Documentation Links

- [CURAND API Reference](https://docs.nvidia.com/cuda/curand/index.html)
- [Philox Generator in CURAND](https://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-3)
