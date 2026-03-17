# GPU Architecture Levels
Originally, Graphitti used CUDA Compute Capability (CC 3.5). This later changed to use a flexible architecture model, such that Graphitti can now be compiled for modern hardware (such as Volta or Lovelace). This allows the simulator to leverage modern GPU features while maintaining backwards compatibility for older cards by using conditional compilation.

# Supported Architectures

| Architecture | Compute Capability | Project Build Compatibility |
| :--- | :--- | :--- |
| **Kepler** | 3.5 / 3.7 | **Baseline**: Minimum version for backwards compatibility. |
| **Volta** | 7.0 | **Target**: Primary architecture for high-performance server runs. |
| **Ampere** | 8.0 / 8.6 | **Development**: Common for modern local development. |
| **Ada Lovelace**| 8.9 | **Current**: Latest generation available in the lab. |

## Compute Capability
* Binary Compatibility (cubin): Strictly backwards compatible. A binary for raiju (3.7) runs on ghidorah (8.9), but not vice versa.
- Backwards Compatibility: Any code compiled on an older architecture will work on newer ones.

- Forwards Compatibility (PTX): Any code compiled for a specific architecture will require that or a newer one to run.
    - Parallel Thread Execution (PTX): PTX is included in lab builds.

# Specifying Target Architecture

By default, if not user-specified, TARGET_ARCH to set to `"native"` 
which auto-detects and utilizes local hardware.

Legacy Support: -DTARGET_ARCH=35

Otachi Server: -DTARGET_ARCH=70

# Performance Notes
## Lab Servers Reference Table
Info retrieved from `nvidia-smi` (in terminal).

| Lab Server | GPU Model | Architecture | Compute Capability | Recommended `TARGET_ARCH` |
| :--- | :--- | :--- | :--- | :--- |
| **raiju** | Tesla K80 | **Kepler** | 3.7 | `37` |
| **otachi** | Tesla V100-PCIE-16GB | **Volta** | 7.0 | `70` |
| **ghidorah** | RTX 4500 Ada Generation | **Ada Lovelace** | 8.9 | `89` |

Note: Per project guidelines, conditional compilation adds structural complexity (code cruft). We only implement architecture-specific paths if they produce a measurable benefit.

- Example: Run a 5-10 minute simulation and then measure performance `nvidia-smi` and compare it to baseline results.
