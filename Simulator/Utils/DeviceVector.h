/**
 * @file DeviceVector.h
 *
 * @ingroup Simulator/Utils
 *
 * @brief A Template class that provides a std::vector-like interface for both host and device (GPU) memory
 * @tparam T Type of elements (must be BGFLOAT, int, or bool)
 *
 * This class wraps a std::vector and provides additional operations for device memory
 * allocation, deallocation, and host-device data transfer. Currently, it only supports
 * primitive types, and since the codebase only uses BGFLOAT, int, and bool, these are
 * the only types explicitly allowed. Support for other primitive types (like char,
 * double, etc.) can be added if needed in the future. Note that
 * this class is not RAII compliant for device memory - it provides methods for manual
 * device memory management but leaves the responsibility of proper allocation,
 * synchronization, and deallocation to the user.
 *
 * The class provides convenient methods for common host-device operations but performs
 * only trivial safety checks. Users must ensure device memory operations are safe and
 * that host-device memory is properly synchronized using the provided methods. For example,
 * operations like copyToDevice() or copyToHost() will not implicitly allocate device
 * memory if it hasn't been allocated - they will throw an error instead. Similar behavior
 * applies to other device operations.
 *
 * Host-side operations are implemented by overloading std::vector operations and operate
 * independently of device memory - they do not automatically sync changes to the device.
 * For detailed behavior of these operations, refer to std::vector documentation. Future
 * needs for additional vector operations can be met by overloading the specific
 * methods or operators as needed.
 
 * Currently, DeviceVector only supports primitive types, and since the codebase only uses
 * BGFLOAT, int, and bool, these are the only types explicitly allowed. Support for other
 * primitive types (like char, double, etc.) can be added if needed in the future.
 *
 * Note: The current implementation prioritizes explicit control over convenience.
 * Future extensions could add RAII compliance, additional safety checks, and automatic
 * host-device synchronization if such behavior becomes desirable.
 */

#pragma once
#include "BGTypes.h"   // For BGFLOAT definition
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
   #include "Book.h"
#endif

#include <variant>

/// Define supported types for DeviceVector using variant
using DeviceVectorTypes = std::variant<BGFLOAT, int, bool>;

/**
 * Type trait system to verify if a type is supported by DeviceVector.
 * Works in two parts:
 *
 * 1. Primary template (base case):
 *    - Inherits from std::false_type, so by default any type is NOT supported
 *    - This catches all cases where Variant is not a std::variant
 */
template <typename T, typename Variant> struct is_device_vector_supported_type : std::false_type {};

/**
  * 2. Partial specialization:
  *    - Matches when the second template parameter is a std::variant
  *    - Uses fold expression (... ||) to check if T matches any type in the variant
  *    - std::is_same_v<T, Types> checks exact type match for each type in variant
  *    - Inherits from std::bool_constant which is true if any type matches
      Note: In C++20 there are better ways to do this using generic lambda with a template parameter list 
      with a combination of constexpr but nvcc doesn't support it yet.
  */
template <typename T, typename... Types>
struct is_device_vector_supported_type<T, std::variant<Types...>> :
   std::bool_constant<(std::is_same_v<T, Types> || ...)> {};

template <typename T> class DeviceVector {
   // Ensure T is one of the allowed types using variant check
   static_assert(is_device_vector_supported_type<T, DeviceVectorTypes>::value,
                 "Unsupported type for DeviceVector");

public:
   /// Reference type that handles std::vector<bool> specialization
   using reference =
      typename std::conditional_t<std::is_same_v<T, bool>, std::vector<bool>::reference, T &>;

   /// @brief Constructor that initializes the host vector without GPU allocation
   /// @param size Initial size of the host vector (defaults to 0)
   /// @post Host vector is created with 'size' default-constructed elements
   /// @post Device pointer is nullptr (no GPU memory allocated)
   explicit DeviceVector(size_t size = 0) : hostData_(size)
   {
#if defined(__CUDACC__)
      devicePtr_ = nullptr;
#endif
   }

   ~DeviceVector() = default;

   /// @brief Copy constructor is deleted to prevent accidental copies that could lead to GPU memory leaks
   DeviceVector(const DeviceVector &) = delete;

   /// @brief Copy assignment is deleted to prevent accidental copies that could lead to GPU memory leaks
   DeviceVector &operator=(const DeviceVector &) = delete;

   /// @brief Adds an element to the end of the host vector
   /// @param value The element to append
   /// @post New element is added at the end of host vector
   /// @note Only modifies host memory, device memory remains unchanged
   void push_back(const T &value)
   {
      hostData_.push_back(value);
   }

   /// @brief Resizes the host vector to contain new_size elements
   /// @param new_size New size of the vector
   /// @post Host vector contains exactly new_size elements
   /// @note Only affects host memory, device memory size remains unchanged
   void resize(size_t new_size)
   {
      hostData_.resize(new_size);
   }

   /// @brief Resizes the host vector to new_size elements, initializing new elements with value
   /// @param new_size New size of the vector
   /// @param value Value to initialize new elements with
   /// @post Host vector contains exactly new_size elements
   /// @post New elements (if any) are copies of value
   /// @note Only affects host memory, device memory size remains unchanged
   void resize(size_t new_size, const T &value)
   {
      hostData_.resize(new_size, value);
   }

   /// @brief Removes all elements from the host vector
   /// @post Host vector becomes empty (size = 0)
   /// @post Capacity remains unchanged
   void clear()
   {
      hostData_.clear();
   }

   /// @brief Reserves storage in the host vector for at least new_cap elements
   /// @param new_cap Minimum capacity to reserve
   /// @post Host vector capacity is at least new_cap
   /// @note No reallocation occurs until size exceeds new capacity
   /// @note Size remains unchanged
   void reserve(size_t new_cap)
   {
      hostData_.reserve(new_cap);
   }

   /// @brief Returns the number of elements in the host vector
   /// @return Current number of elements in host vector
   size_t size() const
   {
      return hostData_.size();
   }

   /// @brief Checks if the host vector contains no elements
   /// @return true if the host vector is empty, false otherwise
   bool empty() const
   {
      return hostData_.empty();
   }

   /// @brief Replaces host vector contents with n copies of value
   /// @param n Number of elements to assign
   /// @param value Value to fill the vector with
   /// @post Host vector contains exactly n elements
   /// @post All elements are copies of value
   void assign(size_t n, const T &value)
   {
      hostData_.assign(n, value);
   }

   /// @brief Gets a const reference to the underlying host vector
   /// @return Const reference to the host vector
   const std::vector<T> &getHostVector() const
   {
      return hostData_;
   }

   /// @brief Gets a copy of the host vector
   /// @return Copy of the host vector
   std::vector<T> getHostVector()
   {
      return hostData_;
   }

   /// @brief Implicit conversion to host vector reference
   /// @return Reference to the underlying host vector
   /// @note Allows using DeviceVector as a std::vector
   /// @warning Modifications affect host memory only
   operator std::vector<T> &()
   {
      return hostData_;
   }

   /// @brief Implicit conversion to const host vector reference
   /// @return Const reference to the underlying host vector
   /// @note Allows using const DeviceVector as a const std::vector
   /// @note Thread-safe for concurrent reads
   operator const std::vector<T> &() const
   {
      return hostData_;
   }

   /// @brief Array subscript operator for host vector access
   /// @param idx Index of the element to access
   /// @return Reference (or proxy for bool) to the element
   /// @note Using reference type alias to match std::vector<bool> behavior that allows bit-level manipulation
   /// @warning No bounds checking is performed
   reference operator[](size_t idx)
   {
      // For bool type, hostData_[idx] returns vector<bool>::reference
      // which our reference type alias handles correctly to match
      // std::vector<bool>'s space-efficient implementation
      return hostData_[idx];
   }

   /// @brief Const array subscript operator for host vector access
   /// @param idx Index of the element to access
   /// @return Copy of the element (actual bool for bool type)
   /// @note Special handling for bool type to return actual bool value
   /// @warning No bounds checking is performed
   const T operator[](size_t idx) const
   {
      if constexpr (std::is_same_v<T, bool>)
         return static_cast<bool>(hostData_[idx]);   // explicit casting to bool value from proxy

      return hostData_[idx];   // normal types
   }

   /// @brief Gets pointer to contiguous host memory array
   /// @return Pointer to the first element in host memory
   /// @note Returns nullptr if vector is empty
   T *data()
   {
      return hostData_.data();
   }

   /// @brief Gets const pointer to contiguous host memory array
   /// @return Const pointer to the first element in host memory
   /// @note Returns nullptr if vector is empty
   const T *data() const
   {
      return hostData_.data();
   }

   /// @brief Safe element access for host vector with bounds checking
   /// @param idx Index of the element to access
   /// @return Reference to the host vector element at specified index
   /// @note Performs bounds checking unlike operator[]
   /// @throws std::out_of_range if idx >= size()
   T &at(size_t idx)
   {
      return hostData_.at(idx);
   }

   /// @brief Safe const element access for host vector with bounds checking
   /// @param idx Index of the element to access
   /// @return Const reference to the host vector element at specified index
   /// @note Performs bounds checking unlike operator[]
   /// @throws std::out_of_range if idx >= size()
   const T &at(size_t idx) const
   {
      return hostData_.at(idx);
   }

   /// @brief Access first element of host vector
   /// @return Reference to the first element
   /// @pre Vector must not be empty
   /// @throws std::out_of_range if vector is empty
   T &front()
   {
      return hostData_.front();
   }

   /// @brief Access first element of host vector (const)
   /// @return Const reference to the first element
   /// @pre Vector must not be empty
   /// @throws std::out_of_range if vector is empty
   const T &front() const
   {
      return hostData_.front();
   }

   /// @brief Access last element of host vector
   /// @return Reference to the last element
   /// @pre Vector must not be empty
   /// @throws std::out_of_range if vector is empty
   T &back()
   {
      return hostData_.back();
   }

   /// @brief Access last element of host vector (const)
   /// @return Const reference to the last element
   /// @pre Vector must not be empty
   /// @throws std::out_of_range if vector is empty
   const T &back() const
   {
      return hostData_.back();
   }

   /// @brief Gets iterator to the first element of host vector
   /// @return Iterator to the beginning
   /// @note Enables range-based for loop support
   auto begin()
   {
      return hostData_.begin();
   }

   /// @brief Gets iterator to one past the last element
   /// @return Iterator to the end
   /// @note Required for range-based for loop support
   auto end()
   {
      return hostData_.end();
   }

   /// @brief Gets const iterator to the first element
   /// @return Const iterator to the beginning
   /// @warning Iterator only valid for host memory
   auto begin() const
   {
      return hostData_.begin();
   }

   /// @brief Gets const iterator to one past the last element
   /// @return Const iterator to the end
   /// @warning Iterator only valid for host memory
   auto end() const
   {
      return hostData_.end();
   }

#if defined(__CUDACC__)
public:
   /// @brief Gets pointer to device memory
   /// @return Raw pointer to device memory
   /// @note Returns nullptr if no device memory is allocated
   T *getDevicePointer()
   {
      return devicePtr_;
   }

   /// @brief Gets const pointer to device memory
   /// @return Const Raw pointer to device memory
   /// @note Returns nullptr if no device memory is allocated
   const T *getDevicePointer() const
   {
      return devicePtr_;
   }

   /// @brief Implicit conversion to device pointer for CUDA kernel calls
   /// @return Raw pointer to device memory
   /// @note Allows using DeviceVector directly in CUDA kernels
   /// @warning Returns nullptr if device memory is not allocated
   operator T *()
   {
      return devicePtr_;
   }

   /// @brief Implicit conversion to const device pointer for CUDA kernel calls
   /// @return Const pointer to device memory
   /// @note Allows using const DeviceVector directly in CUDA kernels
   /// @warning Returns nullptr if device memory is not allocated
   operator const T *() const
   {
      return devicePtr_;
   }

   /// @brief Allocates CUDA device memory for vector data
   /// @pre CUDA device is available and has sufficient memory
   /// @post Device memory is allocated with size() * sizeof(T) bytes
   /// @post Previous device memory (if any) is freed
   /// @note Memory is not initialized after allocation
   /// @throws CUDA errors on allocation failure
   void allocateDeviceMemory()
   {
      if (devicePtr_)
         freeDeviceMemory();
      HANDLE_ERROR(cudaMalloc(&devicePtr_, hostData_.size() * sizeof(T)));
   }

   /// @brief Frees CUDA device memory if allocated
   /// @post Device memory is deallocated
   /// @post Device pointer is set to nullptr
   /// @throws CUDA errors on deallocation failure
   void freeDeviceMemory()
   {
      if (devicePtr_) {
         HANDLE_ERROR(cudaFree(devicePtr_));
         devicePtr_ = nullptr;
      }
   }

   /// @brief Copies data from host to device memory
   /// @pre Device memory must be allocated
   /// @post Device memory contains exact copy of host data
   /// @note Special handling for bool type using intermediate array
   /// @throws std::runtime_error if device memory not allocated
   /// @throws CUDA errors on copy failure
   void copyToDevice()
   {
      if (!devicePtr_)
         throw std::runtime_error("Device memory not allocated. Call allocateDeviceMemory()");

      if constexpr (std::is_same_v<T, bool>) {
         const size_t n = hostData_.size();
         bool raw_data[n];
         for (size_t i = 0; i < n; ++i) {
            raw_data[i] = hostData_[i];
         }
         HANDLE_ERROR(cudaMemcpy(devicePtr_, raw_data, n * sizeof(bool), cudaMemcpyHostToDevice));
      } else {
         HANDLE_ERROR(cudaMemcpy(devicePtr_, hostData_.data(), hostData_.size() * sizeof(T),
                                 cudaMemcpyHostToDevice));
      }
   }

   /// @brief Copies data from device to host memory
   /// @pre Device memory must be allocated
   /// @post Host memory contains exact copy of device data
   /// @note Special handling for bool type using intermediate array
   /// @throws std::runtime_error if device memory not allocated
   /// @throws CUDA errors on copy failure
   void copyToHost()
   {
      if (!devicePtr_)
         throw std::runtime_error("Device memory not allocated.");

      if constexpr (std::is_same_v<T, bool>) {
         const size_t n = hostData_.size();
         bool raw_data[n];
         HANDLE_ERROR(cudaMemcpy(raw_data, devicePtr_, n * sizeof(bool), cudaMemcpyDeviceToHost));
         for (size_t i = 0; i < n; ++i) {
            hostData_[i] = raw_data[i];
         }
      } else {
         HANDLE_ERROR(cudaMemcpy(hostData_.data(), devicePtr_, hostData_.size() * sizeof(T),
                                 cudaMemcpyDeviceToHost));
      }
   }
#endif

private:
   std::vector<T> hostData_;   // Host-side vector

#if defined(__CUDACC__)
   T *devicePtr_;   // Device pointer
#endif
};
