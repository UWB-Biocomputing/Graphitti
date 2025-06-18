/**
 * @file DeviceVectorTests.cpp
 *
 * @brief Unit tests for DeviceVector class using GTest.
 *
 * @ingroup Testing/UnitTesting
 */

#include "DeviceVector.h"
#include "gtest/gtest.h"

/// Test to verify DeviceVector supported types
TEST(DeviceVectorTest, SupportedTypes)
{
   // Verify each type is supported
   static_assert(is_device_vector_supported_type<BGFLOAT, DeviceVectorTypes>::value,
                 "BGFLOAT should be supported");
   static_assert(is_device_vector_supported_type<int, DeviceVectorTypes>::value,
                 "int should be supported");
   static_assert(is_device_vector_supported_type<bool, DeviceVectorTypes>::value,
                 "bool should be supported");
   // Test vector creation with supported types
   DeviceVector<BGFLOAT> v1;   // Valid: BGFLOAT
   DeviceVector<int> v2;       // Valid: int
   DeviceVector<bool> v3;      // Valid: bool

   struct TestStruct {};
   // Verify unsupported types are rejected
   static_assert(!is_device_vector_supported_type<TestStruct, DeviceVectorTypes>::value,
                 "struct should not be supported");

   // The following would cause compile errors:
   // DeviceVector<TestStruct> v4;          // Error: unsupported type
   // DeviceVector<EventBuffer> v5;          // Error: unsupported type
}

// Test default construction
TEST(DeviceVectorTest, DefaultConstruction)
{
   DeviceVector<int> vec;
   EXPECT_EQ(vec.size(), 0);
   EXPECT_TRUE(vec.empty());
}

// Test sized construction
TEST(DeviceVectorTest, SizedConstruction)
{
   DeviceVector<int> vec(5);
   EXPECT_EQ(vec.size(), 5);
   EXPECT_FALSE(vec.empty());
}

// Test assignment operation
TEST(DeviceVectorTest, Assignment)
{
   DeviceVector<int> vec;
   vec.assign(3, 42);
   EXPECT_EQ(vec.size(), 3);
   EXPECT_FALSE(vec.empty());
   for (size_t i = 0; i < vec.size(); ++i) {
      EXPECT_EQ(vec[i], 42);
   }
}

// Test vector resizing
TEST(DeviceVectorTest, Resize)
{
   DeviceVector<int> vec;
   vec.resize(5, 10);
   EXPECT_EQ(vec.size(), 5);
   for (size_t i = 0; i < vec.size(); ++i) {
      EXPECT_EQ(vec[i], 10);
   }

   // Resize smaller
   vec.resize(3);
   EXPECT_EQ(vec.size(), 3);
   for (size_t i = 0; i < vec.size(); ++i) {
      EXPECT_EQ(vec[i], 10);   // Original values should be preserved
   }
}

// Test vector clearing
TEST(DeviceVectorTest, Clear)
{
   DeviceVector<int> vec(5);
   EXPECT_FALSE(vec.empty());
   EXPECT_EQ(vec.size(), 5);

   vec.clear();
   EXPECT_TRUE(vec.empty());
   EXPECT_EQ(vec.size(), 0);
}

// Test element access
TEST(DeviceVectorTest, ElementAccess)
{
   DeviceVector<int> vec(3);
   vec[0] = 10;
   vec[1] = 20;
   vec[2] = 30;

   EXPECT_EQ(vec.front(), 10);
   EXPECT_EQ(vec.back(), 30);
   EXPECT_EQ(vec[1], 20);
}

// Test boolean vector specialization
TEST(DeviceVectorTest, BooleanVector)
{
   DeviceVector<bool> vec(3);
   vec[0] = true;
   vec[1] = false;
   vec[2] = true;
   EXPECT_TRUE(vec[0]);
   EXPECT_FALSE(vec[1]);
   EXPECT_TRUE(vec[2]);
}

// Test const access methods
TEST(DeviceVectorTest, ConstAccess)
{
   DeviceVector<int> vec(3);
   vec.assign(3, 42);

   // Test const methods directly on vec
   EXPECT_EQ(vec.size(), 3);
   EXPECT_EQ(vec[0], 42);
   EXPECT_EQ(vec.front(), 42);
   EXPECT_EQ(vec.back(), 42);
   EXPECT_FALSE(vec.empty());

   // Test through const pointer to ensure const-correctness
   const DeviceVector<int> *constVec = &vec;
   EXPECT_EQ(constVec->size(), 3);
   EXPECT_EQ((*constVec)[0], 42);
   EXPECT_EQ(constVec->front(), 42);
   EXPECT_EQ(constVec->back(), 42);
   EXPECT_FALSE(constVec->empty());
}

#if defined(USE_GPU)
// Test device memory allocation
TEST(DeviceVectorTest, DeviceMemoryAllocation)
{
   DeviceVector<int> vec(3);
   vec.assign(3, 42);
   EXPECT_EQ(vec.getDevicePointer(), nullptr);

   vec.allocateDeviceMemory();
   EXPECT_NE(vec.getDevicePointer(), nullptr);
}

// Test device memory and host memory combine copy operations
TEST(DeviceVectorTest, DeviceMemoryCopy)
{
   DeviceVector<int> vec(3);
   vec.allocateDeviceMemory();

   // Test host to device copy
   vec[0] = 10;
   vec[1] = 20;
   vec[2] = 30;
   vec.copyToDevice();

   // Modify host data
   vec[0] = 0;
   vec[1] = 0;
   vec[2] = 0;

   // Copy back from device and verify
   vec.copyToHost();
   EXPECT_EQ(vec[0], 10);
   EXPECT_EQ(vec[1], 20);
   EXPECT_EQ(vec[2], 30);
}

// Test device memory error handling
TEST(DeviceVectorTest, DeviceMemoryErrors)
{
   DeviceVector<int> vec(3);

   // Attempt to copy without allocation should throw
   EXPECT_THROW(vec.copyToDevice(), std::runtime_error);
   EXPECT_THROW(vec.copyToHost(), std::runtime_error);
}

// Test data pointer access
TEST(DeviceVectorTest, DataAccess)
{
   DeviceVector<int> vec(3);
   vec.assign(3, 42);
   const int *data = vec.data();
   const int *device_data = vec.getDevicePointer();

   EXPECT_NE(data, nullptr);
   EXPECT_EQ(device_data, nullptr);   // No device allocation by default
   EXPECT_EQ(data[0], 42);
   EXPECT_EQ(data[1], 42);
   EXPECT_EQ(data[2], 42);
}
#endif
