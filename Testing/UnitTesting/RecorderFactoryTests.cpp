/**
 * @file RecorderFactoryTests.cpp
 *
 * @brief This file contains unit tests for the RecorderFactory using GTest.
 * 
 * @ingroup Testing/UnitTesting
 * 
 * We test that the RecorderFactory returns an instance of the correct class
 * we are requesting.
 */

#include "Hdf5Recorder.h"
#include "Utils/Factory.h"
#include "Xml911Recorder.h"
#include "gtest/gtest.h"

TEST(RecorderFactory, GetInstanceReturnsInstance)
{
   Factory<Recorder> *recorderFactory = &Factory<Recorder>::getInstance();
   ASSERT_NE(nullptr, recorderFactory);
}

TEST(RecorderFactory, CreateXml911RecorderInstance)
{
   unique_ptr<Recorder> recorder = Factory<Recorder>::getInstance().createType("Xml911Recorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<Xml911Recorder *>(recorder.get()));
}

TEST(RecorderFactory, CreateNonExistentClassReturnsNullPtr)
{
   unique_ptr<Recorder> recorder = Factory<Recorder>::getInstance().createType("NonExistent");
   ASSERT_EQ(nullptr, recorder);
}

#if defined(HDF5)
// This test is only possible if HDF5 compilation is available and enabled
TEST(RecorderFactory, CreateHdf5RecorderInstance)
{
   unique_ptr<Recorder> recorder = Factory<Recorder>::getInstance().createType("Hdf5Recorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<Hdf5Recorder *>(recorder.get()));
}
#endif