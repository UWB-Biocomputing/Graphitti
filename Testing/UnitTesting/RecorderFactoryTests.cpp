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

#include "Hdf5GrowthRecorder.h"
#include "Utils/Factory.h"
#include "Xml911Recorder.h"
#include "XmlGrowthRecorder.h"
#include "XmlSTDPRecorder.h"
#include "gtest/gtest.h"

TEST(RecorderFactory, GetInstanceReturnsInstance)
{
   Factory<IRecorder> *recorderFactory = &Factory<IRecorder>::getInstance();
   ASSERT_NE(nullptr, recorderFactory);
}

TEST(RecorderFactory, CreateXml911RecorderInstance)
{
   unique_ptr<IRecorder> recorder = Factory<IRecorder>::getInstance().createType("Xml911Recorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<Xml911Recorder *>(recorder.get()));
}

TEST(RecorderFactory, CreateXmlGrowthRecorderInstance)
{
   unique_ptr<IRecorder> recorder
      = Factory<IRecorder>::getInstance().createType("XmlGrowthRecorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<XmlGrowthRecorder *>(recorder.get()));
}

TEST(RecorderFactory, CreateXmlSTDPRecorderInstance)
{
   unique_ptr<IRecorder> recorder = Factory<IRecorder>::getInstance().createType("XmlSTDPRecorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<XmlSTDPRecorder *>(recorder.get()));
}

TEST(RecorderFactory, CreateNonExistentClassReturnsNullPtr)
{
   unique_ptr<IRecorder> recorder = Factory<IRecorder>::getInstance().createType("NonExistent");
   ASSERT_EQ(nullptr, recorder);
}

#if defined(HDF5)
// This test is only possible if HDF5 compilation is available and enabled
TEST(RecorderFactory, CreateHdf5GrowthRecorderInstance)
{
   unique_ptr<IRecorder> recorder
      = Factory<IRecorder>::getInstance().createType("Hdf5GrowthRecorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<Hdf5GrowthRecorder *>(recorder.get()));
}
#endif