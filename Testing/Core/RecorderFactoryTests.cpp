/**
 * @file RecorderFactoryTests.cpp
 *
 * @brief This file contains unit tests for the RecorderFactory using GTest.
 * 
 * @ingroup Testing/Core
 * 
 * We test that the RecorderFactory returns an instance of the correct class
 * we are requesting.
 */

#include "Hdf5GrowthRecorder.h"
#include "RecorderFactory.h"
#include "Xml911Recorder.h"
#include "XmlGrowthRecorder.h"
#include "XmlSTDPRecorder.h"
#include "gtest/gtest.h"

TEST(RecorderFactory, GetInstanceReturnsInstance)
{
   RecorderFactory *recorderFactory = &RecorderFactory::getInstance();
   ASSERT_NE(nullptr, recorderFactory);
}

TEST(RecorderFactory, CreateXml911RecorderInstance)
{
   shared_ptr<IRecorder> recorder = RecorderFactory::getInstance().createRecorder("Xml911Recorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<Xml911Recorder *>(recorder.get()));
}

TEST(RecorderFactory, CreateXmlGrowthRecorderInstance)
{
   shared_ptr<IRecorder> recorder
      = RecorderFactory::getInstance().createRecorder("XmlGrowthRecorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<XmlGrowthRecorder *>(recorder.get()));
}

TEST(RecorderFactory, CreateXmlSTDPRecorderInstance)
{
   shared_ptr<IRecorder> recorder
      = RecorderFactory::getInstance().createRecorder("XmlSTDPRecorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<XmlSTDPRecorder *>(recorder.get()));
}

TEST(RecorderFactory, CreateNonExistentClassReturnsNullPtr)
{
   shared_ptr<IRecorder> recorder = RecorderFactory::getInstance().createRecorder("NonExistent");
   ASSERT_EQ(nullptr, recorder);
}

#if defined(HDF5)
// This test is only possible if HDF5 compilation is available and enabled
TEST(RecorderFactory, CreateHdf5GrowthRecorderInstance)
{
   shared_ptr<IRecorder> recorder
      = RecorderFactory::getInstance().createRecorder("Hdf5GrowthRecorder");
   ASSERT_NE(nullptr, recorder);
   ASSERT_NE(nullptr, dynamic_cast<Hdf5GrowthRecorder *>(recorder.get()));
}
#endif