/**
 * @file XmlRecorderTests.cpp
 *
 * @brief This file contains unit tests for the XmlRecorder using GTest.
 * 
 * @ingroup Testing/UnitTesting
 * 
 * We test that XmlRecorder class records correct output into the xml file
 * we are requesting.
 */
#define RUNIT_TEST
#include "AllLIFNeurons.h"
#include "EventBuffer.h"
#include "Utils/Factory.h"
#include "XmlRecorder.h"
#include "gtest/gtest.h"
#include <iostream>
#include <tinyxml.h>

using namespace std;

// Test case for initializing the XmlRecorder
TEST(XmlRecorderTest, CreateInstanceSuccess)
{
   Recorder *recorder = XmlRecorder::Create();
   ASSERT_TRUE(recorder != nullptr);
}

// Test case for open file successfully
TEST(XmlRecorderTest, InitTest)
{
   // Create an instance of XmlRecorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/test_output.xml";
   XmlRecorder recorder(outputFile);
   recorder.init();
   // Test to see if output file exist
   FILE *f = fopen("../Testing/UnitTesting/TestOutput/test_output.xml", "r");
   bool fileExist = f != NULL;
   fclose(f);
   ASSERT_TRUE(fileExist);
}

// Test case for registering a variable
TEST(XmlRecorderTest, RegisterVariableTest)
{
   // Create an instance of XmlRecorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/test_output.xml";
   unique_ptr<XmlRecorder> recorderTest_(new XmlRecorder(outputFile));
   ASSERT_TRUE(recorderTest_ != nullptr);
   // XmlRecorder recorder(outputFile);
   // Create a mock EventBuffer object
   EventBuffer buffer;

   // Register a variable
   // recorder.registerVariable("neuron1", buffer);
   recorderTest_->registerVariable("neuron1", buffer);

   // Verify that the registered variable is stored correctly
   ASSERT_EQ("neuron1", recorderTest_->getNeuronName());
   ASSERT_EQ(&buffer, &recorderTest_->getSingleNeuronEvents());
}

// Test case for compiling histories
TEST(XmlRecorderTest, CompileHistoriesTest)
{
   // Create an instance of XmlRecorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/test_output.xml";
   unique_ptr<XmlRecorder> recorderTest_(new XmlRecorder(outputFile));
   // Create a mock AllVertices object
   unique_ptr<AllVertices> vertices
      = Factory<AllVertices>::getInstance().createType("AllLIFNeurons");
   ASSERT_NE(nullptr, vertices);
   // Create a mock EventBuffer object
   EventBuffer buffer(4);

   // Register a variable
   recorderTest_->registerVariable("neuron1", buffer);

   // Insert some events into the event buffer
   buffer.insertEvent(1);
   buffer.insertEvent(2);
   buffer.insertEvent(3);

   // Call the compileHistories method
   recorderTest_->compileHistories(*vertices.get());
   std::vector<uint64_t> history = recorderTest_->getHistory();
   // Verify the neuron name
   EXPECT_EQ("neuron1", recorderTest_->getNeuronName());

   // Verify the single neuron events compiled hisotry
   EXPECT_EQ(1, history[0]);
   EXPECT_EQ(2, history[1]);
   EXPECT_EQ(3, history[2]);
}

// Test case for saving simulation data
TEST(XmlRecorderTest, SaveSimDataTest)
{
   // Create an instance of XmlRecorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/test_output.xml";
   unique_ptr<XmlRecorder> recorderTest_(new XmlRecorder(outputFile));
   // Create a mock AllVertices object
   unique_ptr<AllVertices> vertices
      = Factory<AllVertices>::getInstance().createType("AllLIFNeurons");
   ASSERT_NE(nullptr, vertices);
   // Create a mock EventBuffer object
   EventBuffer buffer(4);

   // initialize the XmlRecorder object
   recorderTest_->init();

   // Register a variable
   recorderTest_->registerVariable("neuron1", buffer);

   // Insert some events into the event buffer
   buffer.insertEvent(1);
   buffer.insertEvent(2);
   buffer.insertEvent(3);

   // Call the compileHistories method
   recorderTest_->compileHistories(*vertices.get());
   // Call the saveSimData() function
   recorderTest_->saveSimData(*vertices.get());

   // Open the test_output.xml file and read its content
   std::ifstream inputFile("../Testing/UnitTesting/TestOutput/test_output.xml");
   std::stringstream outputBuffer;
   outputBuffer << inputFile.rdbuf();
   inputFile.close();

   // checks for saving simulation data
   // For example, check if the output file contains the expected XML content
   stringstream os;
   os << "<Matrix ";
   os << "name=\"" << recorderTest_->getNeuronName() << "\" ";
   os << "type=\"complete\" rows=\"" << 1 << "\" columns=\"" << recorderTest_->getHistory().size()
      << "\" multiplier=\"1.0\">" << endl;
   os << "   ";
   for (int i = 0; i < recorderTest_->getHistory().size(); i++) {
      os << recorderTest_->getHistory()[i] << " ";
   }
   os << endl;
   os << "</Matrix>";
   std::string expectedXML = "<?xml version=\"1.0\" standalone=\"no\"?>\n" + os.str() + "\n";
   // vertify the output string
   ASSERT_EQ(outputBuffer.str(), expectedXML);
}