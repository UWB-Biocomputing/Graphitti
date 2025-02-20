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
#include "RecordableBase.h"
#include "RecordableVector.h"
#include "Recorder.h"
#include "Utils/Factory.h"
#include "Utils/Matrix/VectorMatrix.h"
#include "XmlRecorder.h"
#include "gtest/gtest.h"
#include <iostream>
#include <tinyxml.h>
#include <variant>

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

// Test case for registering a RecordableBase variable
// Test EventBuffer
TEST(XmlRecorderTest, RegisterVariableTest)
{
   // Create an instance of XmlRecorder
   XmlRecorder recorder;
   // Create an EventBuffer for testing
   EventBuffer eventBuffer;

   // Register the EventBuffer variable
   recorder.registerVariable("eventBuffer", eventBuffer, Recorder::UpdatedType::DYNAMIC);

   // Verify that the variable is stored correctly
   ASSERT_EQ("eventBuffer", recorder.getVariableName(0));
   ASSERT_EQ(&eventBuffer, &recorder.getSingleVariable(0));
   // check the type or other details
   ASSERT_EQ(typeid(uint64_t).name(), recorder.getDataType(0));
}

// Test case for registering a RecordableBase variable
// Test VertexMatrix
TEST(XmlRecorderTest, RegisterVectorMatrixTest)
{
   // Create an instance of XmlRecorder
   XmlRecorder recorder;
   // Create an EventBuffer for testing
   VectorMatrix locations;

   // Register the EventBuffer variable
   recorder.registerVariable("location", locations, Recorder::UpdatedType::DYNAMIC);

   // Verify that the variable is stored correctly
   ASSERT_EQ("location", recorder.getVariableName(0));
   ASSERT_EQ(&locations, &recorder.getSingleVariable(0));
   // check the type or other details
   ASSERT_EQ(typeid(BGFLOAT).name(), recorder.getDataType(0));
}

// Test case for registering a RecordableBase variable
// Test standard library vector and RecordableVector
TEST(XmlRecorderTest, RegisterRecordableVectorTest)
{
   // Create an instance of XmlRecorder
   XmlRecorder recorder;
   // Create an EventBuffer for testing
   RecordableVector<BGFLOAT> vectorRadii;

   // Register the EventBuffer variable
   recorder.registerVariable("vectorRadii", vectorRadii, Recorder::UpdatedType::DYNAMIC);

   // Verify that the variable is stored correctly
   ASSERT_EQ("vectorRadii", recorder.getVariableName(0));
   ASSERT_EQ(&vectorRadii, &recorder.getSingleVariable(0));
   // check the type or other details
   ASSERT_EQ(typeid(BGFLOAT).name(), recorder.getDataType(0));
}

// Test for registering a RecordableVector of VertexType enums (EXC, INH)
TEST(XmlRecorderTest, RegisterVectorVertexTypeTest)
{
   // Create an instance of XmlRecorder
   XmlRecorder recorder;

   // Create a RecordableVector<vertexType> 
   RecordableVector<vertexType> vertTypes;
   vertTypes.resize(2);
   vertTypes[0] = vertexType::EXC;
   vertTypes[1] = vertexType::INH;

   // Register the RecordableVector of vertexType
   recorder.registerVariable("VertexTypes", vertTypes, Recorder::UpdatedType::DYNAMIC);

   // Verify that the variable is registered correctly
   ASSERT_EQ("VertexTypes", recorder.getVariableName(0));
   ASSERT_EQ(&vertTypes, &recorder.getSingleVariable(0));

   // Check the data type of the registered variable (matches vertexType)
   ASSERT_EQ(typeid(vertexType).name(), recorder.getDataType(0));
}

// Unit test for registerVariable method with a vector of RecordableBase
TEST(XmlRecorderTest, RegisterVectorVariableTest)
{
   // Create an instance of XmlRecorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/test_output.xml";
   unique_ptr<XmlRecorder> recorderTest_(new XmlRecorder(outputFile));
   ASSERT_TRUE(recorderTest_ != nullptr);

   // Create mock EventBuffer objects for testing
   EventBuffer buffer0;
   EventBuffer buffer1;

   // Create a vector of pointers to EventBuffer objects
   std::vector<RecordableBase *> bufferPointers = {&buffer0, &buffer1};

   // Register variables
   recorderTest_->registerVariable("neuron_", bufferPointers, Recorder::UpdatedType::DYNAMIC);

   // Verify that the registered variables are stored correctly
   ASSERT_EQ("neuron_0", recorderTest_->getVariableName(0));
   ASSERT_EQ("neuron_1", recorderTest_->getVariableName(1));
   ASSERT_EQ(&buffer0, &recorderTest_->getSingleVariable(0));
   ASSERT_EQ(&buffer1, &recorderTest_->getSingleVariable(1));
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
   // buffer size is set to 4
   EventBuffer buffer0(4);

   // Register variables
   recorderTest_->registerVariable("neuron0", buffer0, Recorder::UpdatedType::DYNAMIC);

   // Insert some events into the event buffer
   buffer0.insertEvent(1);
   buffer0.insertEvent(2);

   // Call the compileHistories method
   recorderTest_->compileHistories();
   vector<std::variant<uint64_t, bool, int, BGFLOAT, vertexType>> history = recorderTest_->getHistory(0);

   // Verify the events compiled hisotry
   uint64_t data = 1;
   for (int i = 0; i < 2; i++) {
      EXPECT_EQ(data, get<uint64_t>(history[i]));
      data = data + 1;
   }
}

// Test case for ToXML method
TEST(XmlRecorderTest, ToXML)
{
   // Create an instance of XmlRecorder
   std::string outputFile = "../Testing/UnitTesting/TestOutput/test_output.xml";
   unique_ptr<XmlRecorder> recorderTest_(new XmlRecorder(outputFile));

   // Add some dummy data to variableHistory_
   vector<std::variant<uint64_t, bool, int, BGFLOAT, vertexType>> variableHistory
      = {uint64_t(15), uint64_t(20)};

   // Test the toXML method
   std::string xmlOutput
      = recorderTest_->getToXML("TestVar", variableHistory, typeid(uint64_t).name());

   // Verify the expected XML output
   stringstream os;
   os << "<Matrix ";
   os << "name=\"" << "TestVar" << "\" ";
   os << "type=\"complete\" rows=\"" << 1 << "\" columns=\"" << variableHistory.size()
      << "\" multiplier=\"1.0\">" << endl;
   os << "   ";
   for (int i = 0; i < variableHistory.size(); i++) {
      os << get<uint64_t>(variableHistory[i]) << " ";
   }
   os << endl;
   os << "</Matrix>";
   string expectedOutput = os.str();
   EXPECT_EQ(xmlOutput, expectedOutput);
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
   recorderTest_->registerVariable("neuron0", buffer, Recorder::UpdatedType::DYNAMIC);
   // Insert some events into the event buffer
   buffer.insertEvent(1);
   buffer.insertEvent(2);
   buffer.insertEvent(3);

   // Call the compileHistories method
   recorderTest_->compileHistories();
   // Call the saveSimData() function
   recorderTest_->saveSimData();

   // Open the test_output.xml file and read its content
   std::ifstream inputFile("../Testing/UnitTesting/TestOutput/test_output.xml");
   std::stringstream outputBuffer;
   outputBuffer << inputFile.rdbuf();
   inputFile.close();
   // checks for saving simulation data
   vector<std::variant<uint64_t, bool, int, BGFLOAT, vertexType>> mock_history
      = {uint64_t(1), uint64_t(2), uint64_t(3)};
   std::string expect_header = "<?xml version=\"1.0\" standalone=\"no\"?>\n";
   std::string expect_end = "\n";
   std::string expectXML
      = expect_header + recorderTest_->getToXML("neuron0", mock_history, typeid(uint64_t).name())
        + expect_end;
   // vertify the output string
   ASSERT_EQ(outputBuffer.str(), expectXML);
}

// Test case for saving simulation data for vector of enums (vertexType)
TEST(XmlRecorderTest, SaveSimDataVertexTypeTest)
{
   std::string outputFile = "../Testing/UnitTesting/TestOutput/test_vertex_type.xml";
   unique_ptr<XmlRecorder> recorderTest_(new XmlRecorder(outputFile));
   
   // Create a recordable vector 
   RecordableVector<vertexType> vertTypes;
   vertTypes.resize(2);
   vertTypes[0] = vertexType::EXC;  
   vertTypes[1] = vertexType::INH;  

   // Register the RecordableVector of VertexTypes
   recorderTest_->registerVariable("VertexTypes", vertTypes, Recorder::UpdatedType::DYNAMIC);

   // initialize the XmlRecorder object
   recorderTest_->init();
   
   // Call the compileHistories method
   recorderTest_->compileHistories();
   // Call the saveSimData() function
   recorderTest_->saveSimData();

   // Open the test_output.xml file and read its content
   std::ifstream inputFile("../Testing/UnitTesting/TestOutput/test_vertex_type.xml");
   std::stringstream outputBuffer;
   outputBuffer << inputFile.rdbuf();
   inputFile.close();
   
   // checks for saving simulation data
   vector<std::variant<uint64_t, bool, int, BGFLOAT, vertexType>> mock_history
      = {vertexType::EXC, vertexType::INH};

   std::string expect_header = "<?xml version=\"1.0\" standalone=\"no\"?>\n";
   std::string expect_end = "\n";
   std::string expectXML
      = expect_header + recorderTest_->getToXML("VertexTypes", mock_history, typeid(vertexType).name())
        + expect_end;
   // Vertify the output string
   ASSERT_EQ(outputBuffer.str(), expectXML);
}