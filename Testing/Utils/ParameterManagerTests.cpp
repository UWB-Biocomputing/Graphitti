/**
 * @file ParameterManagerTests.cpp
 *
 * @brief  This class is used for unit testing the ParameterManager using GTest.
 *
 * @ingroup Testing/Utils
 */

#include "BGTypes.h"
#include "ParameterManager.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

using namespace std;

/// Used for comparing types with no "==" operator
template <typename T> static bool AreEqual(T f1, T f2)
{
   return (fabs(f1 - f2) <= numeric_limits<T>::epsilon() * fmax(fabs(f1), fabs(f2)));
}

TEST(ParameterManager, GetInstanceReturnsInstance)
{
   ParameterManager *parameterManager = &ParameterManager::getInstance();
   ASSERT_TRUE(parameterManager != nullptr);
}

TEST(ParameterManager, LoadingXMLFile)
{
   string filepath = "../ThirdParty/TinyXPath/test.xml";
   ASSERT_TRUE(ParameterManager::getInstance().loadParameterFile(filepath));
}

TEST(ParameterManager, LoadingMultipleValidXMLFiles)
{
   string valid[] {"../configfiles/test-medium-100.xml", "../configfiles/test-small.xml",
                   "../configfiles/test-tiny.xml"};
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().loadParameterFile(valid[i]));
   }
}

TEST(ParameterManager, LoadingMultipleInvalidFiles)
{
   string invalid[] = {"../Core/Graphitti_Main.cpp", "/.this"};
   for (int i = 0; i < 2; i++) {
      ASSERT_FALSE(ParameterManager::getInstance().loadParameterFile(invalid[i]));
   }
}

TEST(ParameterManager, ValidStringTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string validXPaths[] {
      "/BGSimParams/ModelParams/RecorderParams/RecorderFiles/resultFileName/text()",
      "//resultFileName/text()", "//VerticesParams/@class"};
   string result[] = {"Output/Results/test-medium-500-out.xml",
                      "Output/Results/test-medium-500-out.xml", "AllLIFNeurons"};
   string referenceVar;
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().getStringByXpath(validXPaths[i], referenceVar));
      EXPECT_EQ(referenceVar, result[i]);
   }
}

TEST(ParameterManager, ValidIntTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string valid_xpath[] = {"//maxFiringRate/text()",
                           "//RNGConfig/NoiseRNGSeed/text()",
                           "//numEpochs/text()"};
   int result[] = {200, 1, 500};
   int referenceVar;
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().getIntByXpath(valid_xpath[i], referenceVar));
      ASSERT_EQ(referenceVar, result[i]);
   }
}

TEST(ParameterManager, InvalidIntTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string invalidXPath[] = {"//Iinject",
                            "//activeNListFileName/text()",
                            "//beta/text()",
                            "//Iinject/min/text()",
                            "//LayoutFiles/@name",
                            "//NoSuchPath",
                            ""};
   int referenceVar;
   for (int i = 0; i < 7; i++) {
      ASSERT_FALSE(ParameterManager::getInstance().getIntByXpath(invalidXPath[i], referenceVar));
   }
}

TEST(ParameterManager, ValidFloatTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string validXPaths[]
      = {"//Vthresh/min/text()", "//Vresting/min/text()", "//epochDuration/text()"};
   float vals[] = {15.0e-03f, 0.0f, 100.0f};
   float referenceVar;
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().getFloatByXpath(validXPaths[i], referenceVar));
      ASSERT_TRUE(AreEqual(referenceVar, vals[i]));
   }
}

TEST(ParameterManager, InvalidFloatTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string invalidXPaths[] = {"//starter_vthresh", "//nonexistent", ""};
   float referenceVar;
   for (int i = 0; i < 3; i++) {
      ASSERT_FALSE(ParameterManager::getInstance().getFloatByXpath(invalidXPaths[i], referenceVar));
   }
}

TEST(ParameterManager, ValidDoubleTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string validXPaths[]
      = {"//Vthresh/min/text()", "//Vresting/min/text()", "//epochDuration/text()"};
   double vals[] = {15.0e-03, 0.0, 100.0};
   double referenceVar;
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().getDoubleByXpath(validXPaths[i], referenceVar));
      ASSERT_TRUE(AreEqual(referenceVar, vals[i]));
   }
}

TEST(ParameterManager, InvalidDoubleTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string invalidXPaths[]
      = {"//stateOutputFileName/text()", "//starter_vthresh", "//nonexistent", ""};
   double referenceVar;
   for (int i = 0; i < 4; i++) {
      ASSERT_FALSE(
         ParameterManager::getInstance().getDoubleByXpath(invalidXPaths[i], referenceVar));
   }
}

TEST(ParameterManager, ValidBGFloatTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string validXPaths[]
      = {"//Vthresh/min/text()", "//Vresting/min/text()", "//epochDuration/text()"};
   BGFLOAT vals[] = {15.0e-03, 0.0, 100.0};
   BGFLOAT referenceVar;
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().getBGFloatByXpath(validXPaths[i], referenceVar));
      ASSERT_TRUE(AreEqual(referenceVar, vals[i]));
   }
}

TEST(ParameterManager, InvalidBGFloatTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string invalidXPaths[]
      = {"//stateOutputFileName/text()", "//starter_vthresh", "//nonexistent", ""};
   BGFLOAT referenceVar;
   for (int i = 0; i < 4; i++) {
      ASSERT_FALSE(
         ParameterManager::getInstance().getBGFloatByXpath(invalidXPaths[i], referenceVar));
   }
}

TEST(ParameterManager, ValidLongTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string valid_xpath[] = {"//maxFiringRate/text()",
                           "//RNGConfig/NoiseRNGSeed/text()",
                           "//numEpochs/text()"};
   long result[] = {200, 1, 500};
   long referenceVar;
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().getLongByXpath(valid_xpath[i], referenceVar));
      EXPECT_EQ(referenceVar, result[i]);
   }
}

TEST(ParameterManager, InvalidLongTargeting)
{
   ASSERT_TRUE(
      ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml"));
   string invalid_xpath[] = {"//Iinject",
                             "//activeNListFileName/text()",
                             "//beta/text()",
                             "//Iinject/min/text()",
                             "//LayoutFiles/@name",
                             "//NoSuchPath",
                             ""};
   long referenceVar;
   for (int i = 0; i < 7; i++) {
      ASSERT_FALSE(ParameterManager::getInstance().getLongByXpath(invalid_xpath[i], referenceVar));
   }
}

TEST(ParameterManager, InvalidIntVectorTargeting)
{
   vector<int> referenceVar;
   ASSERT_FALSE(ParameterManager::getInstance().getIntVectorByXpath(
      "../configfiles/NList/ActiveNList10x10-0.1.xml", "P", referenceVar));

   ASSERT_FALSE(
      ParameterManager::getInstance().getIntVectorByXpath("notfound.xml", "A", referenceVar));
}
