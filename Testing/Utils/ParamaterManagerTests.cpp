/**
 * @file ParamaterManagerTests.cpp
 *
 * @brief  This class is used for unit testing the ParameterManager using GTest.
 *
 * @ingroup Testing
 *
 */

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"

#include "ParameterManager.h"
#include "BGTypes.h"

using namespace std;

template<typename T>
static bool AreEqual(T f1, T f2) {
   return (fabs(f1 - f2) <= numeric_limits<T>::epsilon() * fmax(fabs(f1), fabs(f2)));
}

TEST(ParameterManager, GetInstanceReturnsInstance) {
   ParameterManager *parameterManager = &ParameterManager::getInstance();
   ASSERT_TRUE(parameterManager != nullptr);
}

TEST(ParameterManager, LoadingXMLFile) {
   fstream fstream;
   string filepath = "../configfiles/test-medium-100.xml";
   fstream.open("C:\\Users\\chris\\OneDrive\\Desktop\\SummerOfBrain\\Utils\\text.txt");
   ASSERT_TRUE(fstream.is_open());
   //ASSERT_TRUE(ParameterManager::getInstance().loadParameterFile(filepath));
}

TEST(ParameterManager, LoadingMultipleValidXMLFiles) {
   string valid[] = {"../configfiles/test-medium-100.xml", "../configfiles/test-large-conected.xml",
                     "../configfiles/test-medium.xml", "../configfiles/test-tiny.xml"};
   for (int i = 0; i < valid->size(); i++) {
      ASSERT_TRUE(ParameterManager::getInstance().loadParameterFile(valid[i]));
   }
}

TEST(ParameterManager, LoadingMultipleInvalidFiles) {
   string invalid[] = {"../Core/BGDriver.cpp", "/.this"};
   for (int i = 0; i < invalid->size(); i++) {
      ASSERT_FALSE(ParameterManager::getInstance().loadParameterFile(invalid[i]));
   }
}

TEST(ParameterManager, ValidStringTargeting) {
   ParameterManager::getInstance().loadParameterFile("../configfiles/test-medium-500.xml");
   string validXPaths[] = {"/BGSimParams/SimInfoParams/OutputParams/stateOutputFileName/text()",
                            "//stateOutputFileName/text()", "//NeuronsParams/@class"};
   string result[] = {"results/test-medium-500-out.xml", "results/test-medium-500-out.xml", "AllLIFNeurons"};
   string s;
   for (int i = 0; i < 3; i++) {
      ASSERT_TRUE(ParameterManager::getInstance().getStringByXpath(validXPaths[i], s));
      EXPECT_EQ(s, result[i]);
   }
}

//   bool testValidIntTargeting() {
//      cout << "\nEntered test method for targeting valid integers in XML file" << endl;
//      ParameterManager *pm = new ParameterManager();
//      if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
//         cout << "Testing valid integers..." << endl;
//         string valid_xpath[] = {"//maxFiringRate/text()", "//PoolSize/x/text()", "//PoolSize/y/text()",
//                                 "//PoolSize/z/text()", "//Seed/value/text()", "//numSims/text()"};
//         int result[] = {200, 30, 30, 1, 1, 500};
//         int val;
//         for (int i = 0; i < 6; i++) {
//            assert(pm->getIntByXpath(valid_xpath[i], val));
//            assert(val == result[i]);
//         }
//         /*
//          * Test the following invalid paths:
//          * string result from XMLNode
//          * string result from string text() value
//          * float/double result
//          * float/double result with scientific notation
//          * @name value that is a string
//          * invalid xpath (node doesn't exist)
//          */
//         cout << "Testing NON-valid integers..." << endl;
//         string invalid_xpath[] = {"//Iinject", "//activeNListFileName/text()",
//                                   "//beta/text()", "//Iinject/min/text()",
//                                   "//LayoutFiles/@name", "//NoSuchPath", ""};
//         for (int i = 0; i < 7; i++) {
//            assert(!(pm->getIntByXpath(invalid_xpath[i], val)));
//         }
//      }
//      delete pm;
//      return true;
//   }
//
//
//   bool testValidFloatTargeting() {
//      cout << "\nEntered test method for targeting valid floats in XML file" << endl;
//      ParameterManager *pm = new ParameterManager();
//      if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
//         string valid_xpaths[] = {"//Vthresh/min/text()", "//Vresting/min/text()", "//Tsim/text()", "//z/text()"};
//         float vals[] = {15.0e-03f, 0.0f, 100.0f, 1};
//         float var;
//         for (int i = 0; i < 4; i++) {
//            assert(pm->getFloatByXpath(valid_xpaths[i], var));
//            assert(AreEqual(var, vals[i]));
//         }
//         string invalid_xpaths[] = {"//starter_vthresh", "//nonexistent", ""};
//         for (int i = 0; i < 3; i++) {
//            assert(!pm->getFloatByXpath(invalid_xpaths[i], var));
//         }
//      }
//      return true;
//   }
//
//   bool testValidDoubleTargeting() {
//      cout << "\nEntered test method for targeting valid doubles in XML file" << endl;
//      ParameterManager *pm = new ParameterManager();
//      if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
//         string valid_xpaths[] = {"//Vthresh/min/text()", "//Vresting/min/text()", "//Tsim/text()", "//z/text()"};
//         double vals[] = {15.0e-03, 0.0, 100.0, 1};
//         double var;
//         for (int i = 0; i < 4; i++) {
//            assert(pm->getDoubleByXpath(valid_xpaths[i], var));
//            assert(AreEqual(var, vals[i]));
//         }
//         string invalid_xpaths[] = {"//stateOutputFileName/text()", "//starter_vthresh", "//nonexistent", ""};
//         for (int i = 0; i < 4; i++) {
//            assert(!pm->getDoubleByXpath(invalid_xpaths[i], var));
//         }
//      }
//      return true;
//   }
//
//   bool testValidBGFloatTargeting() {
//      cout << "\nEntered test method for targeting valid BGFLOATs in XML file" << endl;
//      ParameterManager *pm = new ParameterManager();
//      if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
//         string valid_xpaths[] = {"//Vthresh/min/text()", "//Vresting/min/text()", "//Tsim/text()", "//z/text()"};
//         BGFLOAT vals[] = {15.0e-03, 0.0, 100.0, 1};
//         BGFLOAT var;
//         for (int i = 0; i < 4; i++) {
//            assert(pm->getBGFloatByXpath(valid_xpaths[i], var));
//            assert(AreEqual(var, vals[i]));
//         }
//         string invalid_xpaths[] = {"//stateOutputFileName/text()", "//starter_vthresh", "//nonexistent", ""};
//         for (int i = 0; i < 4; i++) {
//            assert(!pm->getBGFloatByXpath(invalid_xpaths[i], var));
//         }
//      }
//      return true;
//   }
//
//   bool testValidLongTargeting() {
//      cout << "\nEntered test method for targeting valid integers in XML file" << endl;
//      ParameterManager *pm = new ParameterManager();
//      if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
//         cout << "Testing valid integers..." << endl;
//         string valid_xpath[] = {"//maxFiringRate/text()", "//PoolSize/x/text()", "//PoolSize/y/text()",
//                                 "//PoolSize/z/text()", "//Seed/value/text()", "//numSims/text()"};
//         long result[] = {200, 30, 30, 1, 1, 500};
//         long val;
//         for (int i = 0; i < 6; i++) {
//            assert(pm->getLongByXpath(valid_xpath[i], val));
//            assert(val == result[i]);
//         }
//         /*
//          * Test the following invalid paths:
//          * string result from XMLNode
//          * string result from string text() value
//          * float/double result
//          * float/double result with scientific notation
//          * @name value that is a string
//          * invalid xpath (node doesn't exist)
//          */
//         cout << "Testing NON-valid integers..." << endl;
//         string invalid_xpath[] = {"//Iinject", "//activeNListFileName/text()",
//                                   "//beta/text()", "//Iinject/min/text()",
//                                   "//LayoutFiles/@name", "//NoSuchPath", ""};
//         for (int i = 0; i < 7; i++) {
//            assert(!(pm->getLongByXpath(invalid_xpath[i], val)));
//         }
//      }
//      delete pm;
//      return true;
//   }
//
////int main() {
//////   cout << "\nRunning tests for ParameterManager.cpp functionality..." << endl;
//////   bool success = testConstructorAndDestructor();
//////   if (!success) return 1;
//////   success = testValidXmlFileReading();
//////   if (!success) return 1;
//////   success = testValidStringTargeting();
//////   if (!success) return 1;
//////   success = testValidIntTargeting();
//////   if (!success) return 1;
//////   success = testValidFloatTargeting();
//////   if (!success) return 1;
//////   success = testValidDoubleTargeting();
//////   if (!success) return 1;
//////   success = testValidBGFloatTargeting();
//////   if (!success) return 1;
//////   success = testValidLongTargeting();
//////   if (!success) return 1;
//////   return 0;
////}
//
