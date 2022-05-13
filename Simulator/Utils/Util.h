/**
 * @file Util.h
 * 
 * @ingroup Simulator/Utils
 * 
 * @brief Helper function to parse integers in fixed layout
 */

#pragma once

#ifndef _UTIL_H_
   #define _UTIL_H_

   #include <vector>

using namespace std;

void getValueList(const char *val_string, vector<int> *value_list);

#endif
