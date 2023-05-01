/**
 * @file ArraySpeedTest.cpp
 * 
 * @ingroup Testing/ArrayPerformance
 *
 * @brief Test comparing the efficiency of Array, Vector, and Valarray
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <valarray>
#include <stdio.h>

using namespace std;

typedef chrono::high_resolution_clock Clock;

int number_of_elements = 100000000;
int output_width = floor(log10(number_of_elements) + 1);


int* theArray;
int* randomArray;
vector<int> theVector;
vector<int> theAtVector;
valarray<int> theValarray;

void testArrayInitialize() {
  theArray = new int[number_of_elements];
  fill(theArray, theArray+number_of_elements, 0);
}

void testVectorInitialize() {
  theVector.reserve(number_of_elements);
  //fill(theVector.begin(), theVector.end(), 0);
  for (unsigned int i = 0; i < number_of_elements; ++i)
    theVector.push_back(0);
}

void testValarrayInitialize() {
  theValarray.resize(number_of_elements);
  for (unsigned int i = 0; i <= number_of_elements; i++)
    theValarray[i] = 0;
}

int testArraySequentialAccess() {
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; ++i)
    sum += theArray[i];
  return sum;
}

int testVectorSequentialAccess(){
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; i++)
    sum += theVector[i];
  return sum;
}

int testAtVectorSequentialAccess(){
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; i++)
    sum += theVector.at(i);
  return sum;
}

int testValarraySequentialAccess() {
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; i++)
    sum += theValarray[i];
  return sum;
}

int testArrayRandomAccess(int *randomArray) {
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; i++)
    sum += theArray[randomArray[i]];
  return sum;
}

int testVectorRandomAccess(int *randomArray) {
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; i++)
    sum += theVector[randomArray[i]];
  return sum;
}

int testAtVectorRandomAccess(int *randomArray) {
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; i++)
    sum += theVector.at(randomArray[i]);
  return sum;
}

int testValarrayRandomAccess(int *randomArray) {
  int sum = 0;
  for (unsigned int i = 0; i < number_of_elements; i++)
    sum += theValarray[randomArray[i]];
  return sum;
}

int main() {
  //output formatting
  cout << fixed << setprecision(output_width) << std::left;

  //initialize
  auto clockStart = Clock::now();
  testArrayInitialize();
  auto clockEnd = Clock::now();
  cout << "Array initialization took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  clockStart = Clock::now();
  testVectorInitialize();
  clockEnd = Clock::now();
  cout << "Vector initialization took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  clockStart = Clock::now();
  testValarrayInitialize();
  clockEnd = Clock::now();
  cout << "Valarray initialization took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  // access and sum
  clockStart = Clock::now();
  int arraySequentialOutput = testArraySequentialAccess();
  clockEnd = Clock::now();
  cout << "Array access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  clockStart = Clock::now();
  int vectorSequentialOutput = testVectorSequentialAccess();
  clockEnd = Clock::now();
  cout << "Vector [] access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  theAtVector = theVector;

  clockStart = Clock::now();
  int atVectorSequentialOutput = testAtVectorSequentialAccess();
  clockEnd = Clock::now();
  cout << "Vector .at() access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  clockStart = Clock::now();
  int valarraySequentialOutput = testValarraySequentialAccess();
  clockEnd = Clock::now();
  cout << "Valarray access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  // fill random array
  random_device rd;  // seeds the rng
  mt19937 gen(rd()); // mersenne twister seeded with rd()
  uniform_int_distribution<> distrib(0,(number_of_elements - 1));  // range of output
  randomArray = new int[number_of_elements];
  for (unsigned int i = 0; i < number_of_elements; ++i)
    randomArray[i] = distrib(gen);

  // random access and sum
  clockStart = Clock::now();
  int arrayRandomOutput = testArrayRandomAccess(randomArray);
  clockEnd = Clock::now();
  cout << "Array random access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  clockStart = Clock::now();
  int vectorRandomOutput = testVectorRandomAccess(randomArray);
  clockEnd = Clock::now();
  cout << "Vector [] random access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  clockStart = Clock::now();
  int atVectorRandomOutput = testAtVectorRandomAccess(randomArray);
  clockEnd = Clock::now();
  cout << "Vector .at() random access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  clockStart = Clock::now();
  int valarrayRandomOutput = testValarrayRandomAccess(randomArray);
  clockEnd = Clock::now();
  cout << "Valarray random access and sum took \t" << setw(output_width)
    << chrono::duration_cast<chrono::microseconds>(clockEnd - clockStart).count() << "µs" << endl;

  cout << "total sum: " << (arraySequentialOutput + vectorSequentialOutput + valarraySequentialOutput +
  arrayRandomOutput + vectorRandomOutput + valarrayRandomOutput + atVectorSequentialOutput + atVectorRandomOutput);
}