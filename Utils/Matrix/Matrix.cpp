/**
 @file Matrix.cpp
 @brief Abstract base class for Matrices

 It is the intent of this class to provide an efficient superclass
 for self-allocating and de-allocating 1D and 2D Vectors and
 Matrices. Towards that end, subclasses are expected to implement
 member functions in ways that may require the classes to be friends
 of each other (to directly access internal data
 representation). This base class defines the common interface;
 clients should be able to perform the full range of math operations
 using only Matrix objects.
*/

#include <iostream>
#include "Matrix.h"

// Initialize attributes at construction time
// The subclass constructor must set dimensions
Matrix::Matrix(string t, string i, int r, int c, BGFLOAT m)
  : type(t), init(i), rows(r), columns(c), multiplier(m), dimensions(0) {}


/*
 @brief Convenience mutator
 @param t Matrix type (subclasses add legal values; basically, cheapo reflection)
 @param i Matrix initialization (subclasses can also add legal values to this)
 @param r rows in Matrix
 @param c columns in Matrix
 @param m multiplier used for initialization
 @param d indicates one or two dimensional
 */
void Matrix::SetAttributes(string t, string i, int r, int c,
			   BGFLOAT m, int d)
{
  type = t;
  init = i;
  rows = r;
  columns = c;
  multiplier = m;
  dimensions = d;
}


/*
 Stream output operator for the Matrix class
 hierarchy. Subclasses must implement the Print() method
 to take advantage of this.
 @param os the output stream
 @param obj the Matrix object to send to the output stream
 */
ostream& operator<<(ostream& os, const Matrix& obj)
{
  obj.Print(os);
  return os;
}
