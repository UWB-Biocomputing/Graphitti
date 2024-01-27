/**
 * @file Matrix.h
 * 
 * @ingroup Simulator/Utils/Matrix
 * 
 * @brief Abstract base class for Matrices
 *
 * It is the intent of this class to provide an efficient superclass
 * for self-allocating and de-allocating 1D and 2D Vectors and
 * Matrices. Towards that end, subclasses are expected to implement
 * member functions in ways that may require the classes to be friends
 * of each other (to directly access internal data
 * representation). This base class defines the common interface;
 * clients should be able to perform the full range of math operations
 * using only Matrix objects.
*/

#pragma once

#include "BGTypes.h"
#include "MatrixExceptions.h"
#include <string>

#include "RecordableBase.h"

using namespace std;

class Matrix : public RecordableBase {
public:

// /*{8*/
  virtual int getNumEventsInEpoch() const {
   return 0;
  }

   /// Start a new epoch for the recordable variable.
   /// This method is called at the beginning of each simulation epoch to prepare for recording new events.
   virtual void startNewEpoch() {}

   /// Get the value of the recordable variable at the specified index.
   /// @param index The index of the recorded value to retrieve.
   /// @return A variant representing the recorded value (uint64_t, double, or string).
   virtual variant<uint64_t, double, string, BGFLOAT> getElement(int index) const {
      std::variant<uint64_t, double, string, BGFLOAT> a;
      a = 1.0;
      return a;
   }

   /// set up a string representing the basic data type
   virtual void setDataType() {}

   /// Get A string representing the data type of the recordable variable
   virtual string &getDataType()  {
      basicDataType_ = "variant";
      return basicDataType_;
   }

/*}*/
   /// Virtual Destructor
   virtual ~Matrix() = default;

   /// @brief Generate text representation of the Matrix to a stream
   virtual void Print(ostream &os) const = 0;

protected:
   ///   Initialize attributes at construction time. This is protected to
   ///   prevent construction of Matrix objects themselves. Would be nice
   ///   if C++ just allowed one to declare a class abstract. Really
   ///   obsoleted since the Print() method is pure virtual now.
   ///
   ///   @param t Matrix type (subclasses add legal values; basically, cheapo reflection)
   ///   @param i Matrix initialization (subclasses can also add legal values to this)
   ///   @param r rows in Matrix
   ///   @param c columns in Matrix
   ///   @param m multiplier used for initialization
   Matrix(string t = "", string i = "", int r = 0, int c = 0, BGFLOAT m = 0.0);

   ///  @brief Convenience mutator
   ///  @param t Matrix type (subclasses add legal values; basically, cheapo reflection)
   ///  @param i Matrix initialization (subclasses can also add legal values to this)
   ///  @param r rows in Matrix
   ///  @param c columns in Matrix
   ///  @param m multiplier used for initialization
   ///  @param d indicates one or two dimensional
   void SetAttributes(string t, string i, int r, int c, BGFLOAT m, int d);

   /******************************************
          * @name Attributes from XML files 
         ******************************************/
   ///@{

   /// "complete" == all locations nonzero,
   /// "diag" == only diagonal elements nonzero,
   /// or "sparse" == nonzero values may be anywhere
   string type;

   /// "const" == nonzero values with a fixed constant,
   /// "random" == nonzero values with random numbers,
   /// or "implementation" == uses a built-in function of the specific subclass
   string init;

   int rows;             ///< Number of rows in Matrix (>0)
   int columns;          ///< Number of columns in Matrix (>0)
   BGFLOAT multiplier;   ///< Constant used for initialization
   int dimensions;       ///< One or two dimensional
   ///@}
};

///  Stream output operator for the Matrix class
///  hierarchy. Subclasses must implement the Print() method
///  to take advantage of this.
///  @param os the output stream
///  @param obj the Matrix object to send to the output stream
ostream &operator<<(ostream &os, const Matrix &obj);
