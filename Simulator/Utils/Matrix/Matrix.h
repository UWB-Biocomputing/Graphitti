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
#include "RecordableBase.h"
#include <string>
//cereal
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>

using namespace std;

class Matrix : public RecordableBase {
   friend class cereal::access;

public:
   /// Virtual Destructor
   virtual ~Matrix() = default;

   /// @brief Generate text representation of the Matrix to a stream
   virtual void Print(ostream &os) const = 0;

   ///  Cereal serialization method
   template <class Archive> void serialize(Archive &archive);

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

CEREAL_REGISTER_TYPE(Matrix);
///  Cereal serialization method
template <class Archive> void Matrix::serialize(Archive &archive)
{
   archive(cereal::virtual_base_class<RecordableBase>(this), cereal::make_nvp("type", type), cereal::make_nvp("init", init),
           cereal::make_nvp("rows", rows), cereal::make_nvp("columns", columns),
           cereal::make_nvp("multiplier", multiplier), cereal::make_nvp("dimensions", dimensions));
}
