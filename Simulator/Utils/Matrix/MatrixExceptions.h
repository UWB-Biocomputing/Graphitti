/**
 * @file MatrixExceptions.h
 * 
 * @ingroup Simulator/Utils/Matrix
 * 
 * @brief Exception class hierarchy for Matrix classes
 */


#pragma once

#include <stdexcept>

using namespace std;

/// Master base class for Matrix exceptions
class MatrixException : public runtime_error
{
public:
  explicit MatrixException(const string&  __arg) : runtime_error(__arg) {}
};

/// Signals memory allocation error for Matrix classes
class Matrix_bad_alloc : public MatrixException
{
public:
  explicit Matrix_bad_alloc(const string&  __arg) : MatrixException(__arg) {}
};

/// Signals bad cast among Matrices for Matrix classes
class Matrix_bad_cast : public MatrixException
{
public:
  explicit Matrix_bad_cast(const string&  __arg) : MatrixException(__arg) {}
};

/// Signals bad function argument for Matrix classes
class Matrix_invalid_argument : public MatrixException
{
public:
  explicit Matrix_invalid_argument(const string&  __arg) : MatrixException(__arg) {}
};
 
/// Signals value bad for domain for Matrix classes
class Matrix_domain_error : public MatrixException
{
public:
  explicit Matrix_domain_error(const string&  __arg) : MatrixException(__arg) {}
};


