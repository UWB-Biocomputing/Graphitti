/**
 * @file CompleteMatrix.cpp
 * 
 * @ingroup Simulator/Utils/Matrix
 * 
 * @brief An efficient implementation of a dynamically-allocated 2D array.
 */


#include <iostream>
#include <sstream>

#include "CompleteMatrix.h"
#include "Global.h"

// Create a complete 2D Matrix
/// Allocate storage and initialize attributes. If "v" (values) is
/// not empty, it will be used as a source of data for initializing
/// the matrix (and must be a list of whitespace separated textual
/// numeric data with rows * columns elements, if "t" (type) is
/// "complete", or number of elements in diagonal, if "t" is "diag").
/// 
/// If "i" (initialization) is "const", then "m" will be used to initialize either all
/// elements (for a "complete" matrix) or diagonal elements (for "diag").
/// 
/// "random" initialization is not yet implemented.
/// 
/// @throws Matrix_bad_alloc
/// @throws Matrix_invalid_argument
/// @param t Matrix type (defaults to "complete")
/// @param i Matrix initialization (defaults to "const")
/// @param r rows in Matrix (defaults to 2)
/// @param c columns in Matrix (defaults to 2)
/// @param m multiplier used for initialization (defaults to zero)
/// @param v values for initializing CompleteMatrix (this std::string is parsed as a list of floating point numbers)
CompleteMatrix::CompleteMatrix(std::string t, std::string i, int r,
                               int c, BGFLOAT m, std::string values)
	: Matrix(t, i, r, c, m), theMatrix(nullptr) {
	DEBUG_MATRIX(std::cerr << "Creating CompleteMatrix, size: ";)

	// Bail out if we're being asked to create nonsense
	if (!((rows > 0) && (columns > 0))) throw Matrix_invalid_argument(
		"CompleteMatrix::CompleteMatrix(): Asked to create zero-size");

	// We're a 2D Matrix, even if only one row or column
	dimensions = 2;

	DEBUG_MATRIX(std::cerr << rows << "X" << columns << ":" << std::endl;)

	// Allocate storage
	alloc(rows, columns);

	if (values != "") {
		// Initialize from the text std::string
		std::istringstream valStream(values);
		if (type == "diag") {
			// diagonal matrix with values given
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					theMatrix[i][j] = 0.0; // Non-diagonal elements are zero
					if (i == j) {
						valStream >> theMatrix[i][j];
						theMatrix[i][j] *= multiplier;
					}
				}
			}
		}
		else if (type == "complete") {
			// complete matrix with values given
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					valStream >> theMatrix[i][j];
					theMatrix[i][j] *= multiplier;
				}
			}
		}
		else {
			clear();
			throw Matrix_invalid_argument("Illegal type for CompleteMatrix with 'none' init: " + type);
		}
	}
	else if (init == "const") {
		if (type == "diag") {
			// diagonal matrix with constant values
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					theMatrix[i][j] = 0.0; // Non-diagonal elements are zero
					if (i == j) theMatrix[i][j] = multiplier;
				}
			}
		}
		else if (type == "complete") {
			// complete matrix with constant values
			for (int i = 0; i < rows; i++) { for (int j = 0; j < columns; j++) theMatrix[i][j] = multiplier; }
		}
		else {
			clear();
			throw Matrix_invalid_argument("Illegal type for CompleteMatrix with 'none' init: " + type);
		}
	}
	//  else if (init == "random")
	DEBUG_MATRIX(std::cerr << "\tInitialized " << type << " matrix" << std::endl;)
}


// "Copy Constructor"
CompleteMatrix::CompleteMatrix(const CompleteMatrix& oldM) : theMatrix(nullptr) {
	DEBUG_MATRIX(std::cerr << "CompleteMatrix copy constructor:" << std::endl;)
	copy(oldM);
}

// Destructor
CompleteMatrix::~CompleteMatrix() {
	DEBUG_MATRIX(std::cerr << "Destroying CompleteMatrix" << std::endl;)
	clear();
}


// Assignment operator
CompleteMatrix& CompleteMatrix::operator=(const CompleteMatrix& rhs) {
	if (&rhs == this) return *this;

	DEBUG_MATRIX(std::cerr << "CompleteMatrix::operator=" << std::endl;)

	clear();
	DEBUG_MATRIX(std::cerr << "\t\tclear() complete, ready to copy." << std::endl;)
	copy(rhs);
	DEBUG_MATRIX(std::cerr << "\t\tcopy() complete; returning by reference." << std::endl;)
	return *this;
}

// Clear out storage
void CompleteMatrix::clear(void) {
	DEBUG_MATRIX(std::cerr << "\tclearing " << rows << "X" << columns << " CompleteMatrix...";)

	if (theMatrix != nullptr) {
		for (int i = 0; i < rows; i++) {
			if (theMatrix[i] != nullptr) {
				delete [] theMatrix[i];
				theMatrix[i] = nullptr;
			}
		}
		delete [] theMatrix;
		theMatrix = nullptr;
	}
	DEBUG_MATRIX(std::cerr << "done." << std::endl;)
}


// Copy matrix to this one
void CompleteMatrix::copy(const CompleteMatrix& source) {
	DEBUG_MATRIX(std::cerr << "\tcopying " << source.rows << "X" << source.columns
		<< " CompleteMatrix...";)

	SetAttributes(source.type, source.init, source.rows,
	              source.columns, source.multiplier, source.dimensions);

	alloc(rows, columns);

	for (int i = 0; i < rows; i++) { for (int j = 0; j < columns; j++) theMatrix[i][j] = source.theMatrix[i][j]; }
	DEBUG_MATRIX(std::cerr << "\t\tdone." << std::endl;)
}


/// Allocate internal storage
///
/// Note: If you are getting this memory allocaiton error:
///
/// " terminate called after throwing an instance of 'std::bad_alloc'
///       what():  St9bad_alloc "
///
/// Please refer to LIFModel::Connections()
void CompleteMatrix::alloc(int rows, int columns) {
	if (theMatrix != nullptr) throw MatrixException("Attempt to allocate storage for non-cleared Matrix");

	if ((theMatrix = new BGFLOAT*[rows]) == nullptr) throw
		Matrix_bad_alloc("Failed allocating storage to copy Matrix.");

	for (int i = 0; i < rows; i++) {
		if ((theMatrix[i] = new BGFLOAT[columns]) == nullptr) throw Matrix_bad_alloc(
			"Failed allocating storage to copy Matrix.");
	}
	DEBUG_MATRIX(std::cerr << "\tStorage allocated for "<< rows << "X" << columns << " Matrix." << std::endl;)

}


/// @brief Polymorphic output. Produces text output on stream os. Used by operator<<()
/// @param os stream to output to
void CompleteMatrix::Print(std::ostream& os) const {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) os << theMatrix[i][j] << " ";
		os << std::endl;
	}
}

// convert Matrix to XML std::string
std::string CompleteMatrix::toXML(std::string name) const {
	std::stringstream os;

	os << "<Matrix ";
	if (name != "") os << "name=\"" << name << "\" ";
	os << "type=\"complete\" rows=\"" << rows
		<< "\" columns=\"" << columns
		<< "\" multiplier=\"1.0\">" << std::endl;
	os << "   " << *this << std::endl;
	os << "</Matrix>";

	return os.str();
}


// Math operations. For efficiency's sake, these methods will be
// implemented as being "aware" of each other (i.e., using "friend"
// and including the other subclasses' headers).

const CompleteMatrix CompleteMatrix::operator+(const CompleteMatrix& rhs) const {
	if ((rhs.rows != rows) || (rhs.columns != columns)) throw Matrix_domain_error(
		"Illegal matrix addition: dimension mismatch");
	// Start with this
	CompleteMatrix result(*this);
	// Add in rhs
	for (int i = 0; i < rows; i++) { for (int j = 0; j < columns; j++) result.theMatrix[i][j] += rhs.theMatrix[i][j]; }

	return result;
}


// Multiply the rhs into the current object
const CompleteMatrix CompleteMatrix::operator*(const CompleteMatrix& rhs) const {
	throw Matrix_domain_error("CompleteMatrix product not yet implemented");
}

// Element-wise square root of a vector
const CompleteMatrix sqrt(const CompleteMatrix& m) {
	// Start with vector
	CompleteMatrix result(m);

	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.columns; j++) result.theMatrix[i][j] = sqrt(result.theMatrix[i][j]);
	}

	return result;
}
