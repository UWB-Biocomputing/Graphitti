/**
 * @file MatrixFactory.cpp
 * 
 * @ingroup Simulator/Utils/Matrix
 * 
 * @brief Deserializes Matrices from XML
 */


#include "MatrixFactory.h"
#include "SourceVersions.h"
#include <iostream>

static VersionInfo version("$Id: MatrixFactory.cpp,v 1.1.1.1 2006/11/18 04:42:32 fumik Exp $");

/// Get Matrix attributes
/// Inputs:
///   matElement: pointer to the Matrix TiXmlElement
/// Outputs:
///   type:  "diag" (diagonal matrices), "complete" (all values
///          specified), or "sparse". Required.
///   init:  "none" (initialization data explicitly given, default),
///          "const" (initialized to muliplier, if present, else 1.0),
///          "random" (random values in the range [0,1]),
///          or "implementation" (must be initialized by caller, not
///          creator). Optional (defaults to "none")
///   rows:  number of matrix rows. Required.
///   columns: number of matrix columns. Required.
///   multiplier: constant multiplier used in initialization, default
///               1.0. Optional (defaults to 1.0).
void MatrixFactory::GetAttributes(TiXmlElement *matElement, string &type, string &init, int &rows,
                                  int &columns, FLOAT &multiplier)
{
   const char *temp = nullptr;
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));


#ifdef MDEBUG
   LOG4CPLUS_ERROR(consoleLogger, "Getting attributes:\n");
#endif
   temp = matElement->Attribute("type");
   if (temp != nullptr)
      type = temp;
   else
      type = "undefined";
   if ((type != "diag") && (type != "complete") && (type != "sparse"))
      throw KII_invalid_argument("Illegal matrix type: " + type);
#ifdef MDEBUG
   string message = "\trows=" + rows + ", ";
   LOG4CPLUS_TRACE(consoleLogger,  message);
#endif

   if (matElement->QueryIntAttribute("rows", &rows) != TIXML_SUCCESS)
      throw KII_invalid_argument("Number of rows not specified for Matrix.");
#ifdef MDEBUG
   string message = "\trows=" + rows + ", ";
   LOG4CPLUS_TRACE(consoleLogger, message);
#endif

   if (matElement->QueryIntAttribute("columns", &columns) != TIXML_SUCCESS)
      throw KII_invalid_argument("Number of columns not specified for Matrix.");
#ifdef MDEBUG
   string message = "\tcolumns=" + columns + ", ";
   LOG4CPLUS_TRACE(consoleLogger, message);
#endif

   if (matElement->QueryFLOATAttribute("multiplier", &multiplier) != TIXML_SUCCESS) {
      multiplier = 1.0;
   }
#ifdef MDEBUG
   string message = "\tmultiplier=" + multiplier + ", ";
   LOG4CPLUS_TRACE(consoleLogger, message);
#endif

   temp = matElement->Attribute("init");
   if (temp != nullptr)
      init = temp;
   else
      init = "none";
#ifdef MDEBUG
   string message = "\tinit=" + init + "\n";
   LOG4CPLUS_TRACE(consoleLogger, message);
#endif
}

/// This function creates a Matrix subclass element from the given
/// tinyxml Element and its children. The class created is determined
/// from the attributes of the Element
///
/// Input:
///   matElement: tinyxml DOM node containing a Matrix element
/// Postconditions"
///   If no problems, correct Matrix subclass object created and
///   initialized.
/// Returns:
///   Pointer to created Matrix (nullptr if failure).
Matrix *MatrixFactory::CreateMatrix(TiXmlElement *matElement)
{
   string type;
   string init;
   int rows, columns;
   FLOAT multiplier;
   Matrix *theMatrix = nullptr;
   TiXmlHandle matHandle(matElement);
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef MDEBUG
   string message = "Creating Matrix with attributes: " + type + ", " + init + ", " + rows + "X"
        + columns + ", " + multiplier + "\n";
   LOG4CPLUS_ERROR(consoleLogger, message);
#endif

   if (init == "implementation")
      throw KII_invalid_argument(
         "MatrixFactory cannot create implementation-dependent Matrices; client program must perform creation.");

   if (type == "complete") {
      string values;
      // Get the Text node that contains the matrix values, if needed
      if (init == "none") {
         TiXmlText *valuesNode = matHandle.FirstChild().Text();
         if (valuesNode == nullptr)
            throw KII_invalid_argument("Contents not specified for Matrix with init='none'.");
         values = valuesNode->Value();
#ifdef MDEBUG
         string message = "\tData present for initialization: " + values + "\n";
         LOG4CPLUS_ERROR(consoleLogger, message);
#endif
      }
      if ((rows > 1) && (columns > 1))   // Create a 2D Matrix
         theMatrix = new CompleteMatrix(type, init, rows, columns, multiplier, values);
      else   // Create a 1D Matrix
         theMatrix = new VectorMatrix(type, init, rows, columns, multiplier, values);
   } else if (type == "diag") {   // Implement diagonal matrices as sparse
      if (init == "none") {       // a string of values is present & passed
         TiXmlText *valuesNode = matHandle.FirstChild().Text();
         if (valuesNode == nullptr)
            throw KII_invalid_argument(
               "Contents not specified for Sparse Matrix with init='none'.");
         const char *values = valuesNode->Value();
#ifdef MDEBUG
         string message = "\tData present for initialization: " + values + "\n";
         LOG4CPLUS_ERROR(consoleLogger, message);
#endif
         theMatrix = new SparseMatrix(rows, columns, multiplier, values);
      } else if (init == "const") {   // No string of values or XML row data
         theMatrix = new SparseMatrix(rows, columns, multiplier);
      } else
         throw KII_invalid_argument("Invalid init for sparse matrix");
   } else if (type == "sparse") {
      if (init == "none")   // a sequence of row data nodes is present & passed
         theMatrix = new SparseMatrix(rows, columns, multiplier, matElement);
      else if (init == "const") {   // No row data
         if (multiplier == 0.0)
            theMatrix = new SparseMatrix(rows, columns);
         else
            throw KII_invalid_argument(
               "A sparse matrix can only be initialized to zero with const XML init");
      } else
         throw KII_invalid_argument(
            "A sparse matrix can only be initialized to zero with const XML init");
   } else
      throw KII_invalid_argument("Illegal Matrix type");

   return theMatrix;
}

/// This function creates a VectorMatrix from the given tinyxml Element
/// and its children.
///
/// Input:
///   matElement: tinyxml DOM node containing a Matrix element
/// Postconditions"
///   If no problems, VectorMatrix object created and
///   initialized.
/// Returns:
///   VectorMatrix object (will be empty if some failure occurs).
VectorMatrix MatrixFactory::CreateVector(TiXmlElement *matElement)
{
   string type;
   string init;
   int rows, columns;
   FLOAT multiplier;
   string values;
   TiXmlHandle matHandle(matElement);
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef VDEBUG
   string message = "Creating Vector with attributes: "  + type + ", " + init + ", " + rows + "X"
        + columns + ", " + multiplier + "\n";
   LOG4CPLUS_ERROR(consoleLogger, message);
#endif

   // Get the Text node that contains the matrix values, if needed
   if (init == "none") {
      TiXmlText *valuesNode = matHandle.FirstChild().Text();
      if (valuesNode == nullptr)
         throw KII_invalid_argument("Contents not specified for Vector with init='none'.");

      values = valuesNode->Value();
#ifdef VDEBUG
      LOG4CPLUS_ERROR(consoleLogger, ("\tData present for initialization: " << values << endl));
#endif
   } else if (init == "implementation")
      throw KII_invalid_argument(
         "MatrixFactory cannot create implementation-dependent Matrices; client program must perform creation");

   if (type == "sparse")
      throw KII_invalid_argument("Sparse matrix requested in XML but CreateVector called");

   if ((type == "complete") || (type == "diag")) {
      if ((rows > 1) && (columns > 1))   // Create a 2D Matrix
         throw KII_domain_error("Cannot create Vector with more than one dimension.");
      else   // Create a 1D Matrix
         return VectorMatrix(type, init, rows, columns, multiplier, values);
   } else if (type == "sparse")
      throw KII_invalid_argument("No such thing as sparse Vectors");
   else
      throw KII_invalid_argument("Illegal Vector type");

   return VectorMatrix();
}

/// This function creates a CompleteMatrix from the given tinyxml Element
/// and its children.
///
/// Input:
///   matElement: tinyxml DOM node containing a Matrix element
/// Postconditions"
///   If no problems, CompleteMatrix subclass object created and
///   initialized.
/// Returns:
///   CompleteMatrix object (will be empty if some failure occurs).
CompleteMatrix MatrixFactory::CreateComplete(TiXmlElement *matElement)
{
   string type;
   string init;
   int rows, columns;
   FLOAT multiplier;
   string values;
   TiXmlHandle matHandle(matElement);
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef MDEBUG
   string message = "Creating Matrix with attributes: " + type + ", " + init + ", " + rows + "X"
        + columns + ", " + multiplier + "\n";
   LOG4CPLUS_ERROR(consoleLogger, message);
#endif

   // Get the Text node that contains the matrix values, if needed
   if (init == "none") {
      TiXmlText *valuesNode = matHandle.FirstChild().Text();
      if (valuesNode == nullptr)
         throw KII_invalid_argument("Contents not specified for Matrix with init='none'.");

      values = valuesNode->Value();
#ifdef MDEBUG
      LOG4CPLUS_ERROR(consoleLogger, ("\tData present for initialization: " << values << endl));
#endif
   } else if (init == "implementation")
      throw KII_invalid_argument(
         "MatrixFactory cannot create implementation-dependent Matrices; client program must perform creation.");

   if (type == "sparse")
      throw KII_invalid_argument("Sparse matrix requested by XML but CreateComplete called");

   if ((type == "complete") || (type == "diag"))
      return CompleteMatrix(type, init, rows, columns, multiplier, values);
   else if (type == "sparse")
      throw KII_invalid_argument("No such thing as sparse CompleteMatrices");
   else
      throw KII_invalid_argument("Illegal Vector type");

   return CompleteMatrix();
}

///  Create a SparseMatrix, based
///  on the XML attributes. The object is returned by value.
///  @throws KII_invalid_argument
///  @param matElement pointer to Matrix XML element
///  @result The SparseMatrix object.
SparseMatrix MatrixFactory::CreateSparse(TiXmlElement *matElement)
{
   string type;
   string init;
   int rows, columns;
   FLOAT multiplier;
   TiXmlHandle matHandle(matElement);
   log4cplus::Logger consoleLogger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("console"));

   GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef MDEBUG
   string message = "Creating SparseMatrix with attributes: " + type + ", " + init << ", " + rows + "X"
        + columns + ", " + multiplier + "\n";
   LOG4CPLUS_ERROR(consoleLogger, );
#endif

   if (type == "diag") {
      if (init == "none") {   // a string of values is present & passed
         TiXmlText *valuesNode = matHandle.FirstChild().Text();
         if (valuesNode == nullptr)
            throw KII_invalid_argument(
               "Contents not specified for Sparese Matrix with init='none'.");
         const char *values = valuesNode->Value();
#ifdef MDEBUG
         string message = "\tData present for initialization: " + values + "\n";
         LOG4CPLUS_ERROR(consoleLogger, message);
#endif
         return SparseMatrix(rows, columns, multiplier, values);
      } else if (init == "const") {   // No string of values or XML row data
         if (multiplier == 0.0)
            return SparseMatrix(rows, columns);
         else
            throw KII_invalid_argument(
               "A sparse matrix can only be initialized to zero with const XML init");
      } else
         throw KII_invalid_argument("Invalid init for sparse matrix");
   } else if (type == "sparse") {
      if (init == "none")   // a sequence of row data nodes is present & passed
         return SparseMatrix(rows, columns, multiplier, matElement);
      else if (init == "const") {   // No row data
         if (multiplier == 0.0)
            return SparseMatrix(rows, columns);
         else
            throw KII_invalid_argument(
               "A sparse matrix can only be initialized to zero with const XML init");
      } else
         throw KII_invalid_argument(
            "A sparse matrix can only be initialized to zero with const XML init");
   }

   // If we get here, then something is really wrong
   throw KII_invalid_argument("Invalid type specified for sparse matrix.");
}
