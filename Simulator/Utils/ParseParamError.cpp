/**
 * @file ParseParamError.cpp
 * 
 * @ingroup Simulator/Utils
 *
 * @brief Handles parameter error messaging
 *
 */

#include "ParseParamError.h"

ParseParamError::ParseParamError(const std::string paramName, const std::string errorMessage) :
	m_paramName(paramName)
	, m_errorMessage(errorMessage) {
	// Constructor
}

void ParseParamError::print(std::ostream& output) const {
	output << "ERROR :: Failed to parse parameter \"" << m_paramName << "\".";
	output << " Cause: " << m_errorMessage;
}
