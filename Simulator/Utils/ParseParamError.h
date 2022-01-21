/**
 * @file ParseParamError.h
 * 
 * @ingroup Simulator/Utils
 *
 * @brief Handles parameter error messaging
 * 
 */

#ifndef _PARSEPARAMERROR_H_
#define _PARSEPARAMERROR_H_

#include <iostream>
#include <string>

class ParseParamError {
	public:
		ParseParamError(const std::string paramName, const std::string errorMessage);
		void print(std::ostream& output) const;

	private:
		const std::string m_paramName;
		const std::string m_errorMessage;
};

#endif
