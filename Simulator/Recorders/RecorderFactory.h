/**
 * @file RecorderFactory.h
 *
 * @ingroup Simulator/Recorders
 * 
 * @brief A factory class for creating Recorder objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "IRecorder.h"

class RecorderFactory {

	public:
		~RecorderFactory();

		static RecorderFactory* getInstance() {
			static RecorderFactory instance;
			return &instance;
		}

		// Invokes constructor for desired concrete class
		std::shared_ptr<IRecorder> createRecorder(const std::string& className);

		/// Delete these methods because they can cause copy instances of the singleton when using threads.
		RecorderFactory(const RecorderFactory&) = delete;
		void operator=(const RecorderFactory&) = delete;

	private:
		/// Constructor is private to keep a singleton instance of this class.
		RecorderFactory();

		/// Pointer to neurons instance
		std::shared_ptr<IRecorder> recorderInstance;

		/// Defines function type for usage in internal map
		using CreateFunction = IRecorder *(*)(void);

		/// Defines map between class name and corresponding ::Create() function.
		using RecorderFunctionMap = std::map<std::string, CreateFunction>;

		/// Makes class-to-function map an internal factory member.
		RecorderFunctionMap createFunctions;

		/// Retrieves and invokes correct ::Create() function.
		IRecorder* invokeCreateFunction(const std::string& className);

		/// Register neuron class and it's create function to the factory.
		void registerClass(const std::string& className, CreateFunction function);
};
