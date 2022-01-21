/**
 * @file LayoutFactory.h
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief A factory class for creating Vertices objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "Layout.h"

class LayoutFactory {

	public:
		~LayoutFactory();

		static LayoutFactory* getInstance() {
			static LayoutFactory instance;
			return &instance;
		}

		// Invokes constructor for desired concrete class
		std::shared_ptr<Layout> createLayout(const std::string& className);

		/// Delete these methods because they can cause copy instances of the singleton when using threads.
		LayoutFactory(const LayoutFactory&) = delete;
		void operator=(const LayoutFactory&) = delete;

	private:
		/// Constructor is private to keep a singleton instance of this class.
		LayoutFactory();

		/// Smart pointer to layout instance
		std::shared_ptr<Layout> layoutInstance;

		/// Defines function type for usage in internal map
		using CreateFunction = Layout *(*)(void);

		/// Defines map between class name and corresponding ::Create() function.
		using LayoutFunctionMap = std::map<std::string, CreateFunction>;

		/// Makes class-to-function map an internal factory member.
		LayoutFunctionMap createFunctions;

		/// Retrieves and invokes correct ::Create() function.
		Layout* invokeCreateFunction(const std::string& className);

		/// Register neuron class and it's create function to the factory.
		void registerClass(const std::string& className, CreateFunction function);
};
