/**
 * @file ConnectionsFactory.h
 * 
 * @ingroup Simulator/Connections
 *
 * @brief A factory class for creating Connections objects.
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Connections.h"
#include "Global.h"

class ConnectionsFactory {

	public:
		~ConnectionsFactory();

		static ConnectionsFactory* getInstance() {
			static ConnectionsFactory instance;
			return &instance;
		}

		/// Invokes constructor for desired concrete class
		std::shared_ptr<Connections> createConnections(const std::string& className);

		/// Delete these methods because they can cause copy instances of the singleton when using threads.
		ConnectionsFactory(const ConnectionsFactory&) = delete;
		void operator=(const ConnectionsFactory&) = delete;

	private:
		/// Constructor is private to keep a singleton instance of this class.
		ConnectionsFactory();

		/// Pointer to connections instance.
		std::shared_ptr<Connections> connectionsInstance;

		/// Defines function type for usage in internal map
		using CreateFunction = Connections *(*)(void);

		/// Defines map between class name and corresponding ::Create() function.
		using ConnectionsFunctionMap = std::map<std::string, CreateFunction>;

		/// Makes class-to-function map an internal factory member.
		ConnectionsFunctionMap createFunctions;

		/// Retrieves and invokes correct ::Create() function.
		Connections* invokeCreateFunction(const std::string& className);

		/// Register connection class and it's create function to the factory.
		void registerClass(const std::string& className, CreateFunction function);
};
