/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "Global.h"
#include "IAllNeurons.h"

using namespace std;

class VerticesFactory {

public:
   ~VerticesFactory();

   static VerticesFactory *getInstance() {
      static VerticesFactory instance;
      return &instance;
   }

   // Invokes constructor for desired concrete class
   shared_ptr<IAllNeurons> createNeurons(const string &className);

   // Shortcut for copy constructor for existing concrete object
   IAllNeurons *createNeuronsCopy();

   /// Delete these methods because they can cause copy instances of the singleton when using threads.
   VerticesFactory(VerticesFactory const &) = delete;
   void operator=(VerticesFactory const &) = delete;

private:
   /// Constructor is private to keep a singleton instance of this class.
   VerticesFactory();

   /// Pointer to neurons instance
   shared_ptr<IAllNeurons> neuronsInstance;

   string neuronClassName;

   /* Type definitions */
   /// Defines function type for usage in internal map
   typedef IAllNeurons *(*CreateNeuronsFn)(void);

   /// Defines map between class name and corresponding ::Create() function.
   typedef map<string, CreateNeuronsFn> NeuronFunctionMap;

   /// Makes class-to-function map an internal factory member.
   NeuronFunctionMap createFunctions;

   string getVerticeType();

   /// Retrieves and invokes correct ::Create() function.
   IAllNeurons *invokeNeuronsCreateFunction(const string &className);

   /// Register neuron class and it's create function to the factory.
   void registerNeurons(const string &neuronsClassName, CreateNeuronsFn function);
};
