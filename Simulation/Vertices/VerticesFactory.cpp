/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#include "VerticiesFactory.h"

#include "ParameterManager.h"
#include "AllLIFNeurons.h"
#include "AllIZHNeurons.h"

/// Constructor is private to keep a singleton instance of this class.
VerticesFactory::VerticesFactory() {
   // register neurons classes
   registerNeurons("AllLIFNeurons", &AllLIFNeurons::Create);
   registerNeurons("AllIZHNeurons", &AllIZHNeurons::Create);
}

VerticesFactory::~VerticesFactory() {
   createFunctions.clear();
}

/*
 *  Register neurons class and its creation function to the factory.
 *
 *  @param  neuronsClassName  neurons class name.
 *  @param  Pointer to the class creation function.
 */
void VerticesFactory::registerNeurons(const string &neuronsClassName, CreateNeuronsFn function) {
   createFunctions[neuronsClassName] = function;
}


/// Creates concrete instance of the desired neurons class.
shared_ptr<IAllNeurons> VerticesFactory::createNeurons(const string &className) {
   neuronsInstance = shared_ptr<IAllNeurons>(invokeNeuronsCreateFunction(className));
   return neuronsInstance;
}

/**
 * Create an instance of the neurons class using the static ::Create() method.
 *
 * The calling method uses this retrieval mechanism in 
 * value assignment.
 */
IAllNeurons *VerticesFactory::invokeNeuronsCreateFunction(const string &className) {
   for (auto i = createFunctions.begin(); i != createFunctions.end(); ++i) {
      if (className == i->first)
         return i->second();
   }
   return NULL;
}
