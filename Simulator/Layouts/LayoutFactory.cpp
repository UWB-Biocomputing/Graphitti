/**
 * @file LayoutFactory.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief  A factory class for creating Layout objects.
 */

#include "LayoutFactory.h"
#include "FixedLayout.h"
#include "DynamicLayout.h"
#include "Layout911.h"

/// Constructor is private to keep a singleton instance of this class.
LayoutFactory::LayoutFactory() {
   // register layout classes
   registerClass("FixedLayout", &FixedLayout::Create);
   registerClass("DynamicLayout", &DynamicLayout::Create);
   registerClass("Layout911", &Layout911::Create);
}

LayoutFactory::~LayoutFactory() {
   createFunctions.clear();
}

///  Register layout class and its creation function to the factory.
///
///  @param  className  vertices class name.
///  @param  Pointer to the class creation function.
void LayoutFactory::registerClass(const string &className, CreateFunction function) {
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired layout class.
shared_ptr<Layout> LayoutFactory::createLayout(const string &className) {
   layoutInstance = shared_ptr<Layout>(invokeCreateFunction(className));
   return layoutInstance;
}

/// Create an instance of the layout class using the static ::Create() method.
///
/// The calling method uses this retrieval mechanism in
/// value assignment.
Layout *LayoutFactory::invokeCreateFunction(const string &className) {
   for (auto i = createFunctions.begin(); i != createFunctions.end(); ++i) {
      if (className == i->first)
         return i->second();
   }
   return nullptr;
}
