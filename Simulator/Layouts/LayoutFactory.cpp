/**
 * @file LayoutFactory.cpp
 *
 * @ingroup Simulator/Layouts
 * 
 * @brief  A factory class for creating Layout objects.
 */

#include "LayoutFactory.h"
#include "DynamicLayout.h"
#include "FixedLayout.h"
#include "Layout911.h"

/// Constructor is private to keep a singleton instance of this class.
LayoutFactory::LayoutFactory()
{
   // register layout classes
   registerClass("FixedLayout", &FixedLayout::Create);
   registerClass("DynamicLayout", &DynamicLayout::Create);
   registerClass("Layout911", &Layout911::Create);
}

LayoutFactory::~LayoutFactory()
{
   createFunctions.clear();
}

///  Register layout class and its creation function to the factory.
///
///  @param  className  Layout class name.
///  @param  function Pointer to the class creation function.
void LayoutFactory::registerClass(const string &className, CreateFunction function)
{
   createFunctions[className] = function;
}


/// Creates concrete instance of the desired layout class.
///
/// @param className Layout class name.
/// @return Shared pointer to layout instance if className is found in
///         createFunctions map, nullptr otherwise.
shared_ptr<Layout> LayoutFactory::createLayout(const string &className)
{
   auto createLayoutIter = createFunctions.find(className);
   if (createLayoutIter != createFunctions.end()) {
      layoutInstance = shared_ptr<Layout>(createLayoutIter->second());
   } else {
      layoutInstance = nullptr;
   }

   return layoutInstance;
}
