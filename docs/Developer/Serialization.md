# Serialization and Deserialization using Cereal

This guide explains how to implement serialization using the Cereal library within Graphitti. If you're looking to add serialization to your class, follow this guide. For more comprehensive information on Cereal, refer to their [official documentation](https://uscilab.github.io/cereal/index.html).

<strong>
What is Serialization and Deserialization? 
</strong>

Serialization involves converting an object or data structure into a format that can be stored or transmitted, while deserialization is the reverse process— reconstructing an object from the serialized format.
<br>

<strong>
What is Cereal?
</strong>

Cereal is a lightweight C++11 library designed for object serialization and deserialization. It provides a straightforward interface for serializing objects and supports a wide range of data types. Cereal is efficient and well-suited for handling large data sets, making it a preferred choice for serialization tasks in Graphitti.
<br>


<strong>
Why Graphitti Uses Serialization?
</strong>

Graphitti utilizes Cereal to enable efficient serialization and deserialization of its simulation state and network structure. Serialized data can serve as a checkpoint for large simulations or as input for subsequent simulations with varying conditions. This flexibility enhances Graphitti's efficiency and adaptability in modeling scenarios.
<br>

## Understand Cereal at a High Level

### C++ Features Supported by Cereal

- **Standard Library Support**: Cereal fully supports the C++11  [standard library](http://en.cppreference.com/w/). You can include the necessary headers from Cereal to enable support for various types (e.g., If you need to serialize a `std::vector`, add `<cereal/types/vector.hpp>`). Refer to the complete list of supported types below or in [Cereal Doxygen docs](https://uscilab.github.io/cereal/assets/doxygen/group__STLSupport.html).

- **Smart Pointers**: Cereal supports modern smart pointers (`std::shared_ptr` and `std::unique_ptr`) via `<cereal/types/memory.hpp>`. <strong> However, raw pointers or references are NOT supported.</strong>

- **Inheritance and Polymorphism**: Cereal can seamlessly handles inheritance and polymorphism- more on this in the coming section.

### Serialization Archive Types

- **Supported Archive Types**: Cereal provides three basic archive types: binary (with a portable version), XML, and JSON. Graphitti primarily utilizes binary and XML archives, managed within the `Serializer.cpp` class.

- **RAII Handling**: Cereal archives are designed to be used in a Resource Acquisition Is Initialization (RAII) manner and flush their contents only upon destruction.

- **Serialization Order**: By default, Cereal deserializes data in the order it was serialized.

### Serialization Functions in Cereal

- **Defining Serialization Functions**: Cereal requires to know which data members to serialize in a class. By implementing a serialization function `serialize` within a class, you indicate to Cereal which data members should be serialized. 

- **Default Constructor Requirement**: Cereal requires access to a default constructor for classes it serializes to construct the object during deserialization.

- **Name-Value Pairs**: Cereal supports name-value pairs, allowing you to attach names to serialized objects. This feature is particularly useful for XML archives and is adopted in Graphitti's serialization process.

- **Class Versioning**: While still in work in progress in Graphitti, Cereal supports class versioning, enabling compatibility between different versions of serialized objects.

## Incorporate Cereal in your class 

- Cereal supports two approaches for serialization functions: internal or external. Cereal also provides two types of serialization function `serialize` and `load and save`. For consistency within Graphitti, use an internal single `serialize` function.
- Throughout Graphitti, we typically serialize all data members (private, protected, and public) of a class.

### **STEP 01: ADD CEREAL HEADERS**
Before implementing serialization in your class, you need to include the appropriate Cereal headers for the types of data members you want to serialize. Cereal provides headers for various standard types, such as vectors, strings, and other containers.

- For example, if you're serializing a `std::vector<int>`, you'll need to include the following header:
```cpp
#include <cereal/types/vector.hpp>
```

- If you are using custom types, ensure the serialization process is correctly implemented for those types as well. The same approach applies to user-defined types, so include the appropriate headers and define the serialize function in that class or struct.

<details>

<summary><strong>Commonly used C++ data types and their corresponding Cereal headers</strong></summary>


| Type                | Header to include                               |
|:--------------------|:----------------------------------------------|
| `std::array`        | `#include <cereal/types/array.hpp>`            |
| `std::atomic`       | `#include <cereal/types/atomic.hpp>`           |
| `std::bitset`       | `#include <cereal/types/bitset.hpp>`           |
| `std::chrono`       | `#include <cereal/types/chrono.hpp>`           |
| `std::complex`      | `#include <cereal/types/complex.hpp>`          |
| `std::deque`        | `#include <cereal/types/deque.hpp>`            |
| `std::forward_list` | `#include <cereal/types/forward_list.hpp>`     |
| `std::functional`   | `#include <cereal/types/functional.hpp>`       |
| `std::list`         | `#include <cereal/types/list.hpp>`             |
| `std::map`          | `#include <cereal/types/map.hpp>`              |
| `std::memory`       | `#include <cereal/types/memory.hpp>`           |
| `std::optional`     | `#include <cereal/types/optional.hpp>`         |
| `std::queue`        | `#include <cereal/types/queue.hpp>`            |
| `std::set`          | `#include <cereal/types/set.hpp>`              |
| `std::stack`        | `#include <cereal/types/stack.hpp>`            |
| `std::string`       | `#include <cereal/types/string.hpp>`           |
| `std::tuple`        | `#include <cereal/types/tuple.hpp>`            |
| `std::unordered_map`| `#include <cereal/types/unordered_map.hpp>`    |
| `std::unordered_set`| `#include <cereal/types/unordered_set.hpp>`    |
| `std::utility`      | `#include <cereal/types/utility.hpp>`          |
| `std::valarray`     | `#include <cereal/types/valarray.hpp>`         |
| `std::variant`      | `#include <cereal/types/variant.hpp>`          |
| `std::vector`       | `#include <cereal/types/vector.hpp>`           |

</details>

<br>

### **STEP 02: ADD SERIALIZE FUNCTION**

Within your class header file
- Firstly, declare the `serialize` function inside the class:
    - Add the following template function signature in the **public** section of your class to allow serialization for any archive type.
```cpp 
    template <class Archive> 
    void serialize(Archive & archive); 
```  
- Secondly, define the `serialize` function outside the class:
    - After your class declaration, define the serialize function at the end of the header file. 
    - This function specifies which member variables are serialized and deserialized, and how that process occurs.
```cpp
    template <class Archive> 
    void YOUR_CLASS_NAME::serialize(Archive &archive)
    {
        archive(ADD_YOUR_MEMBER_VARIABLES_HERE);
        // Refer to the example below
    }
```
Here’s a sample implementation to guide you:

``` cpp

// STEP 01: Add Necessary Header
#include <cereal/vector.hpp>    

class MyCoolClass
{

  public:
    // STEP 02 (a): Declare the serialize function in the public section
    template <class Archive> 
    void serialize( Archive & archive );    

  private:
    std::vector<int> myVector_;
    int x_;
};

//STEP 02 (b): Define the serialize function outside the class

template <class Archive> 
void MyCoolClass::serialize(Archive &archive)
{
   archive(cereal::make_nvp("myVector", myVector_), cereal::make_nvp("myInt", x_));
}


```

Adjust the function names and data member names as per your specific requirements.

NOTE:

- The `template <class Archive>` declaration allows the serialize function to be flexible, enabling it to work with different types of Cereal archives, such as JSON, XML, or binary formats.

- When defining the `serialize` function, use `make_nvp()` and `CEREAL_NVP()` for each member variable:
  - `make_nvp()` is used when you want to assign custom names to your serialized member variables, which can be helpful for clarity in the serialized output.
  - `CEREAL_NVP()` automatically uses the variable's name for serialization without the need to explicitly name it.

- Why define `serialize` in the header?
  - Cereal relies heavily on templates, and C++ templates require full implementation details to be available during compilation for proper instantiation. Since templates must be instantiated at compile time, placing the serialize function in a `.cpp` file could result in missing template information, leading to linker errors. By defining the function in the header file, the compiler has all the necessary information to properly instantiate the serialize function for various data types.

- Defining the function outside the class (but still in the header) promotes a clean code style, making the class declaration less cluttered and easier to maintain. It also makes serialized code easier to find, which is especially important in larger projects like Graphitti.

### **STEP 03: SPECIAL CASES**

Cereal requires additional steps for certain special cases such as inheritance, polymorphism, and templates. In this section, we outline specific steps based on whether your class matches one of these conditions. If you answer "yes" to any of the following questions, follow the corresponding steps. Some classes may fall under multiple categories, so be sure to review all the details carefully.

### **1. DERIVED CLASS?**
This step explains how to serialize base classes in a derived class. Cereal requires a path from the derived to the base type(s), typically done with `cereal::base_class` or `cereal::virtual_base_class`. 

<details>
<summary><strong> Virtual Inheritance ? </summary></strong>
If your derived class uses virtual inheritance (`class Derived : virtual Base`), use `cereal::virtual_base_class<BaseT>(this)` to cast the derived class to its base class. Ensure this is placed at the start of the `archive` in the `serialize` function before member variables in the derived class.

```cpp

class MyDerived : virtual MyBase
{
  int x_;
  template <class Archive>
  void serialize( Archive & ar );
};

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  { 
    // We pass this cast to the base type for each base type we need to serialize. 
    archive(cereal::virtual_base_class<MyBase>(this), cereal::make_nvp("myInt", x_)); 

    // For multiple inheritance, link all the base classes one after the other
    //archive(cereal::virtual_base_class<MyBase1>(this), cereal::virtual_base_class<MyBase2>(this), cereal::make_nvp("myInt", X_));
  }
```
</details>

<details>
<summary><strong> Normal Inheritance ?</summary></strong>
For non-virtual inheritance (`class Derived : public Base`), use `cereal::base_class<BaseT>(this) to serialize the base class. Ensure this is placed at the start of the `archive` in the `serialize` function before member variables in the derived class.

```cpp

class MyDerived : public MyBase
{
  int x_;
  template <class Archive>
  void serialize( Archive & ar );
};

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  { 
    // We pass this cast to the base type for each base type we need to serialize. 
    archive(cereal::base_class<MyBase>(this), cereal::make_nvp("myInt", x_)); 

    // For multiple inheritance, link all the base classes one after the other
    //archive(cereal::base_class<MyBase1>(this), cereal::base_class<MyBase2>(this), cereal::make_nvp("myInt", X_));
  }
```
For more details, refer to the official Cereal documentation on [inheritance](https://uscilab.github.io/cereal/inheritance.html)

</details>

### **2. EXHIBIT POLYMORPHISM?**

If you answered "yes" to the previous question about your class being a derived class, this is likely "yes" as well.

<details>
<summary><strong>
Follow these steps if your class exhibits polymorphic behavior:
</summary></strong>

1. Include Necessary Headers:

Make sure to include the polymorphic header to enable support for polymorphism in Cereal in the derived class.
``` #include <cereal/types/polymorphic.hpp> ```

2. Register Your Derived Types:

Register each derived class above the definition of the `serialize` function using `CEREAL_REGISTER_TYPE(DerivedClassName)` in the respective derived class.

```cpp
// be sure to include support for polymorphism
#include <cereal/types/polymorphic.hpp> 

class MyDerived : public MyBase
{
  int x_;
  template <class Archive>
  void serialize( Archive & ar );
};

//Registering the Derived class
CEREAL_REGISTER_TYPE(MyDerived);

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  { 
    archive(cereal::base_class<MyBase>(this), cereal::make_nvp("myInt", x_)); 
  }
```

3. Register Your Base Class (if not registered automatically):

Normally, registering base classes is handled automatically if you serialize a derived type with either `cereal::base_class` or `cereal::virtual_base_class`. However, in situations where neither of these is used, explicit registration is required using the `CEREAL_REGISTER_POLYMORPHIC_RELATION` macro in the derived class. 

```cpp
struct MyEmptyBase
{
  virtual void foo() = 0;
};

struct MyDerived: MyEmptyBase
{
  void foo() {}
  double y_;
  template <class Archive>
  void serialize( Archive & archive );
};

CEREAL_REGISTER_TYPE(MyDerived)

//Registering the Base Class
CEREAL_REGISTER_POLYMORPHIC_RELATION(MyEmptyBase, MyDerived)

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  {
    archive( cereal::make_nvp("myDouble", y_) );
  }
```

For more detailed information and examples on polymorphism in Cereal, refer to the official documentation on [Polymorphism](https://uscilab.github.io/cereal/polymorphism.html).

</details>

### **3. TEMPLATE?**
<details>
<summary><strong>
Template involves inheritance?
</summary></strong>

Follow all the steps from STEP 01 as if your class is a regular class. However, if the template involves inheritance, you might need to register all potential instantiations of the template during polymorphism handling.

```cpp

// Include necessary Cereal headers
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>  

// A pure virtual base class
struct BaseClass
{
    virtual void sayType() = 0;
};

// A templated class derived from BaseClass
template <typename T>
struct DerivedClassTemplate : public BaseClass
{
    T value_;
    void sayType();

    template <class Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::virtual_base_class<MyEmptyBase>(this), cereal::make_nvp("myValue", value_));
    }
};

// Register template instantiations
CEREAL_REGISTER_TYPE(DerivedClassTemplate<int>);
CEREAL_REGISTER_TYPE(DerivedClassTemplate<float>);

// If using Register polymorphic relationships
// CEREAL_REGISTER_POLYMORPHIC_RELATION(BaseClass, DerivedClassTemplate<int>);
// CEREAL_REGISTER_POLYMORPHIC_RELATION(BaseClass, DerivedClassTemplate<float>);

```
</details>

### **4. NO DEFAULT CONSTRUCTOR?**

Cereal provides a special overload method to handle this situation. Refer to the [Cereal documentation](https://uscilab.github.io/cereal/pointers) for detailed information on this technique.

## Common Cereal Errors

Encountering a Cereal error during compiling or running Graphitti? Here's a checklist to troubleshoot:

1. **Include Correct Cereal Headers**: Ensure you've included the necessary Cereal headers for the types you're serializing. If using polymorphic serialization, include `#include <cereal/types/polymorphic.hpp>`.

2. **Default Constructor**: Verify that your class has a default constructor. If not possible, utilize Cereal's special overload methods for handling this scenario.

3. **Polymorphic Type Registration**: If serialization of polymorphic types fails or results in incorrect type information, double-check your type registration. Use `CEREAL_REGISTER_TYPE` and `CEREAL_REGISTER_POLYMORPHIC_RELATION` to register polymorphic types correctly.

4. **Runtime Exceptions**: If encountering a runtime exception like 
    ```
    what():  Trying to save an unregistered polymorphic type (AllDSSynapses).
    Make sure your type is registered with CEREAL_REGISTER_TYPE and that the archive you are using was included (and registered with CEREAL_REGISTER_ARCHIVE) prior to calling CEREAL_REGISTER_TYPE.
    ```
    include the following two archive headers for the respective class:

    ```
    #include <cereal/archives/portable_binary.hpp>
    #include <cereal/archives/xml.hpp> 
    ```
    Reasoning: Polymorphic type registration requires mapping your registered type to archives included prior to CEREAL_REGISTER_TYPE being called. Missing archive headers in certain classes could lead to this error.
    
With these checks, you should be able to diagnose and resolve common Cereal errors in Graphitti.

---------
[<< Go back to the Developer Documentation page](index.md)

---------
[<< Go back to the Graphitti home page](../index.md)