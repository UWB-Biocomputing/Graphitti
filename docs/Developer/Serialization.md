# Serialization and Deserialization using Cereal

This guide explains how to implement serialization using the Cereal library within Graphitti. If you're looking to add serialization to your class, follow this guide. For more comprehensive information on Cereal, refer to their [official documentation](https://uscilab.github.io/cereal/index.html).

<details>
<summary><strong>
Understanding Serialization and Deserialization 
</strong></summary>

Serialization involves converting an object or data structure into a format that can be stored or transmitted, while deserialization is the reverse process— reconstructing an object from the serialized format.
</details>
<br>
<details>
<summary><strong>
What is Cereal?
</strong></summary>

Cereal is a lightweight C++11 library designed for object serialization and deserialization. It provides a straightforward interface for serializing objects and supports a wide range of data types. Cereal is efficient and well-suited for handling large data sets, making it a preferred choice for serialization tasks in Graphitti.
</details>
<br>

<details>
<summary><strong>
Why Graphitti Uses Serialization?
</strong></summary>

Graphitti utilizes Cereal to enable efficient serialization and deserialization of the network structure. Serialized data can serve as a checkpoint for large simulations or as input for subsequent simulations with varying conditions. This flexibility enhances Graphitti's efficiency and adaptability in modeling scenarios.
</details>
<br>

## Basics of Cereal

### C++ Features Supported by Cereal

- **Standard Library Support**: Cereal fully supports the C++11  [standard library](http://en.cppreference.com/w/). You can include the necessary headers from Cereal to enable support for various types (e.g., `<cereal/types/vector.hpp>`). Refer to the [Cereal Doxygen docs](https://uscilab.github.io/cereal/assets/doxygen/group__STLSupport.html) for a complete list of supported types.

- **Smart Pointers**: Cereal supports modern smart pointers (`std::shared_ptr` and `std::unique_ptr`) via `<cereal/types/memory.hpp>`. However, raw pointers or references are not supported.

- **Inheritance and Polymorphism**: Cereal can seamlessly handles inheritance and polymorphism- more on this in the coming section.

### Serialization Archive Types

- **Supported Archive Types**: Cereal provides three basic archive types: binary (with a portable version), XML, and JSON. Graphitti primarily utilizes binary and XML archives, managed within the `Serializer.cpp` class.

- **RAII Handling**: Cereal archives are designed to be used in a Resource Acquisition Is Initialization (RAII) manner and flush their contents only upon destruction.

- **Serialization Order**: By default, Cereal deserializes data in the order it was serialized.

### Serialization Functions in Cereal

- **Defining Serialization Functions**: Cereal requires to know which data members to serialize in a class. By implementing a serialization function `serialize` within a class, you indicate to Cereal which data members should be serialized. 

- **Default Constructor Requirement**:
    - Cereal requires access to a default constructor for classes it serializes to construct the object during deserialization.

- **Name-Value Pairs**: Cereal supports name-value pairs, allowing you to attach names to serialized objects. This feature is particularly useful for XML archives and is adopted in Graphitti's serialization process.

- **Class Versioning**: While still in work in progress in Graphitti, Cereal supports class versioning, enabling compatibility between different versions of serialized objects.

## Incorporate Cereal in your class 

- Cereal supports two approaches for serialization functions: internal or external. Cereal also provides two types of serialization function `serialize` and `load and save`. For consistency within Graphitti, it's recommended to use an internal single `serialize` function.

- Throughout Graphitti, we typically serialize all data members (private, protected, and public) of a class.
<br>

### **Step 01: Include Necessary Cereal Headers**
 Include the necessary Cereal headers for the data member types you want to serialize. If you are using a custom data member type, follow these steps for the custom data type class or struct as well.

<details>

<summary>Refer to the table below for a reference on commonly used C++ data types and their corresponding Cereal headers</summary>


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

### **Step 02: Include and Define the Serialize Function**

Within your class header file
1. Declare the `serialize` function under the public section. 
2. At the end of your header file, define the `serialize` function for your class.

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
    int X_;
};

//STEP 02 (b): Define the serialize function at the bottom of the header file
template <class Archive> 
void MyCoolClass::serialize(Archive &archive)
{
   archive(cereal::make_nvp("myVector", myVector_), cereal::make_nvp("X", X_));
}


```

Adjust the function names and data member names as per your specific requirements.

### **Step 03: Check for any Special Modifications**

If you answer "yes" to any of the following questions, follow the corresponding steps. Some classes may fall under multiple categories, so review all questions carefully.

<details>
<summary><strong>
Is your class a derived class?
</summary></strong>
Cereal needs a serialization path from the derived class to the base type(s). This is usually handled with either `cereal::base_class` or `cereal::virtual_base_class`. 

#### [a] Is your class derived from a virtual inheritance ?
When inheriting from objects from a virtual inheritance (e.g.`class Derived : virtual Base`), the recommended method is to utilize `cereal::virtual_base_class<BaseT>(this)`  to cast the derived class to the base class. 

```cpp

class MyDerived : virtual MyBase
{
  int y;
  template <class Archive>
  void serialize( Archive & ar );
};

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  { 
    // We pass this cast to the base type for each base type we need to serialize. 
    archive(cereal::virtual_base_class<MyBase>(this), y); 

    // For multiple inheritance, link all the base classes one after the other
    //archive(cereal::virtual_base_class<MyBase1>(this), cereal::virtual_base_class<MyBase2>(this), y);
  }
```
#### [b] Is your class derived from a normal (non-virtual) inheritance ?
When inheriting from objects without using virtual inheritance (e.g.`class Derived : public Base`), the recommended method is to utilize `cereal::base_class<BaseT>(this)` to cast the derived class to the base class. 

```cpp

class MyDerived : public MyBase
{
  int y;
  template <class Archive>
  void serialize( Archive & ar );
};

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  { 
    // We pass this cast to the base type for each base type we need to serialize. 
    archive(cereal::base_class<MyBase>(this), y); 

    // For multiple inheritance, link all the base classes one after the other
    //archive(cereal::base_class<MyBase1>(this), cereal::base_class<MyBase2>(this), y);
  }
```
For more details, refer to the official Cereal documentation on [inheritance](https://uscilab.github.io/cereal/inheritance.html)

</details>
<br>

<details>
<summary><strong>
Does your class exhibit polymorphic behavior?
</summary></strong>

If you answered "yes" to the previous question about your class being a derived class, this is likely "yes" as well.

If your class exhibits polymorphic behavior, particularly if it's a derived class, follow these steps:

1. Include Necessary Headers:
Make sure to include the polymorphic header to enable support for polymorphism in Cereal.
``` #include <cereal/types/polymorphic.hpp> ```

2. Register Your Derived Types:
Register each derived class using `CEREAL_REGISTER_TYPE(DerivedClassName)`.

```cpp
// be sure to include support for polymorphism
#include <cereal/types/polymorphic.hpp> 

class MyDerived : public MyBase
{
  int y;
  template <class Archive>
  void serialize( Archive & ar );
};

//Registering the Derived class
CEREAL_REGISTER_TYPE(MyDerived);

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  { 
    archive(cereal::base_class<MyBase>(this), y); 
  }
```

3. Register Your Base Class (if not registered automatically):
Normally, registering base classes is handled automatically if you serialize a derived type with either `cereal::base_class` or `cereal::virtual_base_class`. However, in situations where neither of these is used, explicit registration is required using the `CEREAL_REGISTER_POLYMORPHIC_RELATION` macro.

```cpp
struct MyEmptyBase
{
  virtual void foo() = 0;
};

struct MyDerived: MyEmptyBase
{
  void foo() {}
  double y;
  template <class Archive>
  void serialize( Archive & archive );
};

CEREAL_REGISTER_TYPE(MyDerived)

//Registering the Base Class
CEREAL_REGISTER_POLYMORPHIC_RELATION(MyEmptyBase, MyDerived)

template <class Archive>
  void MyDerived::serialize( Archive & archive )
  {
    archive( y );
  }
```

For more detailed information and examples on polymorphism in Cereal, refer to the official documentation on [Polymorphism](https://uscilab.github.io/cereal/polymorphism.html).

</details>
<br>

<details>
<summary><strong>
Is your class a template?
</summary></strong>

Follow all the steps from Step 01 as if your class is a regular class. However, if the template involves inheritance, you might need to register all potential instantiations of the template during polymorphism handling.

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
    T value;
    void sayType();

    template <class Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::virtual_base_class<MyEmptyBase>(this), value);
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
<br>

<details>
<summary><strong>
Are you serializing smart pointers to objects that do not have a default constructor?
</summary></strong>

Cereal provides a special overload method to handle this situation. Refer to the [Cereal documentation](https://uscilab.github.io/cereal/pointers) for detailed information on this technique.

</details>
<br>

## Debugging Cereal Errors in Graphitti

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

## Serialization flow in Graphitti (For Debugging purposes) 



---------
[<< Go back to the Developer Documentation page](index.md)

---------
[<< Go back to the Graphitti home page](../index.md)