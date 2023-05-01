# C++ Coding Conventions

C++ is the main development language used in Graphitti. Currently, the code should target C++17, the version targeted will advance over time.

The goal of this guide is to describe in detail the dos and don'ts of writing C++ features that are used in Graphitti. These rules exist to keep the code base manageable while still allowing coders to use C++ language features productively.

### NOTE:

1. The guide is developed using two standard C++ style guides - 
    - [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
    - [C++ Core Guidelines approved by Bjarne Stroustrup & Herb Sutter](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#rfrules-coding-rules)
2. This is a living document and will be updated as Graphitti adopts new C++ features. 
3. For details on features not covered here, refer to the above two guides for the best practice. Please discuss with Professor Stiber and document the feature details here.

## **Use of const and constexpr**

- When variables are preceded by the keyword `const` it indicates the variables will not change once assigned.
- When class functions have the `const` qualifier it indicates the function does not change the state of the class member variables

```c++
 class A { 
    const  int i = 100;

    const int * const num= &i;

    int function1(char c) const; 
    const int* function2(const char* string) const;
};
```
- When variables are preceded by the keyword `constexpr` it indicates the variables are fixed at compilation/link time. 
- When `constexpr` is used in a function it doesn’t promise that the function returns a const value, rather it indicates that- if the parameter passed are compile-time constant then the result returned may be used as a compile-time constant; else it will be computed at run-time.

```c++
 class A { 
    constexpr  int i = 100;

    constexpr int function1(int x, int y);
};

```

Recommendation: 
- It is recommended to use `const` in APIs (i.e., on function parameters, methods, and non-local variables) as it provides consistent, mostly compiler-verified documentation of what objects an operation can change. 
- Using `const` on local variables is neither encouraged nor discouraged.
- - Use `const` to indicate that the value will not change once assigned. `constexpr` is a better choice for some uses of const if the value is known at compile time.
- Use `constexpr` in the function and constructor if the parameters may be known at compile-time.

- We put `const` and `constexpr` first as it is more readable
        `constexpr  int i = 100;` 
        `const  int i = 100;` not  `int const  i = 100;`


## **Copy and Move operations**

Recommendation: 
- A class must make it clear whether it is copyable, move-only, or neither copyable nor movable by explicitly declaring and/or deleting the appropriate operations in the public section of the declaration.
- A copyable class should explicitly declare the copy operations and may also declare move operations in order to support efficient moves.
- A move-only class should explicitly declare the move operations.
- A non-copyable/movable class should explicitly delete the copy operations and a copyable class. 
- Explicitly declaring or deleting all four copy/move operations is encouraged if it improves readability.

```c++
class Copyable {
 public:
  Copyable(const Copyable& other) = default;
  Copyable& operator=(const Copyable& other) = default;
};

class MoveOnly {
 public:
  MoveOnly(MoveOnly&& other) = default;
  MoveOnly& operator=(MoveOnly&& other) = default;
};

class NotCopyable{
 public:
  NotCopyable(const NotCopyable&) = delete;
  NotCopyable& operator=(const NotCopyable&)= delete;
};

class NotMovable{
 public:
  NotMovable(NotMovable&&) = delete;
  NotMovable& operator=(NotMovable&&)= delete;
};

```
- Note that if you explicitly declare or delete either the constructor or assignment operation for copy, the other copy operation is not obvious and must be declared or deleted. Likewise for move operations.


## **Smart Pointers**

- Smart pointers unlike raw pointers are used to make sure that an object is deleted automatically when it goes out of scope.
- There are three smart pointers- `unique_ptr`, `shared_ptr`, and `weak_ptr`. Only `unique_ptr` and `shared_ptr` are used in Graphitti.
- `unique_ptr` make sure that only one copy of an object exists and expresses exclusive ownership of that dynamically allocated object; the object is deleted when the `unique_ptr` goes out of scope. It cannot be copied, but can be moved to represent ownership transfer.
- A `unique_ptr` can be initialized with a pointer upon creation or it can be created without a pointer and assigned one later.

```c++
  std::unique_ptr<int> value1(new int(10)); 

  // OR 

  std::unique_ptr<int> value2; 
  value2.reset(new int(47));
```
- `shared_ptr` is a smart pointer that expresses shared ownership of an object. `shared_ptr` can be copied; ownership of the object is shared among all copies, and the object is deleted when the last `shared_ptr` is destroyed.

Recommendation: 
- Use Smart pointers where possible as it can improve readability by making ownership logic explicit, self-documenting, and unambiguous. It can also eliminate manual ownership bookkeeping, simplifying the code and ruling out large classes of errors.
- If dynamic allocation is necessary, prefer to keep ownership with the code that allocated it by using `uniqe_ptr`. If other code needs access to the object, consider passing it a copy, or passing a pointer or reference without transferring ownership. 
- Avoid using `shared_ptr` without a very good reason. One such reason is to avoid expensive copy operations and it's explicit bookkeeping at run-time, which can be costly. In some cases (e.g., cyclic references), objects with shared ownership may never be deleted.
- Only use `shared_ptr` if the performance benefits are significant, and the underlying object is immutable. 
- Never use `auto_ptr`. Instead, use `unique_ptr`.

## **Aliases**
- Prefer using `using` over `typedef` as it provides a more consistent syntax with the rest of C++ and works with templates.

## **Inputs and Outputs**
- Prefer to return by value or, failing that, return by reference. Avoid returning a pointer unless it can be null.
- Parameters are either inputs to the function, outputs from the function, or both. Non-optional input parameters should usually be values or const references, while non-optional output and input/output parameters should usually be references (which cannot be null). 
- Use `optional` to represent optional by-value inputs, and use a `const` pointer when the non-optional form would have used a reference. Use non-const pointers to represent `optional` outputs and `optional` input/output parameters.
- Use `optional` to express “value-or-not-value”, or “possibly an answer”, or “object with delayed initialization”, as it increases the level of abstraction, making it easier for others to understand what your code is doing. 

## **Namespaces**
TODO

---------
[<< Go back to the Coding Conventions page](codingConventions.md)

---------
[<< Go back to the Developer Documentation page](index.md)

---------
[<< Go back to the Graphitti home page](../index.md)