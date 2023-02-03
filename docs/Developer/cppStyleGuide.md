# C++ Coding Conventions

C++ is the main development language used in Graphitti. Currently, the code should target C++17, the version targeted will advance over time.

The goal of this guide is to describe in detail the dos and don'ts of writing C++ features that are used in Graphitti. These rules exist to keep the code base manageable while still allowing coders to use C++ language features productively.

### NOTE:

1. This guide is not a C++ tutorial.
2. The guide is developed using two standard C++ style guides - 
    - [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
    - [C++ Core Guidelines approved by Bjarne Stroustrup & Herb Sutter](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#rfrules-coding-rules)
3. This is a living document and will be updated as Graphitti adopts new C++ features. 
4. For details on features not covered here, refer to the above two guides for the best practice. Please discuss with Professor Stiber and document the feature details here.

## C++ Features

1. **Use of const and constexpr**

- Use `const` to indicate that the value will not change once assigned. `constexpr` is a better choice for some uses of const if the value is known at compile time.
- When variables and parameters are preceded by the keyword `const` it indicates the variables are not changed. 
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
- Use `constexpr` if the variable can be assigned at compile-time.
- Use `constexpr` in the function and constructor if the parameters may be known at compile-time.

- We put `const` and `constexpr` first as it is more readable
        `constexpr  int i = 100;` 
        `const  int i = 100;` not  `int const  i = 100;`



---------
[<< Go back to the Coding Conventions page](codingConventions.md)

---------
[<< Go back to the Developer Documentation page](index.md)

---------
[<< Go back to the Graphitti home page](../index.md)