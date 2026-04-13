# Code Formatting Etiquettes

Code is written once and read a thousand times. Hence, it is important we keep our codebase consistent to improve readability. In this section, we will discuss all the coding formatting styles that must be followed while contributing to the Graphitti codebase. For C++ style/feature guide please refer to [C++ design and Coding standards](cppStyleGuide.md).

  * File Extensions:
    - We use `.cpp` and `.h` for our C++ code, and `.cu` for CUDA source files. We name files with *exactly* the same name (including capitalization) as the primary classes they define.
  
  * Indentation:
    - We indent using *three spaces*. *Not tabs*. Spaces.
  
  * Naming convention:
    - We use [cC]amelCase naming, rather than underscores. 
    - Classes start with capital letters; 
    - Functions and variables start with lowercase letters.
  
  * Spaces:
    - We put spaces after list items and method parameters (`f(a, b, c)`, not `f(a,b,c)`) and around operators (`x += 1`, not `x+=1`). 
    - We don't put spaces after or before parentheses (`f(a)`, not `f( a )`).
  
  * Braces:
    - We like to [cuddle our braces](http://blog.gskinner.com/archives/2008/11/curly_braces_to.html), avoiding isolating curly braces on their own lines for loops and conditionals (except for right braces closing a code block, so there's a limit to how cuddly we are); we do put isolated braces on their own lines for functions. 

    - We use braces even when a code block is a single line, to prevent bugs when it (inevitably) later expands to multiple lines.
    
  ```c++
    if (x > m) {
        x--;
    } else {
        x++;
    }
  ```   
  

  ```c++
    int f(a)
    {
        return a;
    }
  ``` 
  
  * Line length:
    - We limit code to 100 character line lengths. You never know when someone will want to print something out on an [ASR-33 teletype](https://en.wikipedia.org/wiki/Teletype_Model_33).

  * Condition checks:
    - We use the following style for checks 
  
        `if (aPointerVar == nullptr)`, not `if (aPointerVar == 0)`; 

        `if (!aBoolFlag)`, not `if (aBoolFlag == false)`; 

        `if (aCharVar == '\0')`, not `if (aCharVar == 0)`.

  * Empty Lines:
    - We use an empty line between methods.
    - We use empty lines around multi-line blocks.
    - We use Unix end-of-line characters (`\n`).
    
  * Header guard:
    - We use `#pragma once` instead of `#define` guards.

## clang-format

To help with adhering to our style guide, we developed a clang-format file to assit in this process.

Future development will include creating an automated workflow to do this for all .cpp and .h files on push or pull requests to master.

### running clang-format through command line

* If you just want to see what the changes will look like without actually overwriting the file then run this command (prints to console)

```sh
clang-format fileName
```

* If you want to run the format file and have it make the changes directly to the file then run this command

```sh
clang-format -i fileName
```

* If you want to make changes to the clang-format file options themselves, then visit the [clang-format options online documentation](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)

### running clang-format inside VSCode

If you want to set clang as the default formatter in VSCode and automatically run clang-format on files as they're saved, you can add this to your settings.json

```json
{
    "editor.defaultFormatter": "xaver.clang-format",
    "clang-format.style": "file",
    "clang-format.executable": "/usr/bin/clang-format", 
    "editor.formatOnSave": true,
    "files.associations": {
        "ostream": "cpp"
    }
}
```

Change the file path to wherever you have installed clang-format

---------
[<< Go back to the Developer Documentation page](index.md)

---------
[<< Go back to the Graphitti home page](../index.md)