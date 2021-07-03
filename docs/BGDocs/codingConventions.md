# Coding Conventions

  * We use `.cpp` and `.h` for our C++ code, and `.cu` for CUDA source files. We name files with *exactly* the same name (including capitalization) as the primary classes they define.
  
  * We indent using *three spaces*. *Not tabs*. Spaces.
  
  * We use [cC]amelCase naming, rather than underscores. Classes start with capital letters; functions and variables start with lower-case letters.
  
  * We put spaces after list items and method parameters (`f(a, b, c)`, not `f(a,b,c)`) and around operators (`x += 1`, not `x+=1`). We don't put spaces after or before parentheses (`f(a)`, not `f( a )`).
  
  * We like to [cuddle our braces](http://blog.gskinner.com/archives/2008/11/curly_braces_to.html), avoiding isolating curly braces on their own lines for loops and conditionals (except for right braces closing a code block, so there's a limit to how cuddly we are); we do put isolated braces on their own lines for functions. Examples:
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
  * We use braces even when a code block is a single line, to prevent bugs when it (inevitably) later expands to multiple lines.

  * We limit code to 80 character line lengths. You never know when someone will want to print something out on an [ASR-33 teletype](https://en.wikipedia.org/wiki/Teletype_Model_33).
  * We help keep code clear to human readers. So:
  
  `if (aPointerVar == nullptr)`, not `if (aPointerVar == 0)`; 

  `if (!aBoolFlag)`, not `if (aBoolFlag == false)`; 

  `if (aCharVar == '\0')`, not `if (aCharVar == 0)`.

  * We use `#pragma once` instead of `#define` guards.
  * We use an empty line between methods.
  * We use empty lines around multi-line blocks.
  * We use Unix end-of-line characters (`\n`).