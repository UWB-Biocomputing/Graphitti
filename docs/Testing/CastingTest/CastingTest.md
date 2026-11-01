## Dynamic Cast Performance Testing

This program compares the performance of functions which use the dynamic_cast conversion on one of their parameters. 

This experiment was inspired by Graphitti's `AllSTDPSynapses::advanceEdge` function, which is called many times per time step in a loop in `AllEdges::advanceEdges`. The `advanceEdge` function performs a dynamic_cast on one of its passed parameters, a pointer to a base vertex class. It was observed that some performance improvement could be made by casting the pointer once in `advanceEdges` and changing `advanceEdge` to accept the already cast pointer.  

The file sets up a hierarchy of inherited classes named after Graphitti's neuro vertex classes. Pointers are cast to the `AllSpikingNeurons` class to mimic the actual casting done in Graphitti's `advanceEdge`. 

## How to Use

Compile and run the program by running: 

```sh
g++ -o CastingTest CastingTest.cpp
./CastingTest 
```

If you want to change the test method run, edit both definitions of `advanceEdge` to call `test1()` or `test2()` as desired.

## Process

Mimicking the execution of Graphitti, a pointer of base class `IAllVertices` was created and pointed to an instance of a derived class. In order to test different levels of inheritance, this was done twice with derived classes `AllSpikingNeurons` and `AllLIFNeurons`, at inheritance levels 2 and 4 respectively. 


In a loop running 100,000,000 times, the `advanceEdge` function is called in one of two ways:

1. Callee-casting: `advanceEdge` accepts a base class pointer and casts the pointer in its body.

2. Caller-casting: `advanceEdge` accepts a derived class pointer, and the calling function casts the pointer before the loop.

The `advanceEdge` function then calls one of two test methods defined in `AllSpikingNeurons`.

The test methods are designed to do as little as possible so that the execution time is mostly just the time taken by the cast. The reason for two tests being run was for fear that the compiler would optimize the more trivial test away and not even perform the function call or the cast. That doesn't appear to have happened, but both tests were still run and reported for consideration regardless.

## Results

The level of inheritance that the derived class started at made a significant impact. Casting from the lowest-level derived class is consistently slower than from a higher-level derived class. Caller-casting times are minimally affected because only one cast occurs, but callee-casting takes less than half the time when casting from two levels deep as opposed to four levels.

Test1 is overall faster because less is being done in the test method. It also has a larger ratio of calleeTime / callerTime, giving a higher apparent speedup.  

Running with different levels of compiler optimization (-O0, -O1, etc) had no obvious consistent impact on running time. 

## Sample output

#### Running test1:
```
Casting 4 levels from base class:
  Callee-casting time:  3333 ms
  Caller-casting time:   473 ms
  Difference in time:   2860 ms
  Casting in the caller method is 7.04651 times faster than casting in the callee method

Casting 2 levels from base class:
  Callee-casting time:  1380 ms
  Caller-casting time:   473 ms
  Difference in time:    907 ms
  Casting in the caller method is 2.91755 times faster than casting in the callee method
  ```

  #### Running test2:
  ```
  Casting 4 levels from base class:
  Callee-casting time:  3572 ms
  Caller-casting time:   789 ms
  Difference in time:   2783 ms
  Casting in the caller method is 4.52725 times faster than casting in the callee method

Casting 2 levels from base class:
  Callee-casting time:  1620 ms
  Caller-casting time:   788 ms
  Difference in time:    832 ms
  Casting in the caller method is 2.05584 times faster than casting in the callee method
  ```