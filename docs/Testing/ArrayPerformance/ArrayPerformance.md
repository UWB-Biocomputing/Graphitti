## Array Performance Testing

This file tests the efficency of C++ arrays, Vectors, and Valarrays.  Each is tested in four areas: Initialization, Filling the array with zeroes, Sequential access of each element in the array, and Random access of each element in the array.  It also tests different methods such as Vector [] with .at() for accessing elements.

## How to Use

Compile the program by running (replace the X with 0-4 for different compiler optimization levels)

> g++ -OX -o arraytest arraySpeedTest.cpp 

Then run the test using

> ./arraytest

## Results

When used properly, Array, Vector, and Valarray have very similar performance.  Built-in compiler optimization makes a huge difference in speed.  If you know the size of your data beforehand, then reserve the size of the array before using it.  Avoid functions that perform bounds checking.

## Sample Output
### Initialization
|                           | Array       | Vector      | Valarray    |
|---------------------------|-------------|-------------|-------------|
| No optimization           | 43989 µs    | 182074 µs   | 58962 µs    |
| O1                        | 23012 µs    | 42519 µs    | 24857 µs    |
| O2                        | 20268 µs    | 37518 µs    | 21361 µs    |

### Access and Sum
|                           | Array       | Vector []   | Vector .at() | Valarray    |
|---------------------------|-------------|-------------|--------------|-------------|
| No optimization           | 22617 µs    | 30256 µs    | 105913 µs    | 30225 µs    |
| O1                        | 6809 µs     | 6338 µs     | 6639 µs      | 6544 µs     |
| O2                        | 6906 µs     | 6638 µs     | 6449 µs      | 6586 µs     |

### Random Access and Sum
|                           | Array       | Vector []   | Vector .at() | Valarray    |
|---------------------------|-------------|-------------|--------------|-------------|
| No optimization           | 105570 µs   | 142558 µs   | 289157 µs    | 143044 µs   |
| O1                        | 73207 µs    | 73429 µs    | 73406 µs     | 73156 µs    |
| O2                        | 73473 µs    | 73400 µs    | 73767 µs     | 75460 µs    |

