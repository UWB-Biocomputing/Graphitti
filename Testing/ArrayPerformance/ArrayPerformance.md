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

>No optimization:
>Array initialization took       	43989   µs
>Vector initialization took      	182074  µs
>Valarray initialization took    	58962   µs
>
>Array access and sum took       	22617   µs
>Vector [] access and sum took   	30256   µs
>Vector .at() access and sum took        105913  µs
>Valarray access and sum took    	30225   µs
>
>Array random access and sum took        105570  µs
>Vector [] random access and sum took    142558  µs
>Vector .at() random access and sum took 289157  µs
>Valarray random access and sum took     143044  µs
>
>O1 :
>Array initialization took       	23012   µs
>Vector initialization took      	42519   µs
>Valarray initialization took    	24857   µs
>
>Array access and sum took       	6809    µs
>Vector [] access and sum took   	6338    µs
>Vector .at() access and sum took        6639    µs
>Valarray access and sum took    	6544    µs
>
>Array random access and sum took        73207   µs
>Vector [] random access and sum took    73429   µs
>Vector .at() random access and sum took 73406   µs
>Valarray random access and sum took     73156   µs
>
>O2:
>Array initialization took       	20268   µs
>Vector initialization took      	37518   µs
>Valarray initialization took    	21361   µs
>
>Array access and sum took       	6906    µs
>Vector [] access and sum took   	6638    µs
>Vector .at() access and sum took        6449    µs
>Valarray access and sum took    	6586    µs
>
>Array random access and sum took        73473   µs
>Vector [] random access and sum took    73400   µs
>Vector .at() random access and sum took 73767   µs
>Valarray random access and sum took     75460   µs