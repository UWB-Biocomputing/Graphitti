#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace std;

typedef chrono::high_resolution_clock Clock;

int totalEdgeCount_ = 100000000;
int arrayLength = 2;
int* r_array;

// base class
class IAllVertices {
public:
    virtual ~IAllVertices() {}

    virtual int test1() = 0;
    virtual int test2() = 0;
};

// derived class
class AllVertices : public IAllVertices {
public:
    virtual ~AllVertices() {};
};

// derived class. the class that gets cast to
class AllSpikingNeurons : public AllVertices {
public:
    virtual ~AllSpikingNeurons() {};

    // trivial test method
    virtual int test1() override {
        return 1;
    };

    // less trivial test method
    virtual int test2() override {
        int count = 0;
        for (int i = 0; i < arrayLength; i++) {
            count += r_array[i];
        }
        return count;
    }
};

// lower level derived class
class AllIFNeurons : public AllSpikingNeurons {
public:
    virtual ~AllIFNeurons() {};
};

// lowest level derived class
class AllLIFNeurons : public AllIFNeurons {
public:
    virtual ~AllLIFNeurons() {};
};



// caller casting
void advanceEdge(AllSpikingNeurons* neurons) {
    int result = neurons->test1();
}

// callee casting
void advanceEdge(IAllVertices* neurons) {
    AllSpikingNeurons* spNeurons = dynamic_cast<AllSpikingNeurons*>(neurons);
    int result = spNeurons->test1();
}


// create, populate, and return an array of random numbers
int* makeRandomArray() {
    // fill random array
    random_device rd;  // seeds the rng
    mt19937 gen(rd()); // mersenne twister seeded with rd()
    uniform_int_distribution<> distrib(0, (arrayLength - 1));  // range of output
    int* randomArray = new int[arrayLength];
    for (unsigned int i = 0; i < arrayLength; ++i)
        randomArray[i] = distrib(gen);
    return randomArray;
}

void runTests(IAllVertices* base, int& calleeTime, int& callerTime) {
    // time the method's execution when callee function casts each time it's called
    auto clockStart = Clock::now();
    for (int i = 0; i < totalEdgeCount_; i++) {
        advanceEdge(base);
    }
    auto clockEnd = Clock::now();
    calleeTime = chrono::duration_cast<chrono::milliseconds>(clockEnd - clockStart).count();

    // time the method's execution when caller function casts once and passes the cast object
    AllSpikingNeurons* derived = dynamic_cast<AllSpikingNeurons*>(base);
    clockStart = Clock::now();
    for (int i = 0; i < totalEdgeCount_; i++) {
        advanceEdge(derived);
    }
    clockEnd = Clock::now();
    callerTime = chrono::duration_cast<chrono::milliseconds>(clockEnd - clockStart).count();
}

void outputResults(string levels, int calleeTime, int callerTime) {
    cout << "Casting " << levels << " levels from base class:" << endl;

    cout << "  Callee-casting time: " << setw(5) << calleeTime << " ms" << endl;
    cout << "  Caller-casting time: " << setw(5) << callerTime << " ms" << endl;
    cout << "  Difference in time:  " << setw(5) << calleeTime - callerTime << " ms" << endl;
    cout << "  Casting in the caller method is " << (float)calleeTime / callerTime
        << " times faster than casting in the callee method" << endl << endl;
}

int main() {
    // populate random array to be used by test methods
    r_array = makeRandomArray();

    // instantiate lower level class assigned to base class pointer
    IAllVertices* base4 = new AllLIFNeurons();   // 4 levels from base class
    IAllVertices* base2 = new AllSpikingNeurons();  // 2 levels from base class

    // declare variables that will hold execution times 
    int calleeTime;
    int callerTime;

    // run tests and output results
    runTests(base4, calleeTime, callerTime);
    outputResults("4", calleeTime, callerTime);

    runTests(base2, calleeTime, callerTime);
    outputResults("2", calleeTime, callerTime);

    return 0;
}
