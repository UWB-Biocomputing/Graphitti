# Recorders

This note describes the Recorder subsystem as it exists in the current codebase. The code is the source of truth. Older design ideas that no longer match the implementation are summarized near the end so the differences are explicit.

## Overview

Graphitti uses a `Recorder` object to collect selected simulation data and write it to an output file after or during a run.

The current subsystem is built around:

- `Recorder`: abstract interface for recorder implementations
- `XmlRecorder`: text output, stores captured history in memory, writes at the end
- `Hdf5Recorder`: binary output, appends dynamic data to HDF5 datasets during the run
- `Xml911Recorder`: a legacy specialized subclass that still exists in the factory, but its overridden `compileHistories()`, `saveSimData()`, and `printParameters()` are effectively empty today
- `RecordableBase`: abstract interface for anything the recorder can observe
- `RecordableVector<T>`: generic 1-D recordable container
- `EventBuffer<T>`: epoch-aware event history buffer used by the recorder as a per-epoch view

The current recorder design is registration-driven. Recorder implementations do not discover data on their own. Instead, simulation subsystems explicitly register recordable objects and assign each one a name and an update cadence.

## What `Recorder` Currently Does

The abstract `Recorder` interface defines:

- `init()`: prepare the output destination
- `term()`: close the output destination
- `compileHistories()`: capture per-epoch history for registered dynamic variables
- `saveSimData()`: write final output
- `printParameters()`: log recorder parameters
- `registerVariable(name, recordable, updatedType)`: register a single recordable object
- `registerVariable(name, vector<RecordableBase*>, updatedType)`: register multiple recordable objects with generated suffixed names

It also defines two update frequencies:

- `CONSTANT`: captured once at final save time
- `DYNAMIC`: captured once per epoch during simulation

The base class itself does not directly implement recorder logic for neuron spikes, NG911 calls, layouts, or connections. However, the `Recorder` interface also defines the `multipleTypes` type alias and declares `getStarterNeuronMatrix(VectorMatrix &matrix, const vector<bool> &starterMap)`.

## Active Recorder Creation and Use

The active recorder is chosen from configuration, not hardcoded.

### Creation path

The current path is:

1. `Core::runSimulation()` loads the XML parameter file.
2. `Simulator::instantiateSimulatorObjects()` creates a `CPUModel` or `GPUModel`.
3. `Model::Model()` reads `//RecorderParams/@class` from the parameter file.
4. `Factory<Recorder>::getInstance().createType(type)` instantiates the requested recorder.

The recorder types currently registered with the factory are:

- `XmlRecorder`
- `Xml911Recorder`
- `Hdf5Recorder` only when the build is compiled with `HDF5`

### How the recorder is used at runtime

After creation, the recorder is owned by `Model` and accessed through `Model::getRecorder()`.

The current lifecycle is:

1. `Model::setupSim()` calls `recorder_->init()`.
2. `Core::runSimulation()` later triggers `registerHistoryVariables` through `OperationManager`.
3. `Simulator::simulate()` runs epochs.
4. At the end of each epoch, `Simulator::simulate()` calls:
   - `model_->updateConnections()`
   - `model_->updateHistory()`
5. `Model::updateHistory()` calls `recorder_->compileHistories()`.
6. After simulation, `Core::runSimulation()` calls `simulator.saveResults()`.
7. `Simulator::saveResults()` calls `model_->saveResults()`.
8. `Model::saveResults()` calls `recorder_->saveSimData()`.
9. After `finish()`, `Core::runSimulation()` calls `simulator.getModel().getRecorder().term()`.

This means the recorder is initialized after vertex setup, edge setup, and layout setup, but before concrete vertex creation and connection setup. Dynamic data are compiled once per epoch, final output is produced after the run, and the recorder is terminated explicitly at shutdown.

## Recorder Lifecycle and Call Flow

### Setup phase

During setup:

- the recorder is instantiated from config
- the recorder registers its own `printParameters()` callback with `OperationManager` in its constructor
- `Model::setupSim()` calls `recorder->init()`

At this point, the recorder has an output target but usually has no registered variables yet.

### Registration phase

After setup, `Core::runSimulation()` executes the `registerHistoryVariables` operation through `OperationManager`. Any subsystem that registered a callback for that operation can then register variables with the recorder.

This registration step happens:

- after model setup
- after parameter loading
- after optional deserialization
- before simulation starts

### Per-epoch phase

At the end of every epoch:

- the model may update connections
- `Model::updateHistory()` calls `Recorder::compileHistories()`

Only `DYNAMIC` variables are handled during this phase.

### Final save phase

After all epochs are complete:

- `Model::saveResults()` calls `Recorder::saveSimData()`

Only then are `CONSTANT` variables captured and written.

### Shutdown phase

Finally:

- `Recorder::term()` closes the file or HDF5 object

## How Variables Are Registered

Registration is explicit and owner-driven.

Each subsystem that owns recordable state typically implements `registerHistoryVariables()` and calls:

```cpp
recorder.registerVariable("name", someRecordableObject, Recorder::UpdatedType::DYNAMIC);
```

or:

```cpp
recorder.registerVariable("baseName", vectorOfRecordablePointers, Recorder::UpdatedType::DYNAMIC);
```

### What registration stores

For each registered variable, the recorder stores:

- the variable name
- the variable update frequency
- a reference to the original `RecordableBase` object
- the runtime basic type string returned by `RecordableBase::getDataType()`

`XmlRecorder` stores this in `singleVariableInfo`.

`Hdf5Recorder` stores this in `hdf5VariableInfo`, which also converts the runtime type to an HDF5 datatype during registration.

### Important implications

- Registration stores references, not copies.
- The recordable object must remain alive for the duration of recording.
- The recorder depends on the recordable object to expose a flat 1-D view through `getNumElements()` and `getElement(index)`.
- The vector overload generates names by appending an integer index such as `neuron_0`, `neuron_1`, and so on.
- The vector overload exists and is unit-tested, but the live simulation code currently registers variables one-by-one rather than using it.

## What Subsystems Currently Register Data

The current live `registerHistoryVariables()` call sites are:

### Layout

`Layout::registerHistoryVariables()` registers:

- `vertexTypeMap` as `CONSTANT`

This is a `RecordableVector<vertexType>`.

### Neuro layout

`LayoutNeuro::registerHistoryVariables()` calls the base layout registration and also registers:

- `x_Location` as `CONSTANT`
- `y_Location` as `CONSTANT`

These are `VectorMatrix` objects containing neuron coordinates.

### NG911 layout

`Layout911::registerHistoryVariables()` calls the base layout registration and also registers:

- `x_Location` as `CONSTANT`
- `y_Location` as `CONSTANT`

These NG911 coordinates are recorded through `RecordableVector<BGFLOAT>` mirrors:

- `xloc_` and `yloc_` remain the simulation/layout coordinate storage.
- `xlocRecorder_` and `ylocRecorder_` are recorder-compatible mirrors.
- The mirrors are populated from GraphML coordinates during layout setup.
- They are registered as `CONSTANT`, so each coordinate is written once at final save time.
- They do not replace `xloc_` or `yloc_`; they only expose the same coordinates to the recorder.

### Neuro vertices

`AllSpikingNeurons::registerHistoryVariables()` registers:

- `Neuron_0`, `Neuron_1`, ... as `DYNAMIC`

Each variable is an `EventBuffer<uint64_t>` that records spike times for one neuron during an epoch.

### Static neuro connections

`ConnStatic::registerHistoryVariables()` registers:

- `weight` as `DYNAMIC`
- `sourceVertex` as `DYNAMIC`
- `destinationVertex` as `DYNAMIC`

These three `RecordableVector` objects form an active edge list for each epoch:

- one value is recorded per active edge for `weight`
- one value is recorded per active edge for `sourceVertex`
- one value is recorded per active edge for `destinationVertex`
- entries at the same index describe the same active edge
- the vectors are populated immediately before dynamic histories are compiled
- the output is the current active edge list per epoch, not only changed edges and not only the initial edge list

### Growth connections

`ConnGrowth::registerHistoryVariables()` registers:

- `radii` as `DYNAMIC`
- `rates` as `DYNAMIC`
- `outgrowth` as `DYNAMIC`
- `deltaR` as `DYNAMIC`

These are 1-D `VectorMatrix` values recorded once per epoch.

Growth `CompleteMatrix` fields are intentionally not recorded:

- examples include weights, deltas, and areas
- `CompleteMatrix` is not a recorder-compatible flat value source
- issue #722 did not add recorder output for those fields

### NG911 connections

`Connections911::registerHistoryVariables()` registers:

- `verticesDeleted` as `DYNAMIC`

This is a `RecordableVector<int>`.

### NG911 vertices

`All911Vertices::registerHistoryVariables()` registers:

Constants:

- `numTrunks`
- `numServers`
- `droppedCalls`
- `receivedCalls`

Dynamic per-vertex histories:

- `BeginTimeHistory_<i>`
- `AnswerTimeHistory_<i>`
- `EndTimeHistory_<i>`
- `WasAbandonedHistory_<i>`
- `QueueLengthHistory_<i>`
- `UtilizationHistory_<i>`

The corresponding current container types are:

- `beginTimeHistory_`: `vector<EventBuffer<uint64_t>>`
- `answerTimeHistory_`: `vector<EventBuffer<uint64_t>>`
- `endTimeHistory_`: `vector<EventBuffer<uint64_t>>`
- `wasAbandonedHistory_`: currently declared as `vector<EventBuffer<uint64_t>>`
- `queueLengthHistory_`: `vector<EventBuffer<uint64_t>>`
- `utilizationHistory_`: `vector<EventBuffer<float>>`
- `droppedCalls_`: `RecordableVector<int>`
- `receivedCalls_`: `RecordableVector<int>`
- `numServers_`: `RecordableVector<int>`
- `numTrunks_`: `RecordableVector<int>`

Other NG911 setup values remain configuration and log values for now:

- `redialP`
- `avgDrivingSpeed`
- `psapsToErase`
- `respsToErase`

They are not registered as recorder constants unless a future scalar-compatible recordable type is added.

## `CONSTANT` vs `DYNAMIC`

The distinction is implementation-significant.

### `DYNAMIC`

`DYNAMIC` variables are processed in `compileHistories()` once per epoch.

For both XML and HDF5 recorders:

- if `getNumElements() > 0`, the recorder captures the current epoch's values
- after capture, the recorder calls `startNewEpoch()` on the recordable object

That `startNewEpoch()` call is the main mechanism that tells the recordable object to expose a fresh epoch window next time.

### `CONSTANT`

`CONSTANT` variables are not captured during the epoch loop.

They are captured only during `saveSimData()`, once, at the end of the simulation.

### Practical meaning

- `CONSTANT` is for values that should be written as one final snapshot
- `DYNAMIC` is for values whose epoch-by-epoch history matters

This is the current behavior regardless of whether the value itself was technically mutable during the run. What matters is how the owner registers it.

## `XmlRecorder` Behavior

`XmlRecorder` is the simpler implementation.

### Initialization

- reads `//RecorderParams/RecorderFiles/resultFileName/text()`
- requires the file name to end in `.xml`
- opens an `ofstream`

### Registration

Each registered variable becomes a `singleVariableInfo` containing:

- `variableName_`
- `dataType_`
- `variableType_`
- `variableLocation_`
- `variableHistory_`

### Dynamic capture

On each `compileHistories()` call:

- iterate over all registered variables
- for each `DYNAMIC` variable:
  - append all current elements from the recordable object into `variableHistory_`
  - call `startNewEpoch()` on the source object

### Final save

On `saveSimData()`:

- write the XML declaration
- for each registered variable:
  - if it is `CONSTANT`, capture its values once at this moment
  - if its accumulated history is non-empty, write one `<Matrix>` element

### Output format

Each variable is written as a flat XML matrix:

- one matrix per registered variable
- `rows="1"`
- `columns` equals the total number of captured values
- values are space-separated in a single flat sequence

The XML writer does not preserve explicit epoch boundaries. If a dynamic variable is captured over multiple epochs, the output is just the concatenated values in time order.

## `Hdf5Recorder` Behavior

`Hdf5Recorder` is only compiled when `HDF5` is enabled.

### Initialization

- reads `//RecorderParams/RecorderFiles/resultFileName/text()`
- requires the file name to end in `.h5`
- verifies that the file is writable
- creates an `H5File` with `H5F_ACC_TRUNC`

### Registration

Each registered variable becomes an `hdf5VariableInfo` containing:

- `variableName_`
- `dataType_`
- `hdf5Datatype_`
- `hdf5DataSet_`
- `variableType_`
- `variableLocation_`

During registration, `convertType()` maps the runtime type string to an HDF5 datatype.

### Dynamic capture

On each `compileHistories()` call:

- iterate over all registered variables
- for each `DYNAMIC` variable:
  - if the dataset does not exist yet:
    - create a 1-D dataset with unlimited max size
    - enable chunking
    - write the first epoch's values
  - otherwise:
    - extend the dataset by the number of current elements
    - append the new values into the extended region
  - call `startNewEpoch()` on the source object

### Constant capture

On `saveSimData()`:

- iterate over all registered variables
- for each `CONSTANT` variable with at least one element:
  - create a fixed-size 1-D dataset
  - write the current values once

### Output semantics

Like `XmlRecorder`, the HDF5 implementation currently records flat 1-D data per variable. It does not attach explicit epoch counts, per-epoch offsets, or a higher-dimensional epoch structure. Dynamic data are appended in order.

### Current caveat

The current HDF5 implementation is clearly intended to support `bool`, `int`, and `vertexType`, but its `NATIVE_INT` write path does not cleanly distinguish those cases. The unit tests cover `uint64_t` and `vertexType` scenarios, and the code path should be treated as "current implementation behavior" rather than as a polished, fully generalized type layer.

## `Xml911Recorder`

`Xml911Recorder` still exists and is still registered in the factory, so it can be selected by configuration.

However, in the current code:

- `compileHistories()` is empty
- `saveSimData()` contains only commented-out legacy code
- `printParameters()` is empty

As a result:

- `Xml911Recorder` is legacy/placeholder code.
- It is not the current intended NG911 recorder path.
- NG911 simulations should use the generic recorder registration system through `XmlRecorder` or, when enabled and appropriate, `Hdf5Recorder`.

## `RecordableBase`, `RecordableVector`, and `EventBuffer`

### `RecordableBase`

`RecordableBase` is the recorder-facing interface. Every recordable type must provide:

- `getNumElements()`
- `getElement(index)`
- `startNewEpoch()`
- `setDataType()`
- `getDataType()`

This is what allows the recorder to treat many different containers in a uniform way.

### `RecordableVector<T>`

`RecordableVector<T>` is the general-purpose 1-D recordable container.

It:

- stores data in `vector<T> dataSeries_`
- exposes the whole vector through `getNumElements()` and `getElement(index)`
- clears the vector in `startNewEpoch()`
- stores the runtime type string using `typeid(T).name()`

It is used for:

- vertex type maps
- deleted vertex lists
- per-run counters stored as vectors
- connection source, destination, and weight histories

### `EventBuffer<T>`

`EventBuffer<T>` is a specialized circular buffer that also acts like a recordable vector from the recorder's point of view.

Its key recorder-facing behavior is different from `RecordableVector<T>`:

- `getNumElements()` returns only the number of elements in the current or just-finished epoch
- `getElement(index)` returns elements relative to the epoch start
- `startNewEpoch()` advances the epoch window without clearing the underlying buffer contents

This is what makes it suitable for spike times and other per-epoch event histories where events may need queue semantics internally, but the recorder should only see the epoch-local slice.

## Supported Data Types

### Recorder variant types

The current `RecordableBase` and `Recorder` variant types include:

- `uint64_t`
- `bool`
- `int`
- `BGFLOAT`
- `vertexType`
- `double`
- `unsigned char`

In the current build configuration, `BGFLOAT` is defined as `float`.

### XML support

`XmlRecorder::toXML()` explicitly handles:

- `uint64_t`
- `bool`
- `int`
- `BGFLOAT`
- `vertexType` written as `int`
- `double`
- `unsigned char`

### HDF5 support

`Hdf5Recorder::convertType()` explicitly maps:

- `uint64_t`
- `bool`
- `int`
- `float`
- `double`
- `vertexType`
- `unsigned char`

Because `BGFLOAT` is currently `float`, current `RecordableVector<BGFLOAT>` values map into the HDF5 float path in this build.

## Current Limitations and Assumptions

The current implementation makes several assumptions that are worth documenting explicitly.

### Data shape assumptions

- Recorded data are effectively 1-D.
- Dynamic histories are flattened across epochs.
- There is no explicit epoch boundary metadata in XML or HDF5 output.

### Lifetime assumptions

- The recorder stores references to registered objects.
- Registered `RecordableBase` instances must outlive recorder use.

### Registration assumptions

- Nothing is recorded unless some subsystem explicitly registers it.
- The recorder does not introspect the model to find data automatically.

### Build assumptions

- `Hdf5Recorder` only exists when Graphitti is compiled with HDF5 support.
- Otherwise, only XML-based recorder types are available through the recorder factory.

### Implementation limitations

- `Xml911Recorder` is not an actively functional specialized recorder at present.
- NG911 scalar setup values remain configuration and log values until there is a scalar-compatible recordable type.
- Some comments in headers still refer to older planned behavior or planned cleanup.
- Several recorder diagrams and older notes in the docs describe designs that do not exactly match the current code.

## Unit Tests That Reflect Current Expectations

The current recorder unit tests cover:

- recorder factory creation
- file initialization and termination
- single-variable registration
- vector-of-pointers registration
- XML output generation
- HDF5 dataset writing and dataset extension

These tests are useful as a secondary source for intended current behavior, especially for:

- generated names such as `neuron_0`
- XML `<Matrix>` formatting
- HDF5 append behavior for dynamic histories

They do not fully prove that every live simulation registration path is correct, but they do show what the recorder implementations are expected to do in isolation.

## What Was Removed or Superseded from Older Notes

Older recorder notes mixed current behavior with historical design discussion. The following older details have been superseded by the current implementation and are intentionally not treated as current behavior:

- old HDF5 dataset names such as `burstinessHist_`, `spikesHistory_`, `Tsim`, and similar legacy examples
- removed methods such as `init(const string&)`, `initDataSet()`, and `initDefaultValues()`
- growth-recorder-specific discussions that assume dedicated HDF5 growth recorder subclasses or built-in growth datasets
- older notes about recorder-specific radius or rate histories as active recorder behavior
- assumptions that HDF5 output has a richer built-in schema than the current flat dataset-per-variable implementation

The current implementation is simpler:

- one active recorder object selected from config
- owner-driven variable registration
- per-variable flat histories
- XML accumulates in memory and writes at the end
- HDF5 appends dynamic data during simulation and writes constants at the end

## Summary

The current recorder subsystem is a generic, registration-based data capture layer. It does not encode model semantics itself. Instead, layouts, vertices, and connections decide what to expose through `RecordableBase` objects.

The practical current model is:

- pick a recorder class from config
- initialize it during model setup
- register variables through subsystem `registerHistoryVariables()` callbacks
- capture dynamic histories once per epoch
- capture constant values once at final save time
- write flat per-variable XML or HDF5 output

That is the behavior the rest of this repository currently implements.
