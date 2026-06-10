# Graphitti Top-level Sequence Diagram

The following is a Diagram of the top-level simulator execution sequence. The object creation and simulation execution sequences are presented in separate diagrams.

```mermaid
sequenceDiagram
    participant Core
    participant ParamContainer
    participant Simulator
    participant ParameterManager
    participant OperationManager
    participant GraphManager
    participant Serializer
    participant Model
    participant Recorder

    Core-->>ParamContainer: Parse Command Line
    activate ParamContainer
    Note right of ParamContainer: Config File<br>Deserialize File<br>Serialize File<br>Stimulus File

    Core->>Simulator: Set File Names
    Simulator->>ParamContainer: Get File Names
    ParamContainer->>Simulator: File Names
    deactivate ParamContainer

    Core->>ParameterManager: Load Parameter File
    ParameterManager->>Simulator: Get Config File Name
    Simulator->>ParameterManager: Config File Name
    Core->>Simulator: Load Parameters
    Simulator->>ParameterManager: Get parameters from XML
    ParameterManager->>Simulator: Parameters
    Note right of Simulator: width_, height_, epochDuration_,<br>numEpochs_, maxFiringRate_,<br>maxEdgesPerVertex_, RNG, etc.

    Core->>Simulator: Instantiate Simulator Objects
    Note right of Simulator: Details are in a separate<br>sequence diagram.

    Core->>OperationManager: Register Graph Properties
    Note right of OperationManager: Ask all objects to register their Graph properties

    Core->>GraphManager: Read Graph

    Core->>OperationManager: Load Parameters
    Note right of OperationManager: Runs loadParameters method for<br>each instantiated object.<br>Methods are registered to<br>OperationManager on instantiation.

    Core->>Simulator: Setup
    Note right of Simulator: Details are in a separate<br>sequence diagram.

    opt if serialized file name available
        Core->>Serializer: Deserialize file
    end

    Core->>Simulator: Simulate
    Note right of Simulator: Details are in a separate<br>sequence diagram.

    opt if serialization file available
        Core->>Serializer: Serialize Synapses
    end

    Core->>Simulator: Finish
    Simulator->>Model: Clean up model resources
    Core->>Recorder: Terminate Recording
    Core->>Core: Exit program
```

# Simulator Objects Creating Sequence Diagram

Graphitti uses the Factory Method and Singleton design patterns for instantiating the object types defined in the configuration file.

```mermaid
sequenceDiagram
    participant Simulator
    participant Model
    participant Factory
    participant Layout
    participant AllVertices
    participant Connections
    participant AllEdges
    participant Recorder
    
    Simulator->>Model: New CPU/GPU Model
    activate Model
    Model->>Factory: Create Layout
    Factory-->>Layout: Instantiate
    Layout->>Factory: Create Vertices
    Factory-->>AllVertices: Instantiate
    
    Model->>Factory: Create Connections
    Factory-->>Connections: Instantiate
    Connections->>Factory: Create  AllEdges
    Factory-->>AllEdges: Instantiate
    
    Model->>Factory: Create Recorders
    Factory-->>Recorder: Instantiate
```

# Simulator Setup Sequence Diagram

This Diagram represents an overview of the setup sequence.

```mermaid
sequenceDiagram
    participant Core
    participant Simulator
    participant Model
    participant Layout
    participant All Vertices
    participant Connections
    participant All Edges
    participant Recorder

    Core->>Simulator: Setup 
    Simulator->>Model: Model Setup
    Model->>Layout: Get Vertices 
    Layout->>All Vertices: Setup Vertices
    Model->>Connections: Get Edges
    Connections->>All Edges: Setup Edges
    Model->>Layout: Setup Layout
    Layout->>Layout: Initialize Vertices Locations
    Model->>Recorder: Initialize Recorder 
    Model->>Model: Create All Vertices*
    Model->>Layout: Generate Vertex Map 
    Model->>Layout: Initialize Starter Map 
    Model->>Layout: Create All Vertices
```

# Simulation Sequence Diagram

This Diagram represents an overview of the simulation process execution sequence.

```mermaid
sequenceDiagram
    participant Simulator
    participant Model
    participant Layout
    participant All Vertices
    participant Connections
    participant All Edges
    participant Recorder

    loop for i=0 to currentEpoch-1
        Simulator->>Simulator: Advance Epoch
        loop for i=0 to epochDuration-1
            Simulator->>Model: Advance
            Model->>Layout: Get Vertices 
            Layout->>All Vertices: Advance Vertices
            Model->>Connections: Get Edges
            Connections->>All Edges: Advance Edges
        end
        Model->>Connections: Update Connections
        opt if updateConnections returns true
            Model->>Connections: Update Synapses Weights
            Model->>Connections: Create Edge Index Map
        end
        Model->>Recorder: Update (Compile) History
    end

    Simulator->>Model: Save Results
    Model->>Recorder: Save Simulation Data
```

# Recorder Sequence Diagram

This Diagram represents an overview of the recorder sequence.

```mermaid
sequenceDiagram
    participant S as SimulationComponent
    participant R as Recorder
    participant RB as RecordableBase

    S->>R: registerVariable(varName, recordVar, variableType, ......)
    activate R
    RB->>R: getDataType(): string
    Note left of R: add all received variables to the table

    loop Simulation Epoch
        S->>RB: updateVariable()
        activate RB
        loop Variable Table Iteration
            opt if variable is DYNAMIC
                alt XmlRecorder
                    Note right of R: Capture value and Accumulate data
                    RB->>R: getElement(index): variant<uint64_t, BGFLOAT, int, bool>
                    Note left of RB: retrieve primitive data<br>that's encapsulated in a variant
                    R->>R: compileHistories()
                else HDF5Recorder
                    Note right of R: Capture data and Write data to HDF5 file
                    R->>R: compileHistories()
                end
            end
        end
    end
    loop Variable Table Iteration
        opt if variable is CONSTANT
            RB->>R: getElement(index): variant<uint64_t, BGFLOAT, int, bool>
            R->>R: captureData()
        end
        deactivate RB
        R->>R: saveSimData()
        Note right of R: Extracting the value from the variant knowing its type<br>Output data 
    end
```
