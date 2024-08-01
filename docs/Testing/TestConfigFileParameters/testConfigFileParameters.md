# Test Config File Parameters

The standard suite of testing config files for the Graphitti simulator consists  
of the 12 listed in Table 1. 

## Changing Parameters

Parameters that vary between the test config files are limited to the pool size,  
starting radius, and number of epochs. These values vary according to Table 1.

### Table 1

|               Config File Name | Pool Size | startRadius | numEpochs |
|-------------------------------:|----------:|------------:|----------:|
|                 test-small.xml |     10x10 |         0.4 |         2 |
|       test-small-connected.xml |     10x10 |        0.49 |         2 |
|            test-small-long.xml |     10x10 |         0.4 |        10 |
|  test-small-connected-long.xml |     10x10 |        0.49 |        10 |
|                test-medium.xml |     30x30 |         0.4 |         2 |
|      test-medium-connected.xml |     30x30 |        0.49 |         2 |
|           test-medium-long.xml |     30x30 |         0.4 |        10 |
| test-medium-connected-long.xml |     30x30 |        0.49 |        10 |
|                 test-large.xml |   100x100 |         0.4 |         2 |
|       test-large-connected.xml |   100x100 |        0.49 |         2 |
|            test-large-long.xml |   100x100 |         0.4 |        10 |
|  test-large-connected-long.xml |   100x100 |        0.49 |        10 |


## Constant Parameters

Values that are constant, remaining the same between the config files, are  
listed in Table 2 below. All other parameters not listed also remain constant  
between these test config files.

### Table 2

|              Parameter |             Value |
|-----------------------:|------------------:|
|          epochDuration |               100 |
|          maxFiringRate |               200 |
|            InitRNGSeed |                 1 |
|           NoiseRNGSeed |                 1 |
|     Active NList Ratio |               0.1 |
| Inhibitory NList Ratio |               0.1 |
|         Vertices Class |     AllLIFNeurons |
|            Edges Class |     AllDSSynapses |
|      Connections Class |        ConnGrowth |
|           Layout Class |       FixedLayout |
|         Recorder Class | XmlGrowthRecorder |