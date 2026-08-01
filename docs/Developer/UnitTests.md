# Unit Testing

Information on unit tests, test config files for regression testing, and testing that has been done internally regarding potential improvements to Graphitti.

## Unit Tests

We use [Googletest](GoogleTestsTutorial.md) to develop our unit tests.


### Running The Unit And Regression tests
Unit and regression tests are important to minimize the chances that changes to the code base will produce undesired side effects. 

We have a battery of tests that are run by a GitHub action on any `push` or `pull request` against the master branch. These tests are only executed for the CPU implementation of Graphitti.

The same tests executed by the described GitHub action can be run locally with the `RunTests.sh` bash script inside the `Testing` directory. The script can be told to exercise the tests against the GPU implementation by running it with the `-g` flag.

To run the tests against the CPU implementation, inside the `Testing` directory run:

    bash RunTests.sh

To run the tests against the GPU implementation, inside the `Testing` directory run:

    bash RunTests.sh -g

**Note**: Currently, the GPU regresssion tests fail because the random numbers generated are different from the ones
generated during the CPU execution, causing the result files to be different to the CPU known good results.

### Growth-to-STDP Integration Regression Test

Serialization lets the output network of a growth simulation be used as the input for an STDP
simulation. When a growth checkpoint (which stores a `ConnGrowth`/`AllDSSynapses` network) is
deserialized into a configuration whose `ConnectionsParams` is a non-growth class (for example
`ConnStatic` with `AllSTDPSynapses`), the deserializer imports only the grown topology — each
edge's source, destination, weight, and type — into the network described by the current
configuration file. The restored vertices, layout, and global simulation state (RNG, simulation
step) are left as loaded from the checkpoint. This behavior lives in `Serializer::deserialize()`.

A dedicated regression test exercises this end to end:

- `Testing/RegressionTesting/configfiles/test-growth-stdp-source.xml` — a short growth run whose
  start radius is large enough to grow a non-trivial set of edges on the `test-small` grid. It is
  run with `-s` to serialize the grown network to a checkpoint.
- `Testing/RegressionTesting/configfiles/test-growth-stdp.xml` — an STDP run (empty `ConnStatic`,
  `AllSTDPSynapses`, edge-free graph) that is run with `-d` pointing at that checkpoint, so every
  edge in the run originates from the imported grown network. Its output is compared against
  `Testing/RegressionTesting/GoodOutput/Cpu/test-growth-stdp-out.xml`.

Because the STDP run depends on the growth run's checkpoint, this is a sequential pipeline. It runs
after the parallel single-simulation regression tests in both `RunTests.sh` and the
`tests.yml` GitHub action. To run it manually from the `build` directory:

    ./cgraphitti -c ../Testing/RegressionTesting/configfiles/test-growth-stdp-source.xml \
        -s ../Testing/RegressionTesting/TestOutput/test-growth-stdp-checkpoint.xml
    ./cgraphitti -c ../Testing/RegressionTesting/configfiles/test-growth-stdp.xml \
        -d ../Testing/RegressionTesting/TestOutput/test-growth-stdp-checkpoint.xml
    ../Testing/RegressionTesting/compare_matrices \
        ../Testing/RegressionTesting/GoodOutput/Cpu/test-growth-stdp-out.xml \
        ../Testing/RegressionTesting/TestOutput/test-growth-stdp-out.xml

If the growth model, the STDP model, or the topology-import path changes in a way that alters the
grown network or how it is imported, this test's output diverges from the known-good file and the
test fails.

### Growth-to-STDP Unit Tests

The regression test above tells you *that* the simulation output changed; the unit tests tell you
*whether the import mechanism itself* is still correct. They run the same two-stage pipeline, but
both stages serialize their final state and the tests assert on the checkpoints:

- `Testing/UnitTesting/GrowthToStdpSourceTest.cpp` — runs the growth configuration with `-s` and
  checks that the checkpoint holds a `ConnGrowth` network with at least one edge, so that the
  second stage has something to import.
- `Testing/UnitTesting/GrowthToStdpImportTest.cpp` — runs the STDP configuration with `-d` on that
  checkpoint and `-s` on a new one, then checks that the resulting network uses the classes named
  in the STDP configuration file (`ConnStatic` and `AllSTDPSynapses`) rather than the checkpoint's
  own classes, and that it contains exactly as many edges as the growth run produced.

Both files share `Testing/UnitTesting/GrowthToStdpHelper.cpp`, which pulls those fields out of a
Cereal checkpoint.

As with the serialization tests, each stage is a separate executable: a test runs a simulation from
start to finish, and running two simulations against the same singleton instances causes a
segmentation fault. The second stage consumes the first stage's checkpoint, so they must run in
order. From the `build` directory:

    ./run_growth_stdp_test.sh

---------
[<< Go back to the Graphitti home page](../index.md)
