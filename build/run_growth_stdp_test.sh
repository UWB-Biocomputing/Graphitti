#! /bin/bash
############################################################################################
# Script for running the Graphitti CPU growth-to-STDP integration tests.
#
# This script runs two tests, in order:
#   1. A growth simulation that serializes the network it grows.
#   2. An STDP simulation that deserializes that network and imports its topology.
#
# The second test consumes the checkpoint written by the first, so they cannot be reordered
# or run concurrently. They are separate executables because each one runs a simulation from
# start to finish, and running two simulations against the same singleton instances results
# in a segmentation fault.
#
# If either of the tests fail, the script will exit with an error message.
#
############################################################################################

# Run the growth simulation that produces the input network
echo "Running growth simulation test..."
./growthStdpSourceTest
if [ $? -ne 0 ]; then
    echo "Error: Growth simulation test failed."
    exit 1
fi

# Run the STDP simulation that starts from the grown network
echo "Running STDP-from-growth import test..."
./growthStdpImportTest
if [ $? -ne 0 ]; then
    echo "Error: STDP-from-growth import test failed."
    exit 1
fi

# If all tests pass
echo "All tests completed successfully."
echo "We grew a network, serialized it, and started an STDP simulation from it, and verified that the STDP simulation used the classes from its own configuration file with every edge from the grown network."
