#! /bin/bash
############################################################################################
# Script for running Graphitti CPU Serialization tests.
#
# This script runs three serialization tests:
#   1. A full simulation test.
#   2. A first half simulation test.
#   3. A second half simulation test.
#
# If any of the tests fail, the script will exit with an error message.
#
############################################################################################

# Run the full simulation test
echo "Running full simulation test..."
./serialFullTest
if [ $? -ne 0 ]; then
    echo "Error: Full simulation test failed."
    exit 1
fi

# Run the first half simulation test
echo "Running first half simulation test..."
./serialFirstHalfTest
if [ $? -ne 0 ]; then
    echo "Error: First half simulation test failed."
    exit 1
fi

# Run the second half simulation test
echo "Running second half simulation test..."
./serialSecondHalfTest
if [ $? -ne 0 ]; then
    echo "Error: Second half simulation test failed."
    exit 1
fi

# If all tests pass
echo "All tests completed successfully."
echo "We ran one full simulation and two half simulations, and verified that the resulting serialized files were identical."
echo "This confirms that the serialization process is working correctly."
