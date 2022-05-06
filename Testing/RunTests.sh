#! /bin/bash
############################################################################################
# Scrip for running Graphitti unit tests and regression tests.
#
# It contains the same tests as the tests.yml workflow that is executed by a
# GitHub action on Pull Requests:
#
#   1. Build Graphitti
#   2. Runs our unit tests
#   3. Runs simulations, in parallel, for the test config files defined in the
#      TEST_FILES array.
#   4. Verifies that the simulation output files match the known-good output files
#
# If you run this script with the -g flag it will attempt to build the GPU
# implementation of Graphitti. This will only succeed if Nvidia CUDA is installed.
# e.g.
#
#   RunTests.sh -g
#
############################################################################################

# Bash scrip usage message
usage="$(basename "$0") [-g] -- Script to Build and Test Graphitti

    Defaults to the CPU implementation unless the [-g] flag is specified

where:
    -g  Builds and tests the GPU implementation (defaults to CPU)
    -h  Prints this help message"

# Check for passed uptions [-g] or [-h]
BUILD_GPU=false
GRAPHITTI=./cgraphitti
while getopts "gh" option
do
    case "$option" in
        h) echo "$usage"
           exit ;;
        g) BUILD_GPU=true
           GRAPHITTI=./ggraphitti ;;
       \?) printf "illegal option: -%s" "$OPTARG" >&2
           echo "$usage" >&2
           exit 1 ;;
    esac
done

# Color constants
BLUE='\033[4;34m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color

# Files and directory variables
CONFIG_DIR=../Testing/RegressionTesting/configfiles
TEST_OUT_DIR=../Testing/RegressionTesting/TestOutput
GOOD_OUT_DIR=../Testing/RegressionTesting/GoodOutput
declare -a TEST_FILES=("test-tiny"
                       "test-small"
                       "test-small-connected"
                       "test-small-long"
                       "test-small-connected-long"
                       "test-medium"
                       "test-medium-connected"
                       "test-medium-long"
                       "test-medium-connected-long")

# This function starts the simulations in parallel
function run_simulations() {
    echo -e "Run simulations in parallel, using: ${GRAPHITTI}"
    for i in "${TEST_FILES[@]}"
    do
        echo -e "${BLUE}[ RUN TEST ]${NC} ${GRAPHITTI} -c ${CONFIG_DIR}/$i.xml"
        ${GRAPHITTI} -c ${CONFIG_DIR}/$i.xml > /dev/null &
    done
}

# This function verifies the outputs from the simulation runs
function verify_outputs() {
    for i in "${TEST_FILES[@]}"
    do
        echo -e "${BLUE}[--------]${NC}Verifying ${i}.xml simulation output...${NC}"
        if (cmp -s ${TEST_OUT_DIR}/$i-out.xml ${GOOD_OUT_DIR}/$i-out.xml); then
            echo -e "${GREEN}[        ]${NC} Output file: ${TEST_OUT_DIR}/$i-out.xml"
            echo -e "${GREEN}[  AND   ]${NC} Good output: ${GOOD_OUT_DIR}/$i-out.xml"
            echo -e "${GREEN}[ PASSED ]${NC} Are equal"
        else
            echo -e "${RED}[        ]${NC} Output file: ${TEST_OUT_DIR}/$i-out.xml"
            echo -e "${RED}[  AND   ]${NC} Good output: ${GOOD_OUT_DIR}/$i-out.xml"
            echo -e "${RED}[ FAILED ]${NC} Are NOT equal"
        fi
    done
}

############################################################################################
#                                    SCRIPT STARTS HERE                                    #
############################################################################################

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}|                            GRAPHITTI BUILD                               |${NC}"
echo -e "${BLUE}============================================================================${NC}"

# Prepare either the CPU or GPU version for the build
cd ../build
if [ $BUILD_GPU == true ]; then
    # Abort if Nvidia CUDA is not installed in the system
    command -v nvcc >/dev/null 2>&1 || {
        echo >&2 "CUDA is required to build the GPU implementation but it's not installed."; 
        echo >&2 "Aborting ...";
        exit 1;
    }
    cmake .. -D ENABLE_CUDA=YES
else
    cmake ..
fi
# Build Graphitti
make

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}|                                UNIT TESTS                                |${NC}"
echo -e "${BLUE}============================================================================${NC}"
./tests

echo
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}|                             REGRESSION TESTS                             |${NC}"
echo -e "${BLUE}============================================================================${NC}"
# Multiple simulations are started in parallel
run_simulations

# Wait until simulations finish execution and verify outputs
echo -e "${BLUE}[ RUN TEST ]${NC} Waiting for simulations to finish..."
wait

echo
echo -e "${BLUE}[========]${NC} Start verification"
verify_outputs
