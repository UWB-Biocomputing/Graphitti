#! /bin/bash
# 
# Scrip for running Graphitti unit tests and regression tests.
#
# It contains the same tests as the tests.yml workflow that is executed by a
# GitHub action on Pull Requests:
#
#   1. Runs our unit tests
#   2. Runs simulations, in parallel, for the test config files defined in the
#      TEST_FILES array.
#   3. Verifies that the simulation output files match the known-good output files

BLUE='\033[4;34m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color
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

function run_simulations() {
    for i in "${TEST_FILES[@]}"
    do
        echo -e "${BLUE}[ RUN TEST ]${NC} Starting $i simulation in parallel..."
        ./cgraphitti -c ${CONFIG_DIR}/$i.xml > /dev/null &
    done
}

function verify_outputs() {
    for i in "${TEST_FILES[@]}"
    do
        echo -e "${BLUE}[--------]${NC}Verifying $i simulation output...${NC}"
        if (diff ${TEST_OUT_DIR}/$i-out.xml ${GOOD_OUT_DIR}/$i-out.xml); then
            echo -e "${GREEN}[ PASSED ]${NC} Output file ${TEST_OUT_DIR}/$i-out.xml"
        else
            echo -e "${RED}[ FAILED ]${NC} Output file ${TEST_OUT_DIR}/$i-out.xml"
        fi
    done
}

echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE}|                        UNIT TESTS                      |${NC}"
echo -e "${BLUE}==========================================================${NC}"
./tests

echo
echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE}|                    REGRESSION TESTS                    |${NC}"
echo -e "${BLUE}==========================================================${NC}"
# Multiple simulations are started in parallel
run_simulations

# Wait until simulations finish execution and verify outputs
echo -e "${BLUE}[ RUN TEST ]${NC} Waiting for simulations to finish..."
wait
echo
echo -e "${BLUE}[========]${NC} Start verification"
verify_outputs
