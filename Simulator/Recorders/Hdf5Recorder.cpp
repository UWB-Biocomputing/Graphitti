#include "Hdf5Recorder.h"
#include "Simulator.h"
#include "Model.h"
#include "OperationManager.h"
#include "ParameterManager.h"
#include <iostream>
#include <fstream>

Hdf5Recorder::Hdf5Recorder()
{
    ParameterManager::getInstance().getStringByXpath(
        "//RecorderParams/RecorderFiles/resultFileName/text()", resultFileName_);

    function<void()> printParametersFunc = std::bind(&Hdf5Recorder::printParameters, this);
    OperationManager::getInstance().registerOperation(Operations::printParameters,
                                                      printParametersFunc);

    fileLogger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("file"));

    resultOut_ = nullptr;
    init();
}

Recorder* Hdf5Recorder::Create() {
    return new Hdf5Recorder();
}

// Other member functions implementation...
