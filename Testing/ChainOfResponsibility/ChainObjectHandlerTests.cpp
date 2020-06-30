//
// Created by Chris O'Keefe on 6/26/2020.
//

#include "ChainOperationHandler.h"
#include "Operations.h"
#include "gtest/gtest.h"

TEST(ChainObjectHandlerTest, ChainObjectInstanceIsNotNull) {
    ASSERT_TRUE(ChainOperationHandler::getInstance() != nullptr);
}
