//
//  tensorflowConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TENSORFLOWCONVERTER_HPP
#define TENSORFLOWCONVERTER_HPP

#include <string>

#include "MNN_generated.h"

#include <expected.hpp>
/**
 * @brief convert tensorflow model to MNN model
 * @param inputModel tensorflow model name(xx.pb)
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
tl::expected<std::string, std::string>
tensorflow2MNNNet(const std::string modelStr, const std::string bizCode);

#endif // TENSORFLOWCONVERTER_HPP
