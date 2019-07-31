//
//  onnxConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ONNXCONVERTER_HPP
#define ONNXCONVERTER_HPP

#include "MNN_generated.h"

#include <expected.hpp>

/**
 * @brief convert ONNX model to MNN model
 * @param inputModel ONNX model name(xxx.onnx)
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
tl::expected<std::string, std::string>
onnx2MNNNet(const std::string modelStr, const std::string bizCode);

#endif // ONNXCONVERTER_HPP
