//
//  caffeConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CAFFECONVERTER_HPP
#define CAFFECONVERTER_HPP

#include "MNN_generated.h"

/**
 * @brief convert caffe model(prototxt and caffemodel) to MNN model
 * @param prototxtFile prototxt file name
 * @param modelFile caffemodel file name
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
std::string caffe2MNNNet(const std::string prototxtStr, const std::string modelStr, const std::string bizCode);

#endif // CAFFECONVERTER_HPP
