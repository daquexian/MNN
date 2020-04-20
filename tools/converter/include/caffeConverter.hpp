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
int caffe2MNNNet(void **txt_buf, const size_t txt_buflen, void **model_buf, const size_t model_buflen, const std::string bizCode,
                 std::unique_ptr<MNN::NetT>& netT);

#endif // CAFFECONVERTER_HPP
