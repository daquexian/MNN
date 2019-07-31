//
//  tensorflowConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>

#include "MNN_generated.h"
#include "TfUtils.hpp"
#include "TmpGraph.hpp"
#include "logkit.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "tfOpConverter.hpp"
#include "optimizer.hpp"
#include "writeFb.hpp"

#include "tensorflowConverter.hpp"

tl::expected<std::string, std::string>
tensorflow2MNNNet(const std::string modelStr, const std::string bizCode) {
    try {
        std::unique_ptr<MNN::NetT> netT =
            std::unique_ptr<MNN::NetT>(new MNN::NetT());
    tensorflow::GraphDef tfGraph;

    // load
    bool success = tfGraph.ParseFromString(modelStr);
    DCHECK(success) << "read_proto_from_binary failed";

    // build the temp graph
    auto tempGraph = new TmpGraph(tfGraph);
    tempGraph->buildGraph();

    const int nodeCount = tempGraph->opsInOrder.size();
    // Node ==> FlatBuffer
    for (int i = 0; i < nodeCount; i++) {
        TmpNode *tempNode = tempGraph->_getTmpNode(tempGraph->opsInOrder[i]);

        if (!tempNode->isCovered && !tempNode->isDelete) {
            MNN::OpT *op = new MNN::OpT;

            auto creator = tfOpConverterSuit::get()->search(tempNode->opType);
            DCHECK(creator) << "MNN Converter NOT_SUPPORTED_OP: [ " << tempNode->opType << " ]";
            op->name      = tempNode->opName;
            op->type      = creator->opType();
            op->main.type = creator->type();

            creator->run(op, tempNode, tempGraph);
            netT->oplists.emplace_back(op);
        }
    }

    std::map<std::string, int> tensorName;
    // Compute TensorCount
    for (int i = 0; i < nodeCount; i++) {
        TmpNode *node = tempGraph->_getTmpNode(tempGraph->opsInOrder[i]);
        if (!node->isCovered && !node->isDelete) {
            if (node->outTensors.size() == 0) {
                // this node is output node(or last node)
                tensorName.insert(std::make_pair(node->opName, tensorName.size()));
                netT->tensorName.push_back(node->opName);
            } else {
                for (int i = 0; i < node->outTensors.size(); i++) {
                    // tensor name may be NULL, because this tensor is not used by any
                    // other nodes
                    if ("" == node->outTensors[i]) {
                        node->outTensors[i] = node->opName;
                    }
                    if (tensorName.find(node->outTensors[i]) == tensorName.end()) {
                        tensorName.insert(std::make_pair(node->outTensors[i], tensorName.size()));
                        netT->tensorName.push_back(node->outTensors[i]);
                    }
                }
            }
        }
    }

    // Set Input Ouput Tensor Index
    for (auto &op : netT->oplists) {
        TmpNode *curNode = tempGraph->_getTmpNode(op->name);
        if (!curNode->isCovered && !curNode->isDelete) {
            // output index

            if (curNode->outTensors.size() == 0) {
                // the last node output index
                const std::map<std::string, int>::iterator outputTensorNameIndex = tensorName.find(curNode->opName);
                if (outputTensorNameIndex == tensorName.end()) {
                    DLOG(ERROR) << "Output Tensor Not Found!!! ===> " << curNode->opName;
                }

                else {
                    op->outputIndexes.emplace_back(tensorName.find(curNode->opName)->second);
                }
            } else {
                for (int i = 0; i < curNode->outTensors.size(); i++) {
                    const std::map<std::string, int>::iterator outputTensorNameIndex =
                        tensorName.find(curNode->outTensors[i]);
                    if (outputTensorNameIndex == tensorName.end()) {
                        DLOG(ERROR) << "Output Tensor Not Found!!! ===> " << curNode->opName;
                    }

                    else {
                        op->outputIndexes.emplace_back(tensorName.find(curNode->outTensors[i])->second);
                    }
                }
            }

            // input index
            for (int i = 0; i < curNode->inTensors.size(); i++) {
                const std::map<std::string, int>::iterator inputNameIndex = tensorName.find(curNode->inTensors[i]);
                if (inputNameIndex == tensorName.end()) {
                    DLOG(ERROR) << "Input Tensor Not Found!!! ===> " << curNode->opName
                                << "-->Input name: " << curNode->inTensors[i];
                } else {
                    op->inputIndexes.emplace_back(inputNameIndex->second);
                }
            }
        }
    }
    netT->sourceType = MNN::NetSource_TENSORFLOW;
    netT->bizCode    = bizCode;

    delete tempGraph;

        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT);
        return writeFb(newNet, false);

    } catch (const std::exception &e) {
        return tl::make_unexpected(e.what());
    }
}
