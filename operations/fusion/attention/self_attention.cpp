/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "operations/fusion/attention/self_attention.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/ops/attn_operation.h"
#include "operations/aclnn/ops/dequant_rope_quant_kvcache_operation.h"
#include "operations/aclnn/ops/flash_attention_score_operation.h"
#include "operations/aclnn/ops/reshape_decode_kv_cache_operation.h"
#include "operations/aclnn/ops/x_attention_operation.h"
#include "operations/fusion/infer_shape_functions.h"
#include "operations/fusion/utils.h"
#include <customize/customize_op_params.h>
#include <securec.h>

namespace atb_speed {
namespace common {

template <typename NormParamType>
int64_t ConstructOneRecCrossFusedInferAttentionNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap);

template <typename NormParamType>
int64_t AddSelfAttention(atb::GraphParam &opGraph,
                         const FusionAttentionParam<NormParamType> &param,
                         std::map<std::string, uint32_t> &tensorMap) {
  std::stringstream ss;
  for (auto tensor = tensorMap.cbegin(); tensor != tensorMap.cend(); ++tensor) {
    ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second
       << std::endl;
  }
  ATB_SPEED_LOG_DEBUG("layer map tensor:\n" << ss.str());
  const bool shouldSkipOneRecCrossPrefillCacheUpdate =
      param.isOneRecCrossAttention && param.isPrefill;
  if (!param.enableRopeQuantKvcache && param.needUpdateKVCache &&
      !shouldSkipOneRecCrossPrefillCacheUpdate) {
    if (!param.isFA) { // Paged Attention path
      if (param.enableXattention && !param.isPrefill) {
        // Decode phase for unified decode-kv-cache pipeline.
        CHECK_OPERATION_STATUS_RETURN(
            AddDecodeKVCacheOperation(opGraph, param, tensorMap));
      } else if (!param.enableXattention) {
        // Fallback: standard reshape-and-cache pipeline.
        CHECK_OPERATION_STATUS_RETURN(
            AddPaKVCacheOperation(opGraph, param, tensorMap));
      }
    } else if (param.isFA &&
               param.attnBackend ==
                   atb_speed::common::OpBackend::ACLNN) { // ACLNN FA
      CHECK_OPERATION_STATUS_RETURN(
          AddFaKVCacheOperation(opGraph, param, tensorMap));
    }
  }
  // SelfAttentionNode
  atb::Node selfAttentionNode;
  // Flash-Attention path does not use the decode-kv-cache prefill pipeline.
  if (param.isFA && !param.enableXattention) { // FA
    if (param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
        param.isPrefill) { // ACLNN FA Encode
      CHECK_OPERATION_STATUS_RETURN(
          ConstructPaEncoderNode(selfAttentionNode, param, tensorMap));
    } else if (param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
               !param.isPrefill) { // ACLNN FA Decode
      CHECK_OPERATION_STATUS_RETURN(
          ConstructAclNNDecoderNode(selfAttentionNode, param, tensorMap));
    } else { // ATB FA
      CHECK_OPERATION_STATUS_RETURN(
          ConstructFaNode(selfAttentionNode, param, tensorMap));
    }
  } else {
    if (param.isPrefill && !param.enableSplitFuse &&
        !param.isOneRecCrossAttention) { // PA Prefill
      if (param.isPrefixCacheWithoutChunk) {
        CHECK_OPERATION_STATUS_RETURN(
            ConstructPrefixEncoderNode(selfAttentionNode, param, tensorMap));
      } else {
        CHECK_OPERATION_STATUS_RETURN(
            ConstructPaEncoderNode(selfAttentionNode, param, tensorMap));
      }
    } else if (param.attnBackend ==
               atb_speed::common::OpBackend::ATB) { // ATB PA Decode
      if (param.enableXattention) {
        CHECK_OPERATION_STATUS_RETURN(
            ConstructDecodeKVCacheNode(opGraph, param, tensorMap));
        return atb::NO_ERROR;
      } else {
        CHECK_OPERATION_STATUS_RETURN(
            ConstructPaDecoderNode(selfAttentionNode, param, tensorMap));
      }
    } else if (param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
               param.isOneRecCrossAttention) {
      CHECK_OPERATION_STATUS_RETURN(ConstructOneRecCrossFusedInferAttentionNode(
          selfAttentionNode, param, tensorMap));
    } else if (param.attnBackend ==
               atb_speed::common::OpBackend::ACLNN) { // ACLNN PA Decode
      CHECK_OPERATION_STATUS_RETURN(
          ConstructAclNNDecoderNode(selfAttentionNode, param, tensorMap));
    }
  }
  selfAttentionNode.outTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_self_attention")};
  opGraph.nodes.push_back(selfAttentionNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t AddFIA(atb::GraphParam &opGraph,
               const FusionAttentionParam<NormParamType> &param,
               std::map<std::string, uint32_t> &tensorMap) {
  if (!param.enableRopeQuantKvcache && param.needUpdateKVCache && !param.isFA) {
    if (!param.enableXattention) {
      CHECK_OPERATION_STATUS_RETURN(
          AddPaKVCacheOperation(opGraph, param, tensorMap));
    }
  }

  atb::Node fiaNode;
  fiaNode.operation = new atb_speed::common::FusedInferAttentionV2Operation(
      "FusedInferAttentionNode", param.aclnnFusedInferAttnParam);
  if (param.aclnnFusedInferAttnParam.enablePa) { // if pa
    fiaNode.inTensorIds = {
        GetTensorIdx(tensorMap,
                     param.aclnnFusedInferAttnParam.inputLayout == "BSND"
                         ? "intermediate_q_bsnd"
                         : "intermediate_q"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_v_cache"),
        GetTensorIdx(tensorMap, "in_seq_len"),
        GetTensorIdx(tensorMap, "in_attention_mask"),
    };
    fiaNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_len"));
    fiaNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_block_tables"));
  } else {
    fiaNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "intermediate_v"),
        GetTensorIdx(tensorMap, "in_seq_len"),
        GetTensorIdx(tensorMap, "in_attention_mask"),
    };
    if (tensorMap, param.aclnnFusedInferAttnParam.inputLayout == "TND") {
      fiaNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_len"));
    }
  }

  fiaNode.inTensorReshapeFuncs.resize(fiaNode.inTensorIds.size());
  auto reshapeToBSNDFunc = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    if (oldShape.dimNum == 3) {
      size_t dim = 0;
      int batchSize = *param.bs;
      if (batchSize <= 0) {
        batchSize = 1;
      }
      newShape.dims[dim++] = batchSize;
      newShape.dims[dim++] = oldShape.dims[0] / batchSize;
      newShape.dims[dim++] = oldShape.dims[1];
      newShape.dims[dim++] = oldShape.dims[2];
      newShape.dimNum = dim;
    } else {
      newShape = oldShape;
    }
  };
  auto reshapeToBnBsHFunc = [=](const atb::Dims &oldShape,
                                atb::Dims &newShape) {
    if (oldShape.dimNum == 4) {
      newShape.dimNum = 3;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = oldShape.dims[1];
      newShape.dims[2] = oldShape.dims[2] * oldShape.dims[3];
    } else {
      newShape = oldShape;
    }
  };
  if (tensorMap, param.aclnnFusedInferAttnParam.inputLayout == "BSND") {
    fiaNode.inTensorReshapeFuncs.at(0) = reshapeToBSNDFunc;
  }
  if (param.aclnnFusedInferAttnParam.enablePa) { // kvcache to BnBsH
    fiaNode.inTensorReshapeFuncs.at(1) = reshapeToBnBsHFunc;
    fiaNode.inTensorReshapeFuncs.at(2) = reshapeToBnBsHFunc;
  }
  fiaNode.outTensorIds = {GetTensorIdx(
      tensorMap, param.aclnnFusedInferAttnParam.inputLayout == "BSND"
                     ? "intermediate_self_attention_bsnd"
                     : "intermediate_self_attention")};
  opGraph.nodes.push_back(fiaNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t AddFaKVCacheOperation(atb::GraphParam &opGraph,
                              const FusionAttentionParam<NormParamType> &param,
                              std::map<std::string, uint32_t> &tensorMap) {
  // moveKCache Node
  atb::infer::KvCacheParam kvCacheParam;
  atb::Node moveKCacheNode;
  CHECK_OPERATION_STATUS_RETURN(
      atb::CreateOperation(kvCacheParam, &moveKCacheNode.operation));
  moveKCacheNode.inTensorIds = {
      param.aclnnIncreAttentionParam.hasKVQuant
          ? GetTensorIdx(tensorMap, "intermediate_k_int8")
          : GetTensorIdx(tensorMap, "intermediate_k"),
      GetTensorIdx(tensorMap, "in_layer_id"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_token_offset"),
      GetTensorIdx(tensorMap, "in_seq_len"),
  };
  moveKCacheNode.inTensorReshapeFuncs.resize(moveKCacheNode.inTensorIds.size());
  if (param.isOneRecCrossAttention) {
    moveKCacheNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape,
                                                    atb::Dims &newShape) {
      if (oldShape.dimNum == 2) {
        newShape = oldShape;
      } else {
        SqueezeBatchAndHiddenSize(oldShape, newShape);
      }
    };
  } else {
    moveKCacheNode.inTensorReshapeFuncs.at(0) = // 0: [B,S,N,D]=>[BS,ND]
        &SqueezeBatchAndHiddenSize;
  }
  moveKCacheNode.inTensorReshapeFuncs.at(2) = [=]( // 2: [B,S,ND]=>[1,B,S,ND]
                                                  const atb::Dims &oldShape,
                                                  atb::Dims &newShape) {
    UnsqueezeAxis(oldShape, newShape, 0);
  };
  moveKCacheNode.outTensorIds = {};
  opGraph.nodes.push_back(moveKCacheNode);

  atb::Node moveVCacheNode;
  CHECK_OPERATION_STATUS_RETURN(
      atb::CreateOperation(kvCacheParam, &moveVCacheNode.operation));
  moveVCacheNode.inTensorIds = {
      param.aclnnIncreAttentionParam.hasKVQuant
          ? GetTensorIdx(tensorMap, "intermediate_v_int8")
          : GetTensorIdx(tensorMap, "intermediate_v"),
      GetTensorIdx(tensorMap, "in_layer_id"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      GetTensorIdx(tensorMap, "in_token_offset"),
      GetTensorIdx(tensorMap, "in_seq_len"),
  };
  moveVCacheNode.inTensorReshapeFuncs.resize(moveVCacheNode.inTensorIds.size());
  if (param.isOneRecCrossAttention) {
    moveVCacheNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape,
                                                    atb::Dims &newShape) {
      if (oldShape.dimNum == 2) {
        newShape = oldShape;
      } else {
        SqueezeBatchAndHiddenSize(oldShape, newShape);
      }
    };
  } else {
    moveVCacheNode.inTensorReshapeFuncs.at(0) = // 0: [B,S,N,D]=>[BS,ND]
        &SqueezeBatchAndHiddenSize;
  }
  moveVCacheNode.inTensorReshapeFuncs.at(2) = [=]( // 2: [B,S,ND]=>[1,B,S,ND]
                                                  const atb::Dims &oldShape,
                                                  atb::Dims &newShape) {
    UnsqueezeAxis(oldShape, newShape, 0);
  };
  moveVCacheNode.outTensorIds = {};
  opGraph.nodes.push_back(moveVCacheNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t AddPaKVCacheOperation(atb::GraphParam &opGraph,
                              const FusionAttentionParam<NormParamType> &param,
                              std::map<std::string, uint32_t> &tensorMap) {
  // ReshapeAndCache Node
  atb::Node reshapeAndCacheNode;
  if (param.enableOmniattention && param.isomnicompressed) {
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
        param.reshapeCacheOmniParm, &reshapeAndCacheNode.operation));
  } else {
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
        param.reshapeCacheParm, &reshapeAndCacheNode.operation));
  }
  reshapeAndCacheNode.inTensorIds = {
      param.pageAttentionParam.quantType ==
                  atb::infer::PagedAttentionParam::QuantType::
                      TYPE_DEQUANT_FUSION ||
              param.pageAttentionParam.quantType ==
                  atb::infer::PagedAttentionParam::QuantType::
                      TYPE_QUANT_QKV_ONLINE
          ? GetTensorIdx(tensorMap, "intermediate_k_int8")
          : GetTensorIdx(tensorMap, "intermediate_k"),
      param.pageAttentionParam.quantType ==
                  atb::infer::PagedAttentionParam::QuantType::
                      TYPE_DEQUANT_FUSION ||
              param.pageAttentionParam.quantType ==
                  atb::infer::PagedAttentionParam::QuantType::
                      TYPE_QUANT_QKV_ONLINE
          ? GetTensorIdx(tensorMap, "intermediate_v_int8")
          : GetTensorIdx(tensorMap, "intermediate_v"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      param.isOneRecCrossAttention
          ? GetTensorIdx(tensorMap, "in_cross_attn_slots")
          : GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
  };
  if (param.reshapeCacheParm.compressType ==
      atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_KVHEAD) {
    reshapeAndCacheNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_batch_wins"));
    reshapeAndCacheNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_seq_len"));
  } else if (param.reshapeCacheParm.compressType ==
                 atb::infer::ReshapeAndCacheParam::CompressType::
                     COMPRESS_TYPE_KVHEAD_ROPE ||
             param.isomnicompressed) {
    reshapeAndCacheNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_batch_wins"));
    reshapeAndCacheNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_reshape_seq_len"));
    reshapeAndCacheNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_pffset_index"));
  }
  if (param.isOneRecCrossAttention) {
    const int64_t kvHeadNum = param.selfAttentionParam.kvHeadNum > 0
                                  ? param.selfAttentionParam.kvHeadNum
                                  : param.pageAttentionParam.kvHeadNum;
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(2);
    auto reshapeKVFunc = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
      int64_t ntokens = oldShape.dims[0];
      if (oldShape.dimNum == 3) {
        ntokens = oldShape.dims[0] * oldShape.dims[1];
      }
      newShape.dims[0] = ntokens;
      newShape.dims[1] = kvHeadNum;
      newShape.dims[2] = param.headDim;
      newShape.dimNum = 3;
    };
    reshapeAndCacheNode.inTensorReshapeFuncs.at(0) = reshapeKVFunc;
    reshapeAndCacheNode.inTensorReshapeFuncs.at(1) = reshapeKVFunc;
  }
  reshapeAndCacheNode.outTensorIds = {
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
  };
  opGraph.nodes.push_back(reshapeAndCacheNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t
AddDecodeKVCacheOperation(atb::GraphParam &opGraph,
                          const FusionAttentionParam<NormParamType> &param,
                          std::map<std::string, uint32_t> &tensorMap) {
  atb::Node reshapeDecodeKVCacheNode;
  atb_speed::common::AclNNReshapeDecodeKvCacheParam reshapeDecodeKVCacheParam;

  reshapeDecodeKVCacheNode.operation =
      new atb_speed::common::ReshapeDecodeKvCacheOperation(
          "ReshapeDecodeKVCacheNode", reshapeDecodeKVCacheParam);
  reshapeDecodeKVCacheNode.inTensorIds = {
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      GetTensorIdx(tensorMap, "intermediate_k"),
      GetTensorIdx(tensorMap, "intermediate_v"),
      GetTensorIdx(tensorMap, "in_block_tables"),
      GetTensorIdx(tensorMap, "in_current_round")};
  reshapeDecodeKVCacheNode.outTensorIds = {
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
  };
  opGraph.nodes.push_back(reshapeDecodeKVCacheNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t
ConstructAclNNDecoderNode(atb::Node &selfAttentionNode,
                          const FusionAttentionParam<NormParamType> &param,
                          std::map<std::string, uint32_t> &tensorMap) {
  // translatedFA QKV [B,S,H] PA Q [B,S,N,D] KV [num_blocks,block_size,ND]
  // translatedFA [B,S,H] PA [BS,N,D]
  selfAttentionNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      GetTensorIdx(tensorMap, "in_attention_mask")};
  if (param.aclnnIncreAttentionParam.isFA) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_token_offset"));
  } else {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_seq_len"));
  }
  selfAttentionNode.inTensorIds.push_back(
      GetTensorIdx(tensorMap, "in_block_tables"));
  if (param.aclnnIncreAttentionParam.hasKVQuant) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_k_dequant_scale"));
    if (param.aclnnIncreAttentionParam.hasQuantOffset) {
      selfAttentionNode.inTensorIds.push_back(
          GetTensorIdx(tensorMap, "in_k_dequant_offset"));
    }
  }
  selfAttentionNode.operation = new atb_speed::common::AttnOperation(
      "AclNNAttentionNode", param.aclnnIncreAttentionParam);
  selfAttentionNode.inTensorReshapeFuncs.resize(
      selfAttentionNode.inTensorIds.size());
  if (param.isFA) {
    selfAttentionNode.inTensorReshapeFuncs.at(0) = &SqueezeHeadNumHeadDim;
  } else {
    selfAttentionNode.inTensorReshapeFuncs.at(0) =
        [=](const atb::Dims &oldShape, atb::Dims &newShape) {
          UnsqueezeAxis(oldShape, newShape, 1);
        };
    selfAttentionNode.inTensorReshapeFuncs.at(1) =
        &SqueezeHeadNumHeadDim; // 1: in_k_cache
    selfAttentionNode.inTensorReshapeFuncs.at(2) =
        &SqueezeHeadNumHeadDim; // 2: in_v_cache
  }
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t ConstructFaNode(atb::Node &selfAttentionNode,
                        const FusionAttentionParam<NormParamType> &param,
                        std::map<std::string, uint32_t> &tensorMap) {
  // translated[nTokens, vHiddenSize] translated[nTokens, vHiddenSize]
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
      param.selfAttentionParam, &selfAttentionNode.operation));
  selfAttentionNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "intermediate_k"),
      GetTensorIdx(tensorMap, "intermediate_v"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
  };
  if (param.selfAttentionParam.maskType !=
      atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_attention_mask"));
  }
  selfAttentionNode.inTensorIds.push_back(
      GetTensorIdx(tensorMap, "in_token_offset"));
  selfAttentionNode.inTensorIds.push_back(
      GetTensorIdx(tensorMap, "in_seq_len"));
  selfAttentionNode.inTensorIds.push_back(
      GetTensorIdx(tensorMap, "in_layer_id"));
  if (param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_slopes"));
  }
  selfAttentionNode.inTensorReshapeFuncs.resize(
      selfAttentionNode.inTensorIds.size());
  selfAttentionNode.inTensorReshapeFuncs.at(0) =
      &SqueezeBatchAndHiddenSize; // 0: [B,S,N,D]=>[BS,ND]
  selfAttentionNode.inTensorReshapeFuncs.at(1) =
      &SqueezeBatchAndHiddenSize; // 1: [B,S,N,D]=>[BS,ND]
  selfAttentionNode.inTensorReshapeFuncs.at(2) =
      &SqueezeBatchAndHiddenSize;                     // 2: [B,S,N,D]=>[BS,ND]
  selfAttentionNode.inTensorReshapeFuncs.at(3) = [=]( // 3: [BS,N,D]=>[1,BS,N,D]
                                                     const atb::Dims &oldShape,
                                                     atb::Dims &newShape) {
    UnsqueezeAxis(oldShape, newShape, 0);
  };
  selfAttentionNode.inTensorReshapeFuncs.at(4) = [=]( // 4: [BS,N,D]=>[1,BS,N,D]
                                                     const atb::Dims &oldShape,
                                                     atb::Dims &newShape) {
    UnsqueezeAxis(oldShape, newShape, 0);
  };
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t ConstructPaEncoderNode(atb::Node &selfAttentionNode,
                               const FusionAttentionParam<NormParamType> &param,
                               std::map<std::string, uint32_t> &tensorMap) {
  // translated[BS, N, D] translated[BS, N, D]
  ATB_SPEED_LOG_DEBUG("Enter ConstructPaEncoderNode");
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
      param.selfAttentionParam, &selfAttentionNode.operation));
  // if use for beam search, the intermediate_k and intermediate_v are the
  // prefill key and value
  if (param.isPrefill && param.enableXattention) {
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "in_decode_k_cache"),
        GetTensorIdx(tensorMap, "in_decode_v_cache"),
    };
  } else {
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "intermediate_v"),
    };
  }
  if (param.selfAttentionParam.maskType !=
      atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_attention_mask"));
  }
  selfAttentionNode.inTensorIds.push_back(
      GetTensorIdx(tensorMap, "in_seq_len"));
  if (param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_slopes"));
  }
  selfAttentionNode.inTensorReshapeFuncs.resize(
      selfAttentionNode.inTensorIds.size());
  if (param.attnBackend == atb_speed::common::OpBackend::ACLNN) {
    selfAttentionNode.inTensorReshapeFuncs.at(0) =
        &SqueezeBatchAndHiddenSize; // 0: [B,S,N,D]=>[BS,ND]
    selfAttentionNode.inTensorReshapeFuncs.at(1) =
        &SqueezeBatchAndHiddenSize; // 1: [B,S,N,D]=>[BS,ND]
    selfAttentionNode.inTensorReshapeFuncs.at(2) =
        &SqueezeBatchAndHiddenSize; // 2: [B,S,N,D]=>[BS,ND]
  }

  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t
ConstructPrefixEncoderNode(atb::Node &selfAttentionNode,
                           const FusionAttentionParam<NormParamType> &param,
                           std::map<std::string, uint32_t> &tensorMap) {
  // translated[BS, N, D] translated[BS, N, D]
  ATB_SPEED_LOG_DEBUG("Enter ConstructPaEncoderNode");
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
      param.selfAttentionParam, &selfAttentionNode.operation));
  selfAttentionNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      GetTensorIdx(tensorMap, "in_block_tables"),
  };
  if (param.selfAttentionParam.maskType !=
      atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_attention_mask"));
  }
  selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_len"));
  selfAttentionNode.inTensorIds.push_back(
      GetTensorIdx(tensorMap, "in_seq_len"));
  if (param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_slopes"));
  }
  selfAttentionNode.inTensorReshapeFuncs.resize(
      selfAttentionNode.inTensorIds.size());
  selfAttentionNode.inTensorReshapeFuncs.at(0) =
      &SqueezeBatchAndHiddenSize; // 0: [B,S,N,D]=>[BS,ND]
  if (param.attnBackend == atb_speed::common::OpBackend::ACLNN) {
    selfAttentionNode.inTensorReshapeFuncs.at(0) =
        &SqueezeBatchAndHiddenSize; // 0: [B,S,N,D]=>[BS,ND]
    selfAttentionNode.inTensorReshapeFuncs.at(1) =
        &SqueezeBatchAndHiddenSize; // 1: [B,S,N,D]=>[BS,ND]
    selfAttentionNode.inTensorReshapeFuncs.at(2) =
        &SqueezeBatchAndHiddenSize; // 2: [B,S,N,D]=>[BS,ND]
  }
  return atb::NO_ERROR;
}
template <typename NormParamType>
int64_t ConstructPaDecoderNode(atb::Node &selfAttentionNode,
                               const FusionAttentionParam<NormParamType> &param,
                               std::map<std::string, uint32_t> &tensorMap) {
  // translated[num_tokens, N, D] [num_block,block_size,N,D]
  // translated[num_tokens, num_head, head_size]
  if (!param.isPrefill && param.enableAclGraphPagedAttention) {
    atb::customize::CustomPagedAttentionParam customPagedAttentionParam(
        param.pageAttentionParam);
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
        customPagedAttentionParam, &selfAttentionNode.operation));
  } else {
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
        param.pageAttentionParam, &selfAttentionNode.operation));
  }
  selfAttentionNode.inTensorIds = {
      param.pageAttentionParam.quantType ==
              atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE
          ? GetTensorIdx(tensorMap, "intermediate_q_int8")
          : GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      GetTensorIdx(tensorMap, "in_block_tables"),
  };
  if (param.pageAttentionParam.compressType ==
          atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD ||
      param.pageAttentionParam.compressType ==
          atb::infer::PagedAttentionParam::CompressType::
              COMPRESS_TYPE_KVHEAD_ROPE) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_ra_seq_len"));
  } else {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_seq_len"));
  }
  if (param.pageAttentionParam.maskType !=
      atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_attention_mask"));
  }
  if (param.pageAttentionParam.calcType ==
      atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_q_len"));
  }
  if (param.pageAttentionParam.quantType ==
      atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_k_dequant_scale"));
    if (param.pageAttentionParam.hasQuantOffset) {
      selfAttentionNode.inTensorIds.push_back(
          GetTensorIdx(tensorMap, "in_k_dequant_offset"));
    }
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_v_dequant_scale"));
    if (param.pageAttentionParam.hasQuantOffset) {
      selfAttentionNode.inTensorIds.push_back(
          GetTensorIdx(tensorMap, "in_v_dequant_offset"));
    }
  }
  if (param.pageAttentionParam.quantType ==
      atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_qk_descale"));
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "fa3_v_quant_scale"));
  }
  if (param.pageAttentionParam.compressType ==
      atb::infer::PagedAttentionParam::CompressType::
          COMPRESS_TYPE_KVHEAD_ROPE) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_ra_offset"));
  }
  if (param.pageAttentionParam.scaleType ==
      atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_log_n_scale"));
  }
  if (!param.isPrefill && param.enableAclGraphPagedAttention) {
    selfAttentionNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "paged_attention_tiling_data"));
  }
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t
ConstructDecodeKVCacheNode(atb::GraphParam &opGraph,
                           const FusionAttentionParam<NormParamType> &param,
                           std::map<std::string, uint32_t> &tensorMap) {
  atb::Node xAttentionNode;
  atb_speed::common::AclNNXAttentionParam xAttentionParam;
  xAttentionParam.layerId = param.layerId;
  xAttentionParam.scaleValue = param.selfAttentionParam.qkScale;
  xAttentionNode.operation = new atb_speed::common::XAttentionOperation(
      "XAttentionNode", xAttentionParam);
  // Now,in_decode_k_cache is the shared k_cache,in_k_cache is unshared k_cache.
  xAttentionNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "in_decode_k_cache"),
      GetTensorIdx(tensorMap, "in_decode_v_cache"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      GetTensorIdx(tensorMap, "in_block_tables"),
      GetTensorIdx(tensorMap, "in_seq_len"),
      GetTensorIdx(tensorMap, "in_current_round"),
  };
  xAttentionNode.outTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_self_attention")};
  opGraph.nodes.push_back(xAttentionNode);
  return atb::NO_ERROR;
}

template int64_t AddFaKVCacheOperation(
    atb::GraphParam &opGraph,
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t AddPaKVCacheOperation(
    atb::GraphParam &opGraph,
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t
AddSelfAttention(atb::GraphParam &opGraph,
                 const FusionAttentionParam<atb::infer::RmsNormParam> &param,
                 std::map<std::string, uint32_t> &tensorMap);
template int64_t
AddFIA(atb::GraphParam &opGraph,
       const FusionAttentionParam<atb::infer::RmsNormParam> &param,
       std::map<std::string, uint32_t> &tensorMap);
template int64_t
ConstructFaNode(atb::Node &selfAttentionNode,
                const FusionAttentionParam<atb::infer::RmsNormParam> &param,
                std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaEncoderNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPrefixEncoderNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaDecoderNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);

template int64_t AddFaKVCacheOperation(
    atb::GraphParam &opGraph,
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t AddPaKVCacheOperation(
    atb::GraphParam &opGraph,
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t
AddSelfAttention(atb::GraphParam &opGraph,
                 const FusionAttentionParam<atb::infer::LayerNormParam> &param,
                 std::map<std::string, uint32_t> &tensorMap);
template int64_t
AddFIA(atb::GraphParam &opGraph,
       const FusionAttentionParam<atb::infer::LayerNormParam> &param,
       std::map<std::string, uint32_t> &tensorMap);
template int64_t
ConstructFaNode(atb::Node &selfAttentionNode,
                const FusionAttentionParam<atb::infer::LayerNormParam> &param,
                std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaEncoderNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPrefixEncoderNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaDecoderNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);

namespace {

template <typename NormParamType>
int64_t
ResolvePerRankHeadNum(const FusionAttentionParam<NormParamType> &param) {
  if (param.selfAttentionParam.headNum > 0) {
    return param.selfAttentionParam.headNum;
  }
  if (param.pageAttentionParam.headNum > 0) {
    return param.pageAttentionParam.headNum;
  }
  if (param.aclnnIncreAttentionParam.headNum > 0) {
    return param.aclnnIncreAttentionParam.headNum;
  }
  if (param.aclnnFAScoreParam.headNum > 0) {
    return param.aclnnFAScoreParam.headNum;
  }
  return 0;
}

template <typename NormParamType>
int64_t
ResolvePerRankKvHeadNum(const FusionAttentionParam<NormParamType> &param) {
  if (param.selfAttentionParam.kvHeadNum > 0) {
    return param.selfAttentionParam.kvHeadNum;
  }
  if (param.pageAttentionParam.kvHeadNum > 0) {
    return param.pageAttentionParam.kvHeadNum;
  }
  if (param.aclnnIncreAttentionParam.kvHeadNum > 0) {
    return param.aclnnIncreAttentionParam.kvHeadNum;
  }
  if (param.aclnnFusedInferAttnParam.numKeyValueHeads > 0) {
    return param.aclnnFusedInferAttnParam.numKeyValueHeads;
  }
  return 0;
}

int64_t ValidateRequiredTensor(
    const std::map<std::string, uint32_t> &tensorMap, const char *tensorName,
    const char *caller) {
  if (tensorMap.find(tensorName) == tensorMap.end()) {
    ATB_SPEED_LOG_ERROR(caller << " missing required tensor: " << tensorName);
    return atb::ERROR_INVALID_PARAM;
  }
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t
ValidatePerRankKvLayout(const FusionAttentionParam<NormParamType> &param,
                        const char *caller) {
  const int64_t headNum = ResolvePerRankHeadNum(param);
  const int64_t kvHeadNum = ResolvePerRankKvHeadNum(param);
  const int64_t headDim = param.headDim;

  if (headNum <= 0 || kvHeadNum <= 0 || headDim <= 0 || kvHeadNum > headNum ||
      headNum % kvHeadNum != 0) {
    ATB_SPEED_LOG_ERROR(caller << " invalid per-rank attention layout: "
                               << "headNum=" << headNum << ", kvHeadNum="
                               << kvHeadNum << ", headDim=" << headDim);
    return atb::ERROR_INVALID_PARAM;
  }

  return atb::NO_ERROR;
}

} // namespace

template <typename NormParamType>
int64_t ConstructOneRecCrossFusedInferAttentionNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap) {
  CHECK_OPERATION_STATUS_RETURN(ValidatePerRankKvLayout(
      param, "ConstructOneRecCrossFusedInferAttentionNode"));
  CHECK_OPERATION_STATUS_RETURN(ValidateRequiredTensor(
      tensorMap, "intermediate_q",
      "ConstructOneRecCrossFusedInferAttentionNode"));
  CHECK_OPERATION_STATUS_RETURN(ValidateRequiredTensor(
      tensorMap, "in_cross_k_cache",
      "ConstructOneRecCrossFusedInferAttentionNode"));
  CHECK_OPERATION_STATUS_RETURN(ValidateRequiredTensor(
      tensorMap, "in_cross_v_cache",
      "ConstructOneRecCrossFusedInferAttentionNode"));
  CHECK_OPERATION_STATUS_RETURN(ValidateRequiredTensor(
      tensorMap, "cross_kv_len",
      "ConstructOneRecCrossFusedInferAttentionNode"));
  ATB_SPEED_LOG_ERROR(
      "OneRecCrossAttention ConstructOneRecCrossFusedInferAttentionNode: "
      << "isOneRecCrossAttention=" << param.isOneRecCrossAttention
      << ", isPrefill=" << param.isPrefill
      << ", enableXattention=" << param.enableXattention
      << ", consume_k=in_cross_k_cache, consume_v=in_cross_v_cache");
  selfAttentionNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "in_cross_k_cache"),
      GetTensorIdx(tensorMap, "in_cross_v_cache"),
      GetTensorIdx(tensorMap, "cross_kv_len"),
  };
  selfAttentionNode.operation =
      new atb_speed::common::FusedInferAttentionV2Operation(
          "FusedInferAttentionNode", param.aclnnFusedInferAttnParam);
  selfAttentionNode.inTensorReshapeFuncs.resize(
      selfAttentionNode.inTensorIds.size());

  const int64_t kvHeadNum = ResolvePerRankKvHeadNum(param);
  const int64_t headNum = ResolvePerRankHeadNum(param);
  const bool is_grouped_query_attention =
      kvHeadNum > 0 && headNum > 0 && kvHeadNum != headNum;
  auto reshapeKVFunc = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    size_t dim = 0;
    int64_t resolvedKvHeadNum = kvHeadNum;
    if (is_grouped_query_attention) {
      const int64_t hiddenWidth =
          oldShape.dimNum > 0 ? oldShape.dims[oldShape.dimNum - 1] : 0;
      if (param.headDim > 0 && hiddenWidth > 0 &&
          hiddenWidth % param.headDim == 0) {
        resolvedKvHeadNum = hiddenWidth / param.headDim;
      }
    }
    newShape.dims[dim++] = oldShape.dims[0];
    newShape.dims[dim++] = oldShape.dims[1];
    newShape.dims[dim++] = resolvedKvHeadNum;
    newShape.dims[dim++] = param.headDim;
    newShape.dimNum = dim;
  };
  selfAttentionNode.inTensorReshapeFuncs.at(1) = reshapeKVFunc;
  selfAttentionNode.inTensorReshapeFuncs.at(2) = reshapeKVFunc;
  selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape,
                                                     atb::Dims &newShape) {
    size_t dim = 0;
    int32_t batchSize = *param.bs;
    if (batchSize <= 0) {
      batchSize = 1;
    }
    int64_t resolvedHeadNum = headNum;
    if (is_grouped_query_attention) {
      const int64_t hiddenWidth = oldShape.dimNum > 1 ? oldShape.dims[1] : 0;
      if (param.headDim > 0 && hiddenWidth > 0 &&
          hiddenWidth % param.headDim == 0) {
        resolvedHeadNum = hiddenWidth / param.headDim;
      }
    }
    newShape.dims[dim++] = batchSize;
    newShape.dims[dim++] = oldShape.dims[0] / batchSize;
    newShape.dims[dim++] = resolvedHeadNum;
    newShape.dims[dim++] = param.headDim;
    newShape.dimNum = dim;
  };
  return atb::NO_ERROR;
}

/// OneRec FAS (FlashAttentionScore) attention node constructor.
/// This function constructs the FAS attention node for OneRec prefill,
/// which replaces the standard SelfAttention encoder path.
template <typename NormParamType>
int64_t ConstructFAScoreNode(atb::Node &selfAttentionNode,
                             const FusionAttentionParam<NormParamType> &param,
                             std::map<std::string, uint32_t> &tensorMap) {
  CHECK_OPERATION_STATUS_RETURN(
      ValidatePerRankKvLayout(param, "ConstructFAScoreNode"));
  selfAttentionNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "intermediate_k"),
      GetTensorIdx(tensorMap, "intermediate_v"),
      GetTensorIdx(tensorMap, "in_attention_mask"),
  };

  selfAttentionNode.operation = new atb_speed::common::FlashAttnScoreOperation(
      "FlashAttnScoreOperation", param.aclnnFAScoreParam);

  selfAttentionNode.inTensorReshapeFuncs.resize(
      selfAttentionNode.inTensorIds.size());
  const int64_t headNum = ResolvePerRankHeadNum(param);
  const int64_t kvHeadNum = ResolvePerRankKvHeadNum(param);

  auto reshapeToBsnhd = [=](const atb::Dims &oldShape, atb::Dims &newShape,
                            int64_t targetHeadNum, const char *tensorName) {
    size_t dim = 0;
    auto seqlen = *param.seqlen;
    const int64_t tokens = oldShape.dimNum > 0 ? oldShape.dims[0] : -1;
    if (seqlen <= 0 || oldShape.dimNum == 0 || oldShape.dims[0] % seqlen != 0) {
      ATB_SPEED_LOG_ERROR("ConstructFAScoreNode reshape "
                          << tensorName << " failed: "
                          << "seqlen=" << seqlen << ", tokens=" << tokens
                          << ", headNum=" << headNum << ", kvHeadNum="
                          << kvHeadNum << ", headDim=" << param.headDim);
      newShape = oldShape;
      return;
    }
    newShape.dims[dim++] = oldShape.dims[0] / seqlen;
    *param.bs = oldShape.dims[0] / seqlen;
    newShape.dims[dim++] = seqlen;
    newShape.dims[dim++] = targetHeadNum;
    newShape.dims[dim++] = param.headDim;
    newShape.dimNum = dim;
  };
  selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape,
                                                     atb::Dims &newShape) {
    reshapeToBsnhd(oldShape, newShape, headNum, "Q");
  };
  selfAttentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape,
                                                     atb::Dims &newShape) {
    reshapeToBsnhd(oldShape, newShape, kvHeadNum, "K");
  };
  selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape,
                                                     atb::Dims &newShape) {
    reshapeToBsnhd(oldShape, newShape, kvHeadNum, "V");
  };

  return atb::NO_ERROR;
}

template int64_t ConstructFAScoreNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructFAScoreNode(
    atb::Node &selfAttentionNode,
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);

} // namespace common
} // namespace atb_speed
