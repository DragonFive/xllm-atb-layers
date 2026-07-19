/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "operations/aclnn/ops/cross_attention_operation.h"

#include "aclnn_cross_attention.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

namespace {
constexpr uint32_t kInputNum = 5;
constexpr uint32_t kOutputNum = 1;
constexpr size_t kQueryIndex = 0;
constexpr size_t kKeyIndex = 1;
constexpr size_t kValueIndex = 2;
constexpr size_t kBlockTableIndex = 3;
constexpr size_t kActualKvLensIndex = 4;
constexpr size_t kAttentionOutIndex = 0;
constexpr const char *kInputNames[kInputNum] = {
    "query", "key", "value", "block_table", "actual_kv_lens"};
} // namespace

CrossAttentionOperation::CrossAttentionOperation(
    const std::string &name, AclNNCrossAttentionParam param)
    : AclNNOperation(name), param_(param) {}

uint32_t CrossAttentionOperation::GetInputNum() const { return kInputNum; }

uint32_t CrossAttentionOperation::GetOutputNum() const { return kOutputNum; }

atb::Status CrossAttentionOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const {
  outTensorDescs.at(kAttentionOutIndex) = inTensorDescs.at(kQueryIndex);
  return atb::NO_ERROR;
}

atb::Status CrossAttentionOperation::CreateAclNNInTensorVariantPack(
    const atb::VariantPack &variantPack) {
  if (variantPack.inTensors.size() != kInputNum) {
    ATB_SPEED_LOG_ERROR(opName_ << " expects " << kInputNum << " inputs, got "
                                << variantPack.inTensors.size());
    return atb::ERROR_INVALID_PARAM;
  }
  for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
    if (variantPack.inTensors.at(i).deviceData == nullptr) {
      ATB_SPEED_LOG_ERROR(opName_ << " requires device tensor for input "
                                  << kInputNames[i] << " (index " << i << ")");
      return atb::ERROR_INVALID_PARAM;
    }
  }
  return AclNNOperation::CreateAclNNInTensorVariantPack(variantPack);
}

int CrossAttentionOperation::SetAclNNWorkspaceExecutor() {
  auto &variantPack = this->aclnnOpCache_->aclnnVariantPack;
  const int ret = aclnnCrossAttentionGetWorkspaceSize(
      variantPack.aclInTensors.at(kQueryIndex)->tensor,
      variantPack.aclInTensors.at(kKeyIndex)->tensor,
      variantPack.aclInTensors.at(kValueIndex)->tensor,
      variantPack.aclInTensors.at(kBlockTableIndex)->tensor,
      variantPack.aclInTensors.at(kActualKvLensIndex)->tensor,
      param_.scaleValue,
      variantPack.aclOutTensors.at(kAttentionOutIndex)->tensor,
      &this->aclnnOpCache_->workspaceSize,
      &this->aclnnOpCache_->aclExecutor);
  ATB_SPEED_LOG_DEBUG(opName_ << " aclnnCrossAttentionGetWorkspaceSize ret="
                              << ret << ", workspaceSize="
                              << this->aclnnOpCache_->workspaceSize);
  return ret;
}

int CrossAttentionOperation::ExecuteAclNNOp(uint8_t *workspace,
                                            aclrtStream &stream) {
  return aclnnCrossAttention(workspace, this->aclnnOpCache_->workspaceSize,
                             this->aclnnOpCache_->aclExecutor, stream);
}

} // namespace common
} // namespace atb_speed
