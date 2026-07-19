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
#ifndef ATB_SPEED_OPERATIONS_ACLNN_CROSS_ATTENTION_OPERATION_H
#define ATB_SPEED_OPERATIONS_ACLNN_CROSS_ATTENTION_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

struct AclNNCrossAttentionParam {
  double scaleValue = 1.0;
};

class CrossAttentionOperation : public AclNNOperation {
public:
  explicit CrossAttentionOperation(const std::string &name,
                                   AclNNCrossAttentionParam param);
  ~CrossAttentionOperation() override = default;

  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status InferShape(
      const atb::SVector<atb::TensorDesc> &inTensorDescs,
      atb::SVector<atb::TensorDesc> &outTensorDescs) const override;

protected:
  atb::Status CreateAclNNInTensorVariantPack(
      const atb::VariantPack &variantPack) override;
  int SetAclNNWorkspaceExecutor() override;
  int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

private:
  AclNNCrossAttentionParam param_;
};

} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_OPERATIONS_ACLNN_CROSS_ATTENTION_OPERATION_H
