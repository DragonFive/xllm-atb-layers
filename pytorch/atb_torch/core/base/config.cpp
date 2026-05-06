/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include <algorithm>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include "utils.h"
#include "atb_speed/log.h"
#include "config.h"

namespace atb_torch {
namespace {
constexpr uint64_t kDefaultWorkspaceSize = 1024 * 1024 * 600;
using WorkspaceCacheKey = std::pair<int32_t, uint64_t>;

struct WorkspaceEntry {
    uint64_t size = 0;
    torch::Tensor tensor;
};

class WorkspaceCacheKeyHash final {
public:
    size_t operator()(const WorkspaceCacheKey &key) const
    {
        size_t deviceHash = std::hash<int32_t>()(key.first);
        size_t streamHash = std::hash<uint64_t>()(key.second);
        return deviceHash ^ (streamHash + 0x9e3779b9 + (deviceHash << 6) + (deviceHash >> 2));
    }
};

thread_local std::unordered_map<WorkspaceCacheKey, WorkspaceEntry, WorkspaceCacheKeyHash> g_workspaceCache;

torch::Tensor CreateWorkspaceTensor(uint64_t size)
{
    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;

    constexpr int kKb1 = 1024;
    tensorDesc.shape.dimNum = 2;
    tensorDesc.shape.dims[0] = kKb1;
    tensorDesc.shape.dims[1] = size / kKb1 + 1;

    return Utils::CreateAtTensorFromTensorDesc(tensorDesc);
}
}  // namespace

Config &Config::Instance()
{
    static Config instance;
    return instance;
}

Config::Config()
{
    isUseTilingCopyStream_ = IsEnable("ATB_USE_TILING_COPY_STREAM");
    isTorchTensorFormatCast_ = IsEnable("ATB_TORCH_TENSOR_FORMAT_CAST");
    const char *taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
    const char *blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
    isTaskQueueEnable_ = !((taskQueueEnv != nullptr && std::string(taskQueueEnv) == "0") ||
                           (blockingEnv != nullptr && std::string(blockingEnv) == "1"));
    defaultWorkspaceSize_ = kDefaultWorkspaceSize;

    ATB_SPEED_LOG_DEBUG("Config [IsTorchTensorFormatCast:" << isTorchTensorFormatCast_
                        << ", IsUseTilingCopyStream:" << isUseTilingCopyStream_
                        << ", DefaultWorkspaceSize:" << defaultWorkspaceSize_
                        << ", IsTaskQueueEnable:" << isTaskQueueEnable_ << "]");
}

Config::~Config() {}

bool Config::IsEnable(const char *env, bool enable) const
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsUseTilingCopyStream() const { return isUseTilingCopyStream_; }

bool Config::IsTorchTensorFormatCast() const { return isTorchTensorFormatCast_; };

bool Config::IsTaskQueueEnable() const { return isTaskQueueEnable_; }

void *Config::GetWorkspace(uint64_t size, int32_t deviceId, uint64_t streamId)
{
    uint64_t workspaceSize = std::max(size, defaultWorkspaceSize_);
    WorkspaceCacheKey cacheKey = std::make_pair(deviceId, streamId);
    auto workspaceIt = g_workspaceCache.find(cacheKey);
    if (workspaceIt != g_workspaceCache.end() && workspaceIt->second.size >= workspaceSize) {
        return workspaceIt->second.tensor.data_ptr();
    }

    if (workspaceIt != g_workspaceCache.end() && aclrtSynchronizeDevice() != 0) {
        return nullptr;
    }

    WorkspaceEntry workspaceEntry;
    workspaceEntry.size = workspaceSize;
    workspaceEntry.tensor = CreateWorkspaceTensor(workspaceSize);
    g_workspaceCache[cacheKey] = workspaceEntry;
    return g_workspaceCache[cacheKey].tensor.data_ptr();
}
} // namespace atb_torch
