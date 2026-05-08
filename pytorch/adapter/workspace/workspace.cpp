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
#include "workspace.h"

#include <acl/acl.h>

#include <stdexcept>

#include "atb_speed/log.h"
#include "buffer_device.h"

namespace atb_speed {
namespace {
constexpr uint64_t kDefaultWorkspaceSize = 629145600;

using WorkspaceCacheKey = std::pair<int32_t, uint64_t>;

WorkspaceCacheKey GetWorkspaceCacheKey(int32_t device_id, uint64_t buffer_key)
{
    return std::make_pair(device_id, buffer_key);
}

int32_t GetCurrentDeviceId()
{
    int32_t device_id = 0;
    aclError ret = aclrtGetDevice(&device_id);
    if (ret != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("Workspace::GetWorkspaceBuffer get current device failed, error:" << ret);
        throw std::runtime_error("aclrtGetDevice fail before get workspace buffer, please check plog.");
    }
    return device_id;
}
} // namespace

Workspace::Workspace() {}

Workspace::~Workspace() {}

void Workspace::ClearCache()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& buff : std::as_const(workspaceBuffer_)) {
        buff.second->ClearBuffer();
    }
}

int32_t Workspace::GetCachedNum()
{
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t rt = 0;
    for (auto& buff : std::as_const(workspaceBuffer_)) {
        rt = rt + buff.second->GetCachedNum();
    }
    return rt;
}

void *Workspace::GetWorkspaceBuffer(uint64_t bufferSize, uint64_t bufferKey)
{
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t device_id = GetCurrentDeviceId();
    WorkspaceCacheKey cache_key = GetWorkspaceCacheKey(device_id, bufferKey);
    if (workspaceBuffer_.count(cache_key) == 0) {
        uint64_t initial_buffer_size = bufferSize > kDefaultWorkspaceSize ? bufferSize : kDefaultWorkspaceSize;
        workspaceBuffer_[cache_key].reset(new BufferDevice(initial_buffer_size));
    }
    return workspaceBuffer_[cache_key]->GetBuffer(bufferSize);
}

} // namespace atb_speed
