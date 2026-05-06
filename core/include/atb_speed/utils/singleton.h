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
#ifndef ATB_SPEED_UTILS_SINGLETON_H
#define ATB_SPEED_UTILS_SINGLETON_H

#include <acl/acl.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace atb_speed {

template <class T> T &GetThreadLocalSingleton()
{
    thread_local static T instance;
    return instance;
}

template <class T> T &GetSingleton()
{
    static T instance;
    return instance;
}

template <class T> T &GetSingletonPerDevice(int32_t dev_id)
{
    static std::once_flag once_flag;
    static std::unordered_map<int32_t, std::unique_ptr<T>> instances;
    static uint32_t device_count = 0;

    std::call_once(once_flag, []() {
        aclError ret = aclrtGetDeviceCount(&device_count);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtGetDeviceCount failed");
        }
        for (uint32_t i = 0; i < device_count; ++i) {
            instances.emplace(static_cast<int32_t>(i), std::make_unique<T>());
        }
    });

    if (dev_id < 0 || static_cast<uint32_t>(dev_id) >= device_count) {
        throw std::out_of_range("device id out of range");
    }

    auto it = instances.find(dev_id);
    if (it == instances.end()) {
        throw std::runtime_error("device singleton is not initialized");
    }
    return *(it->second);
}
} // namespace atb_speed
#endif
