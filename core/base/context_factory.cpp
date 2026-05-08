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
#include "atb_speed/base/context_factory.h"
#include <thread>
#include <unordered_map>
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/config.h"

namespace atb_speed {
namespace {
using ContextMap = std::unordered_map<void *, std::shared_ptr<atb::Context>>;

thread_local ContextMap g_localContexts;
}  // namespace

const int MAX_STREAM_NUM = 2;

bool ContextFactory::cacheWorkspace_ = false;

std::vector<aclrtStream> ContextFactory::GetSubStreams()
{
    static std::vector<aclrtStream> streams;
    static bool initialized = false;
    if (!initialized) {
        aclInit(nullptr);

        for (int i = 0; i < MAX_STREAM_NUM; ++i) {
            aclrtStream subStream;
            
            aclError ret = aclrtCreateStream(&subStream);
            if (ret != ACL_ERROR_NONE) {
                ATB_SPEED_LOG_ERROR("Failed to create aclrtStream: " << ret);
            }
            ret = aclrtSetStreamFailureMode(subStream, ACL_STOP_ON_FAILURE);
            if (ret != 0) {
                ATB_SPEED_LOG_ERROR("Failed to aclrtSetStreamFailureMode: " << ret);
            }
            streams.push_back(subStream);
        }
        initialized = true;
    }

    return streams;
}

std::shared_ptr<atb::Context> ContextFactory::GetAtbContext(void *stream)
{
    auto context_it = g_localContexts.find(stream);
    if (context_it != g_localContexts.end()) {
        ATB_SPEED_LOG_DEBUG("ContextFactory return localContext");
        return context_it->second;
    }

    ATB_SPEED_LOG_DEBUG("ContextFactory create atb::Context start");
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR("ContextFactory create atb::Context fail");
    }

    if (context) {
        context->SetExecuteStream(stream);
        if (atb_speed::GetSingleton<atb_speed::Config>().IsUseTilingCopyStream()) {
            ATB_SPEED_LOG_DEBUG("ContextFactory use tiling copy stream");
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_SPEED_LOG_DEBUG("ContextFactory not use tiling copy stream");
        }
    }

    std::shared_ptr<atb::Context> local_context(
        context, [](atb::Context *context) { atb::DestroyContext(context); });
    g_localContexts[stream] = local_context;

    return local_context;
}

void ContextFactory::FreeAtbContext()
{
    ATB_SPEED_LOG_DEBUG("ContextFactory FreeAtbContext start.");
    if (g_localContexts.empty()) {
        return;
    }

    for (auto context_it = g_localContexts.begin(); context_it != g_localContexts.end();) {
        ATB_SPEED_LOG_DEBUG("ContextFactory localContext use_count: " << context_it->second.use_count());
        if (context_it->second.use_count() == 1) {
            context_it = g_localContexts.erase(context_it);
        } else {
            ++context_it;
        }
    }
}
} 
