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
#include "atb_context_factory.h"
#include <unordered_map>
#include "atb_speed/log.h"
#include "config.h"

namespace atb_torch {
namespace {
thread_local std::unordered_map<void *, std::shared_ptr<atb::Context>> g_localContexts;
}  // namespace

AtbContextFactory &AtbContextFactory::Instance()
{
    static AtbContextFactory instance;
    return instance;
}

std::shared_ptr<atb::Context> AtbContextFactory::GetAtbContext(void *stream)
{
    auto contextIt = g_localContexts.find(stream);
    if (contextIt != g_localContexts.end()) {
        ATB_SPEED_LOG_DEBUG("AtbContextFactory return localContext");
        return contextIt->second;
    }

    ATB_SPEED_LOG_DEBUG("AtbContextFactory create atb::Context start");
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR("AtbContextFactory create atb::Context fail");
    }
    if (context) {
        context->SetExecuteStream(stream);
        if (Config::Instance().IsUseTilingCopyStream()) {
            ATB_SPEED_LOG_DEBUG("AtbContextFactory use tiling copy stream");
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_SPEED_LOG_DEBUG("AtbContextFactory not use tiling copy stream");
        }
    }
    std::shared_ptr<atb::Context> localContext(
        context, [](atb::Context *context) { atb::DestroyContext(context); });
    g_localContexts[stream] = localContext;

    return localContext;
}

void AtbContextFactory::FreeAtbContext()
{
    ATB_SPEED_LOG_DEBUG("AtbContextFactory FreeAtbContext start");
    if (g_localContexts.empty()) {
        return;
    }

    for (auto contextIt = g_localContexts.begin(); contextIt != g_localContexts.end();) {
        ATB_SPEED_LOG_DEBUG("AtbContextFactory localContext use_count: " << contextIt->second.use_count());
        if (contextIt->second.use_count() == 1) {
            contextIt = g_localContexts.erase(contextIt);
        } else {
            ++contextIt;
        }
    }
}
} // namespace atb_torch
