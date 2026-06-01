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
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "atb_speed/utils/statistic.h"
#include "operations/aclnn/utils/utils.h"
#include "atb_speed/utils/singleton.h"
#include "executor_manager.h"
#include "acl_nn_global_cache.h"
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

namespace {
constexpr const char *ACLNN_THREAD_LOCAL_CACHE_ENV = "XLLM_ATB_ACLNN_THREAD_LOCAL_CACHE";

bool IsThreadLocalAclNNCacheEnabled()
{
    static const bool enabled = []() {
        const char *env = std::getenv(ACLNN_THREAD_LOCAL_CACHE_ENV);
        return env != nullptr && std::string(env) == "1";
    }();
    return enabled;
}

atb::Status GetCurrentDeviceId(int32_t &device_id)
{
    aclError ret = aclrtGetDevice(&device_id);
    if (ret != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("Get current device id failed, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }
    return atb::NO_ERROR;
}

std::shared_ptr<ExecutorManager> GetThreadLocalAclNNExecutorManager(int32_t device_id)
{
    thread_local std::unordered_map<int32_t, std::shared_ptr<ExecutorManager>> managers;
    auto it = managers.find(device_id);
    if (it != managers.end()) {
        return it->second;
    }

    auto manager = std::make_shared<ExecutorManager>();
    managers.emplace(device_id, manager);
    return manager;
}

ExecutorManager &GetAclNNExecutorManager(int32_t device_id)
{
    if (IsThreadLocalAclNNCacheEnabled()) {
        return *GetThreadLocalAclNNExecutorManager(device_id);
    }
    return GetSingletonPerDevice<ExecutorManager>(device_id);
}

ExecutorManager &GetAclNNExecutorManager(const std::shared_ptr<AclNNOpCache> &cache, int32_t device_id)
{
    if (cache != nullptr && cache->executorManager != nullptr) {
        return *cache->executorManager;
    }
    return GetAclNNExecutorManager(device_id);
}

void BindThreadLocalAclNNOpCacheOwner(const std::shared_ptr<AclNNOpCache> &cache, int32_t device_id)
{
    if (!IsThreadLocalAclNNCacheEnabled() || cache == nullptr) {
        return;
    }

    cache->executorManager = GetThreadLocalAclNNExecutorManager(device_id);
    cache->ownerThreadId = std::this_thread::get_id();
    cache->ownerThreadRecorded = true;
}

bool IsReusableLocalAclNNOpCache(const std::shared_ptr<AclNNOpCache> &cache, int32_t device_id)
{
    if (cache == nullptr || cache->device_id != device_id) {
        return false;
    }
    if (!IsThreadLocalAclNNCacheEnabled()) {
        return true;
    }
    return cache->ownerThreadRecorded && cache->ownerThreadId == std::this_thread::get_id();
}
} // namespace

AclNNOperation::AclNNOperation(const std::string &opName) : opName_(opName)
{
    this->aclnnOpCache_ = std::make_shared<AclNNOpCache>();
}

AclNNOperation::~AclNNOperation()
{
    ATB_SPEED_LOG_DEBUG("AclNNOperation deconstructor");
    this->DestroyOperation();
}

std::string AclNNOperation::GetName() const { return this->opName_; }

void AclNNOperation::DestroyOperation() const
{
    if (this->operationDestroyed_ || this->aclnnOpCache_ == nullptr) {
        return;
    }
    this->operationDestroyed_ = true;
    this->aclnnOpCache_->Destroy();
}

atb::Status AclNNOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context)
{
    int32_t device_id = 0;
    atb::Status status = GetCurrentDeviceId(device_id);
    if (status != atb::NO_ERROR) {
        return status;
    }
    return Setup(variantPack, workspaceSize, context, device_id);
}

atb::Status AclNNOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context,
                                  int32_t device_id)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " setup start");

    // 1. translatedContexttranslated
    if (context == nullptr) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " setup context is null");
        return atb::ERROR_INVALID_PARAM;
    }

    // 2. translatedExecutortranslatedWorkspace
    int ret = UpdateAclNNOpCache(variantPack, device_id);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call UpdateAclNNOpCache, error:" << ret);
        this->aclnnOpCache_->Destroy();
        return ret;
    }

    // 3. translatedworkspaceSize
    workspaceSize = this->aclnnOpCache_->workspaceSize;

    ATB_SPEED_LOG_DEBUG(GetSingletonPerDevice<AclNNGlobalCache>(device_id).PrintGlobalCache());
    ATB_SPEED_LOG_DEBUG(GetAclNNExecutorManager(this->aclnnOpCache_, device_id).PrintExecutorCount());
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::UpdateAclNNOpCache(const atb::VariantPack &variantPack, int32_t device_id)
{
    // translatedExecutetranslatedExecutortranslatedworkspace
    // translated:GlobalCachetranslatedexecutortranslatedLocalCachetranslated;translatedLocalCachetranslated

    // 1. translatedLocal CachetranslatedExecutortranslated
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Local Cache call IsVariankPackEqual");
    if (IsReusableLocalAclNNOpCache(this->aclnnOpCache_, device_id) && this->aclnnOpCache_->executorRepeatable && \
        IsVariankPackEqual(this->aclnnOpCache_->aclnnVariantPack, variantPack)) {
        // Local Cachetranslated
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" <<
            (this) << "] Cache addr[" << this->aclnnOpCache_.get() << "] Executor addr[" <<
            this->aclnnOpCache_->aclExecutor << "] Local Cache Hit");
        return atb::NO_ERROR;
    }

    if (!IsThreadLocalAclNNCacheEnabled()) {
        // 2. translatedGlobal CachetranslatedExecutortranslated
        std::shared_ptr<AclNNOpCache> globalCache = \
            GetSingletonPerDevice<AclNNGlobalCache>(device_id).GetGlobalCache(this->opName_, variantPack);
        if (globalCache != nullptr) {
            // Global Cachetranslated
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" << (this)
                << "] Cache addr[" << globalCache.get() << "] Executor addr[" << globalCache->aclExecutor
                << "] Global Cache Hit");
            // 2.1 translatedLocal Cache
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: destroy local cache before switching to global cache");
            this->aclnnOpCache_->Destroy();
            // 2.2 translatedLocal Cache
            this->aclnnOpCache_ = globalCache;
            // 2.3 translatedExecutorManager
            int count = GetAclNNExecutorManager(this->aclnnOpCache_, device_id)
                            .IncreaseReference(this->aclnnOpCache_->aclExecutor);
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Executor addr[" <<
                this->aclnnOpCache_->aclExecutor << "] count update to " << count);
            return atb::NO_ERROR;
        }
    }

    // 3. Local CachetranslatedGlobal Cachetranslated
    // 3.1 translatedLocal Cache
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: destroy local cache before create a new one");
    this->aclnnOpCache_->Destroy();
    // 3.2 translatedvariantPack,translatedaclnnOpCache_,translatedWorkSpacetranslatedExecutor
    this->aclnnOpCache_ = std::make_shared<AclNNOpCache>();
    this->aclnnOpCache_->device_id = device_id;
    BindThreadLocalAclNNOpCacheOwner(this->aclnnOpCache_, device_id);
    int ret = CreateAclNNOpCache(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call CreateAclNNOpCache fail, error:" << ret);
        return ret;
    }
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" <<
        (this) << "] Cache addr[" << this->aclnnOpCache_.get() << "] Executor addr[" <<
        this->aclnnOpCache_->aclExecutor << "] create Local Cache");
    // 3.3 translatedExecutorManager,translatedExecutor,counttranslated1
    int count = GetAclNNExecutorManager(this->aclnnOpCache_, device_id)
                    .IncreaseReference(this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] increase Executor addr[" <<
        this->aclnnOpCache_->aclExecutor << "] count update to " << count);

    if (!IsThreadLocalAclNNCacheEnabled()) {
        // 3.4 translatedGlobal Cache(translatedGlobal Cachetranslated)
        GetSingletonPerDevice<AclNNGlobalCache>(device_id).UpdateGlobalCache(this->opName_, this->aclnnOpCache_);
    }

    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNOpCache(const atb::VariantPack &variantPack)
{
    atb::Status ret = CreateAclNNVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call CreateAclNNVariantPack fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    ret = SetAclNNWorkspaceExecutor();
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call SetAclNNWorkspaceExecutor fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    // translatedLocal Cachetranslated
    if (this->aclnnOpCache_ == nullptr) {
        ATB_SPEED_LOG_ERROR("Plugin Op Cache: Op name[" << this->opName_ << "] cache is nullptr after " <<
            "initialization, please check.");
        return atb::ERROR_INTERNAL_ERROR;
    }

    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: create Executor addr[" << this->aclnnOpCache_->aclExecutor << "]");

    // translatedLocal CachetranslatedaclExecutortranslated,translated0,translated
    ret = aclSetAclOpExecutorRepeatable(this->aclnnOpCache_->aclExecutor);
    if (ret != 0) {
        // translated,translatedLocal Cachetranslatedexecutortranslated
        ATB_SPEED_LOG_WARN(this->opName_ << " call aclSetAclOpExecutorRepeatable fail: " << ret);
        this->aclnnOpCache_->executorRepeatable = false;
    } else {
        // translated,translatedLocal Cachetranslatedexecutortranslated
        this->aclnnOpCache_->executorRepeatable = true;
    }
    
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                    atb::Context *context)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " execute start");
    if (!context) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, context param is null. Enable log: "
            << "export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtStream stream = GetExecuteStream(context);
    if (!stream) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, execute stream in context is null. "
            << "Enable log: export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    // translated
    int ret = this->aclnnOpCache_->UpdateAclNNVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call UpdateAclNNVariantPack fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    ATB_SPEED_LOG_DEBUG("Input workspaceSize " << workspaceSize << " localCache workspaceSize " <<
        this->aclnnOpCache_->workspaceSize);
    ret = ExecuteAclNNOp(workspace, stream);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call ExecuteAclNNOp fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    ATB_SPEED_LOG_DEBUG(this->opName_ << " execute end");

    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " CreateAclNNVariantPack start");
    atb::Status ret = 0;
    ATB_SPEED_LOG_DEBUG(this->opName_ << " CreateAclNNInVariantPack start");
    ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail");
        return ret;
    }

    ATB_SPEED_LOG_DEBUG(this->opName_ << " CreateAclNNOutVariantPack start");
    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
        return ret;
    }
    
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), i);
        if (aclnnVariantPack.aclInTensors[i]->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
        if (aclnnVariantPack.aclOutTensors[i]->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed
