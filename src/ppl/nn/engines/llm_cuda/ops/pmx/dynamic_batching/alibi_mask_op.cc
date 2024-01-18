// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "alibi_mask_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/dynamic_batching/alibi_mask_kernel.h"
#include "ppl/nn/oputils/pmx/reshape_dynamic_batching_alibi_mask.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode DynamicBatchingAlibiMaskOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(DATATYPE_FLOAT16);
        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto seqstarts = info->GetInput<TensorImpl>(0);
        auto kvstarts = info->GetInput<TensorImpl>(1);
        int64_t kv_length = 0;
        int64_t seq_length = 0;

        auto kv_length_addr = kvstarts->GetBufferPtr<int64_t>() + kvstarts->GetShape()->GetDim(0) - 1;
        auto kv_length_desc = ppl::nn::BufferDesc(kv_length_addr);
        auto status = kvstarts->GetDevice()->CopyToHost(&kv_length, kv_length_desc, sizeof(kv_length));

        if (status != RC_SUCCESS) {
            LOG(ERROR) << "kvstarts->GetDevice()->CopyToHost() failed: " << GetRetCodeStr(status);
            return status;
        }

        auto seq_length_addr = seqstarts->GetBufferPtr<int64_t>() + seqstarts->GetShape()->GetDim(0) - 1;
        auto seq_length_desc = ppl::nn::BufferDesc(seq_length_addr);
        status = seqstarts->GetDevice()->CopyToHost(&seq_length, seq_length_desc, sizeof(seq_length));
        
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "seqstarts->GetDevice()->CopyToHost() failed: " << GetRetCodeStr(status);
            return status;
        }

        return ppl::nn::pmx::ReshapeDynamicBatchingAlibiMask(info, param_->num_heads, seq_length, kv_length);

    };
    return RC_SUCCESS;
}

RetCode DynamicBatchingAlibiMaskOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::AlibiMaskParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    return CommonInit();
}

KernelImpl* DynamicBatchingAlibiMaskOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<DynamicBatchingAlibiMaskKernel>(param_.get());
}

}}}}} // namespace ppl::nn::llm::cuda::pmx