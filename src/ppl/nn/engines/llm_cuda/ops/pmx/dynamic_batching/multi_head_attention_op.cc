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

#include "multi_head_attention_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/dynamic_batching/multi_head_attention_kernel.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode DynamicBatchingMultiHeadAttentionOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode DynamicBatchingMultiHeadAttentionOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::MultiHeadAttentionParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    LOG(ERROR) << "currently do not support this op";
    return ppl::common::RC_UNSUPPORTED;

    return CommonInit();
}

KernelImpl* DynamicBatchingMultiHeadAttentionOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<DynamicBatchingMultiHeadAttentionKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode DynamicBatchingMultiHeadAttentionOp::SerializeData(const ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::CreateMultiHeadAttentionParam(builder, 
        param_.get()->num_heads,
        param_.get()->num_kv_heads,
        param_.get()->head_dim,
        param_.get()->is_causal);
    auto fb_op_param = pmx::CreateOpParam(builder, pmx::OpParamType_MultiHeadAttentionParam, fb_param.Union());
    pmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode DynamicBatchingMultiHeadAttentionOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = pmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_MultiHeadAttentionParam();
    
    param_ = make_shared<ppl::nn::pmx::MultiHeadAttentionParam>();
    param_.get()->num_heads    = fb_param->num_heads();
    param_.get()->num_kv_heads = fb_param->num_kv_heads();
    param_.get()->head_dim     = fb_param->head_dim();
    param_.get()->is_causal    = fb_param->is_causal();
    
    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::pmx
