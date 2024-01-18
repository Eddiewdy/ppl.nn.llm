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

#include "alibi_mask_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/alibi_mask.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {


ppl::common::RetCode DynamicBatchingAlibiMaskKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(seqstarts, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(kvstarts, 1);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(attention_mask, 2);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [seqstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(seqstarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [kvstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(kvstarts);
    if (attention_mask) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [attention_mask]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(attention_mask);
    }
    PPLNN_LLM_CUDA_DEBUG_TRACE("num_heads: %d\n", param_->num_heads);
    
    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    auto seqstarts_shape = seqstarts->GetShape();
    auto kvstarts_shape = kvstarts->GetShape();
    auto attention_mask_shape = attention_mask->GetShape();
    auto output_shape = output->GetShape();

    return ppl::kernel::llm::cuda::pmx::alibi_mask(
        GetStream(),
        seqstarts_shape,
        seqstarts->GetBufferPtr(),
        kvstarts_shape,
        kvstarts->GetBufferPtr(),
        attention_mask_shape,
        attention_mask->GetBufferPtr(),
        output_shape,
        param_->num_heads,
        output->GetBufferPtr()

    );
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
