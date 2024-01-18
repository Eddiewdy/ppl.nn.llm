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

#include "ppl/nn/oputils/pmx/reshape_dynamic_batching_alibi_mask.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

ppl::common::RetCode ReshapeDynamicBatchingAlibiMask(InputOutputInfo* info, int64_t num_heads, int64_t seq_length, int64_t kv_length) {
    
    const int64_t dim = 3;
    std::vector<int64_t> out_dims(dim);

    out_dims[0] = num_heads;
    out_dims[1] = seq_length;
    out_dims[2] = (kv_length + 15) / 16 * 16;

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(out_dims);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
