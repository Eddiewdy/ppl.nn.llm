#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_ALIBI_MASK_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_ALIBI_MASK_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct AlibiMaskParam final : public ir::TypedAttr<AlibiMaskParam> {
    int32_t num_heads;

    bool operator==(const AlibiMaskParam& p) const {
        return num_heads == p.num_heads;
    }
};

}}} // namespace ppl::nn::pmx

#endif
