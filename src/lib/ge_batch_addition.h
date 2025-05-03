/* Copyright 2025 kTimesG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef POINTS_BUILDER_GE_BATCH_ADD_H
#define POINTS_BUILDER_GE_BATCH_ADD_H

#include <gmp.h>
#include <secp256k1.h>

#include "../common_def.h"

#define NUM_CONST_POINTS    512

typedef void (* on_result_cb)(
    U64 key,
    const U64 * x,
    U8 yParity
);

int batch_add_range(
    const secp256k1_context * ctx,
    U64 numLaunches,
    U64 numLoopsPerLaunch,
    U16 numThreads,
    mpz_srcptr baseKey,
    on_result_cb callback,
    U32 progressMinInterval
);

#endif // POINTS_BUILDER_GE_BATCH_ADD_H
