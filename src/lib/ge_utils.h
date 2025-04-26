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

#ifndef POINTS_BUILDER_GE_UTILS_H
#define POINTS_BUILDER_GE_UTILS_H

#include <gmp.h>
#include <secp256k1.h>

// Ignore "unused function" warnings in 3rd-party includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include "field_impl.h"             // field operations
#include "group_impl.h"             // group operations
#include "int128_native_impl.h"     // use native __int128

#pragma GCC diagnostic pop

#include "../common_def.h"

#define GMP_LE          -1          // little-endian
#define GMP_HE          0           // host endianness
#define GMP_BE          1           // big-endian
#define GMP_ALL_NAILS   0
#define SCALAR_SIZE     32          // Size of a private key, in bytes

int mpz_to_ge(
    secp256k1_ge * ge,
    const secp256k1_context * ctx,
    mpz_srcptr k
);

int buildPts(
    const secp256k1_context * ctx,
    U64 numPoints,
    U64 baseKey,
    U16 stride,
    const char * dbName
);

#endif // POINTS_BUILDER_GE_UTILS_H
