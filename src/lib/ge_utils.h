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

/**
 *
 * Computes the public key of a secp256k1 scalar key.
 *
 * @param[out]  ge      Group element to write into.
 * @param[in]   ctx     A valid secp256k1 context.
 * @param[in]   k       Scalar value (private key).
 *
 * @return Zero on success, non-zero if an error occurred.
 */
int mpz_to_ge(
    secp256k1_ge * ge,
    const secp256k1_context * ctx,
    mpz_srcptr k
);

/**
 * Parses a hex serialized public key into a group element.
 *
 * @param[out]  ge      Group element to write into.
 * @param[in]   ctx     A valid secp256k1 context.
 * @param[in]   hex_pub Serialized public key, as a hex string.
 *
 * @return Zero on success, non-zero if an error occurred.
 */
int hex_pub_to_ge(
    secp256k1_ge * ge,
    const secp256k1_context * ctx,
    const char * hex_pub
);

#endif // POINTS_BUILDER_GE_UTILS_H
