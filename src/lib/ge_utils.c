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

#include <string.h>

#include <stdalign.h>

#include <gmp.h>
#include <sqlite3.h>

#include "ge_utils.h"

#define FN_STR(x)           #x
#define FN_STR1(x)          FN_STR(x)

static inline
int mpz_to_scalar(
    U8 sec[SCALAR_SIZE],
    mpz_srcptr m_sec
) {
    size_t count;

    // export scalar-sized words; else, the result is less than scalar size if front words are 0
    // (e.g., if word size is 1 byte and leading bytes are 0, count is less than 32)
    mpz_export(sec, &count, GMP_BE, SCALAR_SIZE, GMP_BE, GMP_ALL_NAILS, m_sec);

    if (1 != count) {
        fprintf(stderr, "[%zu] mpz_export failed!\n", count);
        return -1;
    }

    return 0;
}

/**
 * Computes a public key from a scalar stored as a MPZ value.
 *
 * @param[out] pub_key      Output buffer for the public key.
 * @param[in] ctx           A valid secp256k1 context.
 * @param k                 Scalar.
 *
 * @return                  0 on success, or else an error code.
 */
static inline
int mpz_to_pub(
    secp256k1_pubkey *pub_key,
    const secp256k1_context *ctx,
    mpz_srcptr k
) {
    U8 privateKey[SCALAR_SIZE];

    int ret = mpz_to_scalar(privateKey, k);

    if (0 != ret) return ret;

    ret = secp256k1_ec_pubkey_create(ctx, pub_key, privateKey);

    if (1 != ret) {
        fprintf(stderr, "[%d] pubKeyCreate failed!\n", ret);
        return -2;
    }

    return 0;
}

int mpz_to_ge(
    secp256k1_ge * ge,
    const secp256k1_context * ctx,
    mpz_srcptr k
) {
    // ensure byte buffer is 64-bit aligned so it can work as a U64 *
    alignas(8) secp256k1_pubkey pub;

    int err = mpz_to_pub(&pub, ctx, k);

    if (err) return err;

    // Safe to cast since we have a 64-bit aligned pointer
    secp256k1_ge_from_storage(ge, (const secp256k1_ge_storage *) &pub);

    return 0;
}

static inline
int serialized_pub_to_ge(
    secp256k1_ge * ge,
    const secp256k1_context * ctx,
    const U8 * input,
    size_t input_len
) {
    // Ensure byte buffer is 64-bit aligned so it can work as a U64 *
    alignas(8) secp256k1_pubkey pub;

    int ret = secp256k1_ec_pubkey_parse(ctx, &pub, input, input_len);

    if (1 != ret) {
        fprintf(stderr, "Invalid public key!!\n");
        return -1;
    }

    // Safe to cast since we have a 64-bit aligned pointer
    secp256k1_ge_from_storage(ge, (const secp256k1_ge_storage *) &pub);

    return 0;
}

int hex_pub_to_ge(
    secp256k1_ge * ge,
    const secp256k1_context * ctx,
    const char * hex_pub
) {
    U8 raw_pub[65];
    size_t input_len = strlen(hex_pub);
    mpz_t tmp;

    // Validate hex string length
    if (33 * 2 != input_len && 65 * 2 != input_len) {
        fprintf(stderr, "Invalid pubKey length: %zu\n", input_len);

        return -1;
    }

    // Import from hex string
    int err = mpz_init_set_str(tmp, hex_pub, 16);
    if (err || mpz_size(tmp) == 0) {
        fprintf(stderr, "Invalid public key\n");

        err = -1;
        goto cleanRet;
    }

    // Export to bytes array
    input_len >>= 1;
    size_t count;
    mpz_export(raw_pub, &count, GMP_BE, input_len, GMP_BE, GMP_ALL_NAILS, tmp);

    if (1 != count) {
        fprintf(stderr, "[%zu] mpz_export failed!\n", count);

        err = -1;
        goto cleanRet;
    }

    // Parse serialized public key bytes
    err = serialized_pub_to_ge(ge, ctx, raw_pub, input_len);

    cleanRet:
    mpz_clear(tmp);

    return err;
}
