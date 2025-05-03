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

#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#include <omp.h>

#include <gmp.h>

#include "../../inc/points_builder.h"
#include "ge_utils.h"
#include "ge_batch_addition.h"
#include "db.h"

/**
 * Compute the required number of launches for a given number
 * of threads to fully traverse a given range size.
 *
 * @param[in,out] rangeSize     Size of the range to be covered.
 *                              This value may get adjusted.
 * @return The number of launches required to cover the entire range.
 */
static
U64 computeNumLaunches(
    U64 * rangeSize,
    const U32 numConstPoints,
    const U32 numThreads,
    const U32 numLoopsPerThread
) {
    const U64 initialRangeSize = *rangeSize;

    // Number of GE results for all threads, for a launch
    const U64 numGE3PerLaunch = numThreads
        * numLoopsPerThread
        * (numConstPoints * 2 - 1);

    // numLaunches = ceil(rangeSize / keysPerThread)

#if 1
    // (division, remainder) is usually a single CPU instruction
    U64 numLaunches = *rangeSize / numGE3PerLaunch;
    U64 numLeftOvers = *rangeSize % numGE3PerLaunch;

    if (numLeftOvers) {
        // an extra launch is required to cover the remaining area
        ++numLaunches;
        // adjust range size with the remaining distance from the end
        *rangeSize += numGE3PerLaunch - numLeftOvers;
    }
#elif 1
    U64 numLaunches = (*rangeSize + numGE3PerLaunch - 1) / numGE3PerLaunch;
    *rangeSize = numGE3PerLaunch * numLaunches;
#endif

    printf(" Points/launch: %" PRIu64 "\n", numGE3PerLaunch);
    printf("Range overhead: %" PRIu64 "\n", *rangeSize - initialRangeSize);
    printf("Required range: %" PRIu64 "\n", initialRangeSize);
    printf("Adjusted range: %" PRIu64 "\n", *rangeSize);

    return numLaunches;
}

static
int validateRange(
    mpz_srcptr mpBaseKey,
    U64 rangeSize,
    U64 numTotalLoops
) {
    // secp256k1 curve order
    const U64 secp256k1_n[4] = {
        0xffffffffffffffff,
        0xfffffffffffffffe,
        0xbaaedce6af48a03b,
        0xbfd25e8cd0364141
    };

    int err = 0;
    mpz_t mpN, mpTmp;

    mpz_init2(mpN, SCALAR_SIZE * 8);            // 256 bits

    // Array elements are in BE order; limb content is in host-endianness
    mpz_import(
        mpN,
        sizeof(secp256k1_n) / sizeof(*secp256k1_n),
        GMP_BE, sizeof(*secp256k1_n), GMP_HE, GMP_ALL_NAILS, secp256k1_n
    );

    mpz_init_set(mpTmp, mpBaseKey);
    mpz_add_ui(mpTmp, mpTmp, rangeSize - 1);    // last scalar

    // gmp_printf("       N: %064Zx\n", mpN);
    gmp_printf("Last Key: %064Zx\n", mpTmp);

    // all range scalars must be below N; this also catches a base key above N - 1
    if (mpz_cmp(mpTmp, mpN) >= 0) {
        fprintf(stderr, "Range exceeds N - 1.\n");
        err = -1;
    }
    else {
        // Pivots = {base + C - 1 + loopIdx * (2C - 1)}
        // If P == Q or P == -Q, the addition will be incorrect.
        // Only two possible cases exist:
        // First pivot: base + C - 1 == 2C - 1 -> base == C
        // Last pivot: base + C - 1 == -2C + 1 -> base == -3C + 2 mod N
        // All other X collision cases imply that the point at infinity is crossed,
        // but this condition was already checked.

        if (mpz_cmp_ui(mpBaseKey, NUM_CONST_POINTS) == 0) {
            // Pivot == Const[0]
            err = -2;
        }

        // Check if the last pivot matches - Const[0]
        // Equivalent to arriving at the point at infinity after all loops
        mpz_set_ui(mpTmp, numTotalLoops);
        mpz_mul_ui(mpTmp, mpTmp, NUM_CONST_POINTS * 2 - 1);
        mpz_add_ui(mpTmp, mpTmp, NUM_CONST_POINTS - 1);
        mpz_add(mpTmp, mpTmp, mpBaseKey);

        if (mpz_cmp(mpTmp, mpN) == 0) {
            err = -3;
        }

        if (err) {
            // Cannot add points that have the same X value.
            fprintf(stderr, "Cannot compute this range - decrease base key by 1.\n");
        }
    }

    mpz_clear(mpTmp);
    mpz_clear(mpN);

    return err;
}

int pointsBuilderGenerate(
    const char * baseKey,
    U64 batchSize,
    U32 numLoopsPerThread,
    U16 numThreads,
    U32 progressMinInterval,
    const char * dbName
) {
    secp256k1_context * ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);

    if (NULL == ctx) {
        fprintf(stderr, "Failed to create secp256k1 context!\n");
        return -1;
    }

    mpz_t mpBaseKey;

    int err = mpz_init_set_str(mpBaseKey, baseKey, 16);
    if (err || mpz_size(mpBaseKey) == 0) {
        fprintf(stderr, "Invalid base key\n");
        return -1;
    }

    gmp_printf("Base Key: %064Zx\n", mpBaseKey);

    U64 numLaunches = computeNumLaunches(
        &batchSize, NUM_CONST_POINTS, numThreads, numLoopsPerThread
    );

    err = validateRange(mpBaseKey, batchSize, numLaunches * numLoopsPerThread);
    if (err) return err;

    on_result_cb result_cb = NULL;

    if (NULL != dbName) {
        db_open(dbName);
        result_cb = db_insert_result;
    }

    clock_t st = clock();
    double ompStartTime = omp_get_wtime();

    err = batch_add_range(
        ctx, numLaunches, numLoopsPerThread, numThreads, mpBaseKey,
        result_cb, progressMinInterval
    );

    double ompEndTime = omp_get_wtime();
    clock_t et = clock();

    db_close();

    double speedNum = (double) batchSize;
    double speed = speedNum / (ompEndTime - ompStartTime);
    printf("Overall gen & store speed: %.3f keys/s\n", speed);
    printf("Total clock time: %.3f\n", (double) (et - st) / CLOCKS_PER_SEC);
    printf("Total wall time: %.3f\n", ompEndTime - ompStartTime);

    secp256k1_context_destroy(ctx);

    mpz_clear(mpBaseKey);

    return err;
}
