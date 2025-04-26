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

int pointsBuilderGenerate(
    U64 baseKey,
    U64 batchSize,
    U32 numLoopsPerThread,
    U16 numThreads,
    char * dbName
) {
    secp256k1_context * ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);

    if (NULL == ctx) {
        fprintf(stderr, "Failed to create secp256k1 context!\n");
        return -1;
    }

    mpz_t mpBaseKey;

    mpz_init_set_ui(mpBaseKey, baseKey);

    U64 numLaunches = computeNumLaunches(
        &batchSize, NUM_CONST_POINTS, numThreads, numLoopsPerThread
    );

    on_result_cb result_cb = NULL;

    if (NULL != dbName) {
        db_open(dbName);
        result_cb = db_insert_result;
    }

    clock_t st = clock();
    double ompStartTime = omp_get_wtime();

    batch_add_range(
        ctx, numLaunches, numLoopsPerThread, numThreads, mpBaseKey,
        result_cb
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

    return 0;
}
