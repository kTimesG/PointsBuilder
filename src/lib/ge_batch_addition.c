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
#include <string.h>

#include <gmp.h>
#include <sqlite3.h>

// Disable "unused function" warnings in 3rd-party includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include "field_impl.h"                 // field operations
#include "group_impl.h"                 // field operations
#include "int128_native_impl.h"         // use native __int128

#pragma GCC diagnostic pop

#include "ge_batch_addition.h"

#include <omp.h>

#include "ge_utils.h"
#include "../common_def.h"

#define GE_CONST_ON_HEAP    (NUM_CONST_POINTS > 16384)

#define FE_INV(r, x)        secp256k1_fe_impl_inv_var(&(r), &(x))
#define FE_MUL(r, a, b)     secp256k1_fe_mul_inner((r).n, (a).n, (b).n)
#define FE_SQR(r, x)        secp256k1_fe_sqr_inner((r).n, (x).n)
#define FE_ADD(r, d)        secp256k1_fe_impl_add(&(r), &(d))
#define FE_NEG(r, a, m)     secp256k1_fe_impl_negate_unchecked(&(r), &(a), (m))

static
int compute_const_points(
    const secp256k1_context * ctx,
    secp256k1_ge * out,
    const U32 numPoints
) {
    secp256k1_gej gej;

    // The first point is the delta to the next center position
    mpz_t k;
    mpz_init_set_ui(k, 2 * numPoints - 1);
    int err = mpz_to_ge(out++, ctx, k);
    mpz_clear(k);

    if (err) return err;

    if (numPoints < 2) return 0;

    // all other points are 1, 2, ... N - 1
    *out = secp256k1_ge_const_g;
    secp256k1_gej_set_ge(&gej, out++);

    for (U32 i = 1; i < numPoints - 1; i++) {
        secp256k1_gej_add_ge_var(&gej, &gej, &secp256k1_ge_const_g, NULL);
        secp256k1_ge_set_gej(out++, &gej);
    }

    return 0;
}

static
void batch_addition(
    secp256k1_fe_storage * xOut,
    uint8_t * yParityOut,
    secp256k1_ge * ge,                  // a single point
    const secp256k1_ge * jp,
    secp256k1_fe * xz,                  // product tree leafs + parent nodes
    secp256k1_fe * xzOut,
    const U32 batch_size
) {
    secp256k1_fe t1, t2, t3;

    S64 i;

    for (i = 0; i < batch_size; i++) {
        xz[i] = ge[0].x;
        FE_NEG(t1, jp[i].x, 1);         // T1 = -x2
        FE_ADD(xz[i], t1);              // XZ[i] = x1 - x2
    }

    // up-sweep inversion tree [SIMD friendly]
    for (i = 0; i < batch_size - 1; i++) {
        FE_MUL(xz[batch_size + i], xz[i * 2], xz[i * 2 + 1]);
    }

    FE_INV(xzOut[batch_size * 2 - 2], xz[2 * batch_size - 2]);

    // down-sweep inversion tree
    for (i = (S64) batch_size - 2; i >= 0; i--) {
        FE_MUL(xzOut[i * 2], xz[i * 2 + 1], xzOut[batch_size + i]);
        FE_MUL(xzOut[i * 2 + 1], xz[i * 2], xzOut[batch_size + i]);
    }

    secp256k1_ge tmp;
    secp256k1_ge * _a = &tmp;
    const secp256k1_fe * _inv = xzOut + batch_size - 1;

    // Output the starting middle point as a result
    secp256k1_fe_to_storage(xOut, &ge->x);
    *yParityOut = ge->y.n[0] & 1;

    for (i = batch_size - 1;; i--) {
        const secp256k1_ge * _b = &jp[i];

        // 1. do P + Q
        tmp = ge[0];

        FE_NEG(t1, _b->y, 1);                       // T1 = -y2
        FE_ADD(_a->y, t1);                          // Y1 = y1 - y2                     m = max_y + 2(1)
        FE_MUL(_a->y, _a->y, *_inv);                // Y1 = m = (y1 - y2) / (x1 - x2)   m = 1
        FE_SQR(t2, _a->y);                          // T2 = m**2                        m = 1
        FE_NEG(t3, _b->x, 1);                       // T3 = -x2
        FE_ADD(t2, t3);                             // T2 = m**2 - x2                   m = 1 + 2(1) = 3(2)
        FE_NEG(_a->x, _a->x, 1);                    // X1 = -x1                         m = max_x + 1
        FE_ADD(_a->x, t2);                          // X1 = x3 = m**2 - x1 - x2         max_x = 3 + max_x + 1
        secp256k1_fe_normalize_var(&_a->x);

        FE_NEG(t2, _a->x, 1);                       // T2 = -x3                         m = 1 + 1 = 2
        FE_ADD(t2, _b->x);                          // T1 = x2 - x3                     m = 2 + 1 = 3
        FE_MUL(_a->y, _a->y, t2);                   // Y1 = m * (x2 - x3)               m = 1
        FE_ADD(_a->y, t1);                          // Y1 = y3 = m * (x2 - x3) - y2     m = 1 + 2 = 3
        secp256k1_fe_normalize_var(&_a->y);

        if (0 == i) {
            *ge = *_a;                              // go to the next interval
            break;
        }

        // delta != 0. Output X as result
        secp256k1_fe_to_storage(xOut + i, &_a->x);
        yParityOut[i] = _a->y.n[0] & 1;

        // 2. Do P - Q using the same inverse
        tmp = ge[0];

        FE_ADD(_a->y, _b->y);                       // Y1 = y1 + y2                     m = max_y + 2(1)
        FE_MUL(_a->y, _a->y, *_inv);                // Y1 = m = (y1 + y2) / (x1 - x2)   m = 1
        FE_SQR(t2, _a->y);                          // T2 = m**2                        m = 1
        FE_NEG(t3, _b->x, 1);                       // T3 = -x2
        FE_ADD(t2, t3);                             // T2 = m**2 - x2                   m = 1 + 2(1) = 3(2)
        FE_NEG(_a->x, _a->x, 1);                    // X1 = -x1                         m = max_x + 1
        FE_ADD(_a->x, t2);                          // X1 = x3 = m**2 - x1 - x2         max_x = 3 + max_x + 1
        secp256k1_fe_normalize_var(&_a->x);

        FE_NEG(t2, _a->x, 1);                       // T2 = -x3                         m = 1 + 1 = 2
        FE_ADD(t2, _b->x);                          // T1 = x2 - x3                     m = 2 + 1 = 3
        FE_MUL(_a->y, _a->y, t2);                   // Y1 = m * (x2 - x3)               m = 1
        FE_ADD(_a->y, _b->y);                       // Y1 = y3 = m * (x2 - x3) + y2     m = 1 + 2 = 3
        secp256k1_fe_normalize_var(&_a->y);

        secp256k1_fe_to_storage(xOut - i, &_a->x);
        yParityOut[-i] = _a->y.n[0] & 1;

        --_inv;
    }
}

/**
 * Computes the first pivot GE, when the base key is provided as a scalar.
 *
 * @param[out] r        GE output.
 * @param ctx           Valid secp256k1 context.
 * @param baseKey       Private key.
 *
 * @return
 */
static
int compute_first_pivot_sec(
    secp256k1_ge * r,
    const secp256k1_context * ctx,
    mpz_srcptr baseKey
) {
    mpz_t k;

    mpz_init_set(k, baseKey);
    mpz_add_ui(k, k, NUM_CONST_POINTS - 1);

    int err = mpz_to_ge(r, ctx, k);

    mpz_clear(k);

    return err;
}

/**
 * Computes the initial pivots for all threads, except the first one.
 * The first pivot is assumed to be valid, and is used as an initial offset.
 *
 * @param[in,out] gej_pivots    GE pivots buffer, including the first thread's pivot.
 * @param[in] ctx               Valid secp256k1 context.
 * @param numLoops              Total number of loops done by each thread.
 * @param numThreads            Total number of threads. This should be > 1.
 *
 * @return
 */
static
int compute_pivots_gej(
    secp256k1_gej * gej_pivots,
    const secp256k1_context * ctx,
    const U64 numLoops,                 // >= 1
    U16 numThreads
) {
    // Scalar delta between the initial pivots of two threads.
    mpz_t kDelta;
    // GE for delta * G
    secp256k1_ge geDelta;

    // after each loop, the pivot of each thread moves by 2 * numConst - 1
    // const U64 delta = numLoops * (2 * NUM_CONST_POINTS - 1);
    // Use big int to prevent 64-bit overflow
    mpz_init_set_ui(kDelta, numLoops);
    mpz_mul_ui(kDelta, kDelta, 2 * NUM_CONST_POINTS - 1);

    int err = mpz_to_ge(&geDelta, ctx, kDelta);
    mpz_clear(kDelta);

    if (err) return err;

    while (--numThreads) {
        secp256k1_gej_add_ge_var(gej_pivots + 1, gej_pivots, &geDelta, NULL);
        ++gej_pivots;
    }

    return 0;
}

static
int compute_pivots(
    secp256k1_ge * ge_pivots,
    const secp256k1_context * ctx,
    const secp256k1_ge * base_ge,
    const secp256k1_ge * ge_const,
    mpz_srcptr baseKey,
    const U64 numLoops,                 // >= 1
    const U16 numThreads
) {
    int err = 0;

    secp256k1_gej gej_stack;
    secp256k1_gej * gej_pivots;

    // Computing all the pivots when the number of threads is a large value
    // (thousands or more) can be optimized heavily, and even parallelized.
    // For our use-case (at most a few hundred threads), the below strategy
    // is good enough.
    // Serially add Jacobian points one after another, and do a final conversion
    // to affine points, using a single field inversion.

    if (numThreads > 1) {
        gej_pivots = malloc(numThreads * sizeof(secp256k1_gej));

        if (NULL == gej_pivots) return -1;
    }
    else {
        gej_pivots = &gej_stack;
    }

    if (NULL != base_ge) {
        // Jacobian Pivot = BasePub + [NUM_CONST - 1] * G
        secp256k1_gej_set_ge(gej_pivots, base_ge);
        secp256k1_gej_add_ge_var(gej_pivots, gej_pivots, ge_const + NUM_CONST_POINTS - 1, NULL);

        if (1 == numThreads) {
            // single pivot; convert it to affine
            secp256k1_ge_set_gej_var(ge_pivots, gej_pivots);
        }
    }
    else {
        // Affine Pivot = [baseKey + NUM_CONST - 1] * G
        err = compute_first_pivot_sec(ge_pivots, ctx, baseKey);
    }

    if (numThreads > 1) {
        if (!err) {
            const int offset = NULL != baseKey ? 1 : 0;

            if (offset) {
                // The first pivot is in affine form already; init Jacobian
                secp256k1_gej_set_ge(gej_pivots, ge_pivots);
            }

            err = compute_pivots_gej(gej_pivots, ctx, numLoops, numThreads);

            if (!err) {
                // convert all J points to affine, using a single field inversion
                secp256k1_ge_set_all_gej_var(ge_pivots + offset, gej_pivots + offset, numThreads - offset);
            }
        }

        free(gej_pivots);
    }

    return err;
}

static
int compute_results(
    secp256k1_fe_storage * xOut,
    U8 * yParityOut,
    const U64 numLoopsPerLaunch,
    const U64 numLaunches,
    const U16 numThreads,
    secp256k1_ge * ge_pivot,
    const secp256k1_ge * ge_const,
    secp256k1_fe * p_trees,
    const U32 resultsSize,
    const U32 treeSize,
    const U32 progressMinInterval,
    const secp256k1_context * ctx,
    const secp256k1_ge * base_ge,
    mpz_srcptr baseKey,
    const on_result_cb callback
) {
    const U64 numResPerLaunch = resultsSize * numLoopsPerLaunch;
    const U64 pivotStride = numLaunches * numResPerLaunch;

    U64 launchKeyOffset = 0;

    double fTotalResPerLaunch = (double) numResPerLaunch * numThreads;
    double fNumTotal = (double) numLaunches * fTotalResPerLaunch;
    printf("Computing ~ %.0f points...\n", fNumTotal);

    double ompStartTime = omp_get_wtime();
    double ompProgressTime = ompStartTime;

    for (U64 launchIdx = 0; launchIdx < numLaunches; launchIdx++) {
        // Generate results step (allows parallelization)
#pragma omp parallel for \
num_threads(numThreads) \
default(none) \
shared(numThreads, numResPerLaunch, numLoopsPerLaunch, xOut, resultsSize, yParityOut, ge_pivot, ge_const, p_trees, treeSize)
        for (U16 tId = 0; tId < numThreads; tId++) {
            const size_t resOffset = tId * numResPerLaunch
                + NUM_CONST_POINTS - 1;

            secp256k1_fe_storage * x = xOut + resOffset;
            U8 * yParity = yParityOut + resOffset;
            secp256k1_ge * pivot = ge_pivot + tId;
            secp256k1_fe * treeProd = p_trees + tId * treeSize * 2;
            secp256k1_fe * treeInv = treeProd + treeSize;

            for (U64 loopIdx = 0; loopIdx < numLoopsPerLaunch; loopIdx++) {
                batch_addition(
                    x, yParity,
                    pivot, ge_const,
                    treeProd, treeInv,
                    NUM_CONST_POINTS
                );

                x += resultsSize;
                yParity += resultsSize;
            }
        }

        if (progressMinInterval) {
            double now = omp_get_wtime();

            if (now - ompProgressTime >= progressMinInterval) {
                double elapsedTime = now - ompStartTime;
                double fNumDone = (double) (launchIdx + 1) * fTotalResPerLaunch;
                double speed = fNumDone / elapsedTime;
                ompProgressTime = now;

                printf(
                    "\r[%.1f%%] [%.0f s] BatchAdd speed: %.0f keys/s [%.1f Mk/ts]",
                    fNumDone * 100 / fNumTotal, elapsedTime, speed, speed / (numThreads * 1000000)
                );
                fflush(stdout);
            }
        }

        // Handle results (serialized)
#if 0
        mpz_t tmp;

        mpz_init(tmp);

        for (U16 tId = 0; tId < numThreads; tId++) {
            secp256k1_fe_storage * x = xOut
                + tId * numResPerLaunch;
            U64 keyOffset = launchKeyOffset
                + tId * pivotStride;
            secp256k1_fe_storage xCheck;
            secp256k1_ge ge_check;

            for (U64 loopIdx = 0; loopIdx < numLoopsPerLaunch; loopIdx++) {
                for (U32 i = 0; i < resultsSize; i++) {
                    mpz_set_ui(tmp, keyOffset);

                    if (NULL != baseKey) {
                        mpz_add(tmp, tmp, baseKey);
                    }

                    int err;

                    if (NULL == base_ge || keyOffset) {
                        // compute either [baseKey + offset]G or [offset]G
                        err = mpz_to_ge(&ge_check, ctx, tmp);
                        if (err) return -1;
                    }

                    if (NULL != base_ge) {
                        if (keyOffset) {
                            // add offset*G to base pubKey
                            secp256k1_gej gej_res;

                            secp256k1_gej_set_ge(&gej_res, base_ge);
                            secp256k1_gej_add_ge_var(&gej_res, &gej_res, &ge_check, NULL);
                            secp256k1_ge_set_gej(&ge_check, &gej_res);

                            // X has magnitude 1, but might not be normalized
                            // Needed for correct 256-bit serialization
                            secp256k1_fe_normalize_var(&ge_check.x);
                        }
                        else {
                            // offset 0; basePubKey should simply match itself
                            ge_check = *base_ge;
                        }
                    }

                    secp256k1_fe_to_storage(&xCheck, &ge_check.x);

                    if (0 != memcmp(x, &xCheck, 32)) {
                        fprintf(stderr,
                            "Check failed k %" PRIu64
                            " launch %" PRIu64 " loop %" PRIu64
                            " tId %" PRIu16 " idx %" PRIu32 "\n"
                            "output: \t%016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n"
                            "expect: \t%016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
                            keyOffset, launchIdx, loopIdx, tId, i,
                            x->n[3], x->n[2], x->n[1], x->n[0],
                            xCheck.n[3], xCheck.n[2], xCheck.n[1], xCheck.n[0]
                        );

                        return -1;
                    }

                    ++keyOffset;
                    ++x;
                }
            }
        }

        mpz_clear(tmp);
#endif

        if (NULL == callback) goto end;

        for (U16 tId = 0; tId < numThreads; tId++) {
            const secp256k1_fe_storage * x = xOut
                + tId * numResPerLaunch;
            const U8 * yParity = yParityOut
                + tId * numResPerLaunch;
            U64 keyOffset = launchKeyOffset
                + tId * pivotStride;

            for (U64 loopIdx = 0; loopIdx < numLoopsPerLaunch; loopIdx++) {
                for (U32 i = 0; i < resultsSize; i++) {
                    callback(keyOffset, x->n, *yParity);

                    ++keyOffset;
                    ++x;
                    ++yParity;
                }
            }
        }

        end:
        launchKeyOffset += numResPerLaunch;
    }

    return 0;
}

int batch_add_range(
    const secp256k1_context * ctx,
    U64 numLaunches,
    U64 numLoopsPerLaunch,
    U16 numThreads,
    mpz_srcptr baseKey,
    const secp256k1_ge * base_ge,
    on_result_cb callback,
    U32 progressMinInterval
) {
    size_t szTotalMem = 0;
    size_t szMem;

    // A large constant array may fail stack allocation at runtime
#if GE_CONST_ON_HEAP
    szMem = NUM_CONST_POINTS * sizeof(secp256k1_ge);

    secp256k1_ge * ge_const = malloc(szMem);
#else
    secp256k1_ge ge_const[NUM_CONST_POINTS];
#endif

    szMem = numThreads * sizeof(secp256k1_ge);
    szTotalMem += szMem;

    secp256k1_ge * ge_pivot = malloc(szMem);

    // Number of elements required for an inversion (or products) tree
    // for a single loop of [pivot + {const}] results list
    const U32 treeSize = NUM_CONST_POINTS * 2 - 1;

    // Memory required for a single tree
    const size_t szMemTree = treeSize * sizeof(secp256k1_fe);

    // Each pivot uses 2 trees
    // Each thread (1 pivot) requires its own working memory area
    szMem = numThreads * szMemTree * 2;
    szTotalMem += szMem;

    secp256k1_fe * p_trees = malloc(szMem);

    // Number of output elements, per pivot, per loop
    const U32 resultsSize = NUM_CONST_POINTS * 2 - 1;

    // memory to hold results, for a single launch
    szMem = numThreads * numLoopsPerLaunch
        * resultsSize
        * sizeof(secp256k1_fe_storage);
    szTotalMem += szMem;

    secp256k1_fe_storage * xOut = malloc(szMem);

    szMem = numThreads * numLoopsPerLaunch * resultsSize * sizeof(U8);
    szTotalMem += szMem;

    U8 * yParityOut = malloc(szMem);

    int err = 0;

    if (
        NULL != xOut
        && NULL != yParityOut
        && NULL != ge_pivot
#if GE_CONST_ON_HEAP
        && NULL != ge_const
#endif
        && NULL != p_trees
    ) {
        szMem = NUM_CONST_POINTS * sizeof(secp256k1_ge);
        printf(
            "Batch add: using %zu KB [T: %" PRIu16 " L: %" PRIu64 " x %" PRIu64 "]"
            " Memory/thread: %zu kB\n",
            (szTotalMem + szMem) >> 10,
            numThreads, numLaunches, numLoopsPerLaunch,
            (szTotalMem / numThreads) >> 10
        );

        err = compute_const_points(ctx, ge_const, NUM_CONST_POINTS);
        if (!err) {
            err = compute_pivots(
                ge_pivot, ctx,
                base_ge, ge_const,
                baseKey,
                numLaunches * numLoopsPerLaunch, numThreads
            );
        }

        if (!err) {
            err = compute_results(
                xOut, yParityOut,
                numLoopsPerLaunch, numLaunches, numThreads,
                ge_pivot, ge_const, p_trees,
                resultsSize, treeSize, progressMinInterval,
                ctx, base_ge, baseKey, callback
            );

            printf("\n");
        }
    }
    else {
        if (NULL == ge_pivot) {
            fprintf(stderr, "Pivot alloc failed\n");
        }

#if GE_CONST_ON_HEAP
        if (NULL == ge_const) {
            fprintf(stderr, "Const alloc failed\n");
        }
#endif

        if (NULL == p_trees) {
            fprintf(stderr, "Trees alloc failed\n");
        }

        if (NULL == xOut) {
            fprintf(stderr, "Results alloc failed\n");
        }

        if (NULL == yParityOut) {
            fprintf(stderr, "Y parity alloc failed\n");
        }

        err = -1;
    }

    if (NULL != yParityOut) free(yParityOut);
    if (NULL != xOut) free(xOut);
    if (NULL != p_trees) free(p_trees);
    if (NULL != ge_pivot) free(ge_pivot);
#if GE_CONST_ON_HEAP
    if (NULL != ge_const) free(ge_const);
#endif

    return err;
}
