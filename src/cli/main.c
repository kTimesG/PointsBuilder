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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../inc/points_builder.h"
#include "../common_def.h"

/**
 * Entry point.
 *
 * @param argc      Number of program arguments.
 * @param argv      Array of program arguments.
 *
 * @return          Program exit code.
 */
int main(int argc, char **argv) {
    const char * baseKey = NULL;
    U64 rangeSize = 0;
    U32 numLoopsPerLaunch = 1;
    U16 numThreads = 1;
    U32 progressMinInterval = 3;
    char *dbName = NULL;

    int arg_idx = 1; // skip arg 0 = binary path

    // parse arguments
    while (arg_idx < argc) {
        char * arg = argv[arg_idx];

        if (strlen(arg) != 2 || '-' != arg[0]) {
            fprintf(stderr, "Unknown argument %s\n", arg);
            return EXIT_FAILURE;
        }

        switch (argv[arg_idx][1]) {
            case 'b':
                baseKey = argc > ++arg_idx ? argv[arg_idx] : NULL;
                break;
            case 's':
                rangeSize = argc > ++arg_idx ? strtoull(argv[arg_idx], NULL, 10) : 0;
                break;
            case 't':
                numThreads = argc > ++arg_idx ? strtoul(argv[arg_idx], NULL, 10) : 0;
                break;
            case 'p':
                progressMinInterval = argc > ++arg_idx ? strtoul(argv[arg_idx], NULL, 10) : 0;
                break;
            case 'n':
                numLoopsPerLaunch = argc > ++arg_idx ? strtoul(argv[arg_idx], NULL, 10) : 0;
                break;
            case 'o':
                dbName = argc > ++arg_idx ? argv[arg_idx] : NULL;
                break;
            default:
                fprintf(stderr, "Unknown argument %s\n", argv[arg_idx]);
                return EXIT_FAILURE;
        }

        ++arg_idx;
    }

    if (NULL == baseKey) {
        fprintf(stderr, "Base key is required.\n");
        return EXIT_FAILURE;
    }

    if (0 == rangeSize) {
        fprintf(stderr, "Range size cannot be 0\n");
        return EXIT_FAILURE;
    }

    if (0 == numThreads) {
        fprintf(stderr, "Num threads cannot be 0\n");
        return EXIT_FAILURE;
    }

    if (0 == numLoopsPerLaunch) {
        fprintf(stderr, "Loop count cannot be 0\n");
        return EXIT_FAILURE;
    }

    if (NULL == dbName) {
        printf("No DB name given - compute only mode\n");
    }

    int err = pointsBuilderGenerate(
        baseKey, rangeSize,
        numLoopsPerLaunch, numThreads, progressMinInterval,
        dbName
    );

    if (err) {
        fprintf(
            stderr,
            "[%d] Failed\n",
            err
        );

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
