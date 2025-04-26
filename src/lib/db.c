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

#include <sqlite3.h>

#include "../common_def.h"

#define CHECK_RESULT(code, res, TAG) do {   \
    if ((code) != (res)) {                  \
        printf("[%d] [%d] " TAG ": %s\n",   \
            (res),                          \
            sqlite3_extended_errcode(g_db), \
            sqlite3_errmsg(g_db)            \
        );                                  \
        return -1;                          \
    }                                       \
} while (0)

#define CHECK_OK(res, TAG)  CHECK_RESULT(SQLITE_OK, res, TAG)

#define FINALIZE_STMT(stmt) do {    \
    if (NULL != (stmt)) {           \
        sqlite3_finalize(stmt);     \
        (stmt) = NULL;              \
    }                               \
} while (0)

#define SQL_INSERT \
"INSERT OR IGNORE INTO pts (x, k) VALUES (?, ?)"

static sqlite3 * g_db = NULL;
static sqlite3_stmt * g_stmt_insert = NULL;

int db_open(
    const char * dbName
) {
    int err = sqlite3_open_v2(
        dbName, &g_db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, // | SQLITE_OPEN_NOMUTEX,
        NULL
    );

    CHECK_OK(err, "open");

    sqlite3_exec(
        g_db,
        "CREATE TABLE IF NOT EXISTS pts (x BLOB PRIMARY KEY, k INT) WITHOUT ROWID",
        NULL, NULL, NULL
    );

    err = sqlite3_prepare_v2(
        g_db, SQL_INSERT, -1, &g_stmt_insert, NULL
    );

    CHECK_OK(err, "prepare");

    // prepare for a large write
    // sqlite3_exec(db, "PRAGMA journal_mode=WAL;", 0, 0, 0);
    // sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", 0, 0, 0);
    // sqlite3_exec(db, "PRAGMA cache_size=-2000;", 0, 0, 0);
    sqlite3_exec(g_db,"PRAGMA journal_mode=MEMORY;", 0, 0, 0);
    sqlite3_exec(g_db,"PRAGMA synchronous=OFF;", 0, 0, 0);
    sqlite3_exec(g_db,"PRAGMA temp_store=MEMORY;", 0, 0, 0);
    sqlite3_exec(g_db,"PRAGMA locking_mode=EXCLUSIVE;", 0, 0, 0);
    sqlite3_exec(g_db,"PRAGMA cache_size=-8000;", 0, 0, 0);

    sqlite3_exec(g_db, "BEGIN IMMEDIATE", NULL, NULL, NULL);

    return 0;
}

int db_close() {
    int result = 0;

    if (NULL != g_db) {
        // assume everything went smoothly - this is not handled for this demo
        sqlite3_exec(g_db, "COMMIT", NULL, NULL, NULL);

        FINALIZE_STMT(g_stmt_insert);

        result = sqlite3_close_v2(g_db);
        CHECK_OK(result, "close");

        g_db = NULL;
    }

    return result;
}

int db_insert_result(
    U64 keyOffset,
    const U64 * x,
    U8 yParity
) {
    const U8 * data = (U8 *) x;
    sqlite3_stmt * stmt = g_stmt_insert;

    int err = sqlite3_bind_blob(stmt, 1, data, 32, SQLITE_TRANSIENT);
    CHECK_OK(err, "bind point");

    err = sqlite3_bind_int64(stmt, 2, (S64) keyOffset);
    CHECK_OK(err, "bind scalar");

    err = sqlite3_step(stmt);
    CHECK_RESULT(SQLITE_DONE, err, "step");

    err = sqlite3_reset(stmt);
    CHECK_OK(err, "reset");

    sqlite3_clear_bindings(stmt);

    return 0;
}
