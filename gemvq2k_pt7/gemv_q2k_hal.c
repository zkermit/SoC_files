#include "gemv_q2k_hal.h"
#include "gemv_q2k_uapi.h"
#include "gemv_q3k_repack.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

enum { MAX_CACHE = 512, Q2K_BYTES = 84, MAX_ROWS = 8192 };
enum { MAX_Q3_CACHE = 128, MAX_Q3_PACKED_BYTES = 192 * 1024 * 1024 };
struct entry {
    const void *weights;
    size_t bytes;
    uint32_t handle; /* zero records a rejected registration */
};
static struct entry cache[MAX_CACHE];
static int fd = -2;
static int last_ok;
static int report_stats;
static uint32_t device_caps;
static uint64_t completed_matvecs, incomplete_matvecs, submitted_weight_bytes;
static uint64_t completed_q3_matvecs;
static size_t q3_packed_bytes;
static struct {
    const void *source;
    size_t blocks;
    uint8_t *packed;
} q3_cache[MAX_Q3_CACHE];
static float q3_partial[MAX_ROWS];

static void print_stats(void)
{
    fprintf(stderr, "gemv_q2k completed=%" PRIu64 " fallback=%" PRIu64
                    " dma_weight_bytes=%" PRIu64 " q3_completed=%" PRIu64
                    " q3_packed_bytes=%zu\n",
            completed_matvecs, incomplete_matvecs, submitted_weight_bytes,
            completed_q3_matvecs, q3_packed_bytes);
}

static int getfd(void)
{
    if (fd == -2) {
        const char *disabled = getenv("GEMV_Q2K_DISABLE");
        fd = disabled && *disabled && strcmp(disabled, "0") != 0
            ? -1 : open("/dev/gemv_q2k", O_RDWR | O_CLOEXEC);
        if (fd >= 0 && ioctl(fd, GEMV_Q2K_CAPABILITIES, &device_caps) != 0)
            device_caps = 0;
        const char *profile = getenv("GEMV_Q2K_OP_TIMING");
        report_stats = fd >= 0 && profile && strcmp(profile, "1") == 0 &&
                       atexit(print_stats) == 0;
    }
    return fd;
}

int gemv_q2k_u96_available(void)
{
    return getfd() >= 0;
}

static uint32_t handle_for(const void *weights, size_t bytes)
{
    size_t i;
    for (i = 0; i < MAX_CACHE && cache[i].weights; ++i) {
        if (cache[i].weights == weights && cache[i].bytes == bytes)
            return cache[i].handle;
    }
    if (i == MAX_CACHE)
        return 0;

    struct gemv_q2k_register request = {
        .address = (uint64_t)(uintptr_t)weights,
        .bytes = bytes,
    };
    if (ioctl(fd, GEMV_Q2K_REGISTER, &request) != 0) {
        if (errno != ENOMEM && errno != EFAULT && errno != ENOSPC)
            return 0;
        request.handle = 0;
    }
    cache[i] = (struct entry){weights, bytes, request.handle};
    return request.handle;
}

static int run_encoded(const void *weights, const void *activation, float *output,
                       uint32_t nb, uint32_t rows, size_t block_bytes, uint32_t start_flag)
{
    last_ok = 0;
    if (!weights || !activation || !output || !nb || nb > 64 ||
        !rows || getfd() < 0)
        return 0;

    const size_t row_bytes = (size_t)nb * block_bytes;
    if (rows > SIZE_MAX / row_bytes)
        return 0;

    const unsigned char *matrix = weights;
    for (uint32_t first = 0; first < rows;) {
        const uint32_t remaining = rows - first;
        const uint32_t count = remaining > MAX_ROWS ? MAX_ROWS : remaining;
        const uint32_t handle = handle_for(matrix + first * row_bytes, count * row_bytes);
        if (!handle)
            goto fallback;

        struct gemv_q2k_submit job = {
            .handle = handle, .start_block = start_flag, .nb = nb, .rows = count, .timeout_ms = 1000,
            .activation = (uint64_t)(uintptr_t)activation,
            .output = (uint64_t)(uintptr_t)(output + first),
        };
        if (ioctl(fd, GEMV_Q2K_SUBMIT, &job) != 0 || job.status != 2)
            goto fallback;
        if (report_stats)
            submitted_weight_bytes += count * row_bytes;
        first += count;
    }
    if (report_stats)
        ++completed_matvecs;
    last_ok = 1;
    return last_ok;

fallback:
    if (report_stats)
        ++incomplete_matvecs;
    return 0;
}

int gemv_q2k_u96_run(const void *weights, const void *activation, float *output,
                     uint32_t nb, uint32_t rows)
{
    return run_encoded(weights, activation, output, nb, rows, Q2K_BYTES, 0);
}

int gemv_q2k_u96_last_ok(void)
{
    return last_ok;
}

static const uint8_t *prepare_q3(const uint8_t *weights, size_t blocks)
{
    size_t slot = 0;
    while (slot < MAX_Q3_CACHE && q3_cache[slot].source) {
        if (q3_cache[slot].source == weights && q3_cache[slot].blocks == blocks)
            return q3_cache[slot].packed;
        ++slot;
    }
    const size_t plane_bytes = blocks * Q2K_BLOCK_BYTES;
    const size_t bytes = plane_bytes * Q3K_REPACK_STREAMS;
    if (slot == MAX_Q3_CACHE || bytes > MAX_Q3_PACKED_BYTES - q3_packed_bytes)
        return NULL;

    uint8_t *packed = malloc(bytes);
    if (!packed)
        return NULL;
    q3_cache[slot].source = weights;
    q3_cache[slot].blocks = blocks;
    for (size_t block = 0; block < blocks; ++block) {
        uint8_t *planes[Q3K_REPACK_STREAMS];
        for (unsigned plane = 0; plane < Q3K_REPACK_STREAMS; ++plane)
            planes[plane] = packed + plane * plane_bytes + block * Q2K_BLOCK_BYTES;
        if (!gemv_q3k_repack_block(weights + block * Q3K_BLOCK_BYTES, planes)) {
            free(packed);
            return NULL;
        }
    }
    q3_cache[slot].packed = packed;
    q3_packed_bytes += bytes;
    return packed;
}

static int run_q3_slab(const uint8_t *weights, const void *activation, float *output,
                       uint32_t nb, uint32_t rows)
{
    const size_t blocks = (size_t)nb * rows;
    const size_t plane_bytes = blocks * Q2K_BLOCK_BYTES;
    const uint8_t *packed = prepare_q3(weights, blocks);
    if (!packed || !gemv_q2k_u96_run(packed, activation, output, nb, rows))
        return 0;
    for (unsigned plane = 1; plane < Q3K_REPACK_STREAMS; ++plane) {
        if (!gemv_q2k_u96_run(packed + plane * plane_bytes, activation, q3_partial, nb, rows))
            return 0;
        for (uint32_t row = 0; row < rows; ++row)
            output[row] += q3_partial[row];
    }
    return 1;
}

int gemv_q3k_u96_run(const void *weights, const void *activation, float *output,
                     uint32_t nb, uint32_t rows)
{
    if (!weights || !activation || !output || !nb || nb > 64 ||
        !rows || getfd() < 0)
        return 0;

    static int override = -2;
    if (override == -2) {
        const char *setting = getenv("GEMV_Q2K_Q3_REPACK");
        override = setting ? (strcmp(setting, "1") == 0) : -1;
    }
    if (override == 0)
        return 0;

    if (device_caps & GEMV_CAP_NATIVE_Q3) {
        const int ok = run_encoded(weights, activation, output, nb, rows,
                                   Q3K_BLOCK_BYTES, GEMV_NATIVE_Q3_START);
        if (ok && report_stats)
            ++completed_q3_matvecs;
        return ok;
    }
    if (override < 0)
        return 0;

    const size_t stride = (size_t)nb * Q3K_BLOCK_BYTES;
    if (rows > SIZE_MAX / stride)
        return 0;
    const uint8_t *matrix = weights;
    for (uint32_t first = 0; first < rows;) {
        const uint32_t remaining = rows - first;
        const uint32_t slab = remaining > MAX_ROWS ? MAX_ROWS : remaining;
        if (!run_q3_slab(matrix + first * stride, activation, output + first, nb, slab))
            return 0;
        first += slab;
    }
    if (report_stats)
        ++completed_q3_matvecs;
    return 1;
}
