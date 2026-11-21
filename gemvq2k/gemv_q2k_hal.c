#include "gemv_q2k_hal.h"
#include "gemv_q2k_uapi.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

enum { BQ8=34, BQ2K=84, TILE=4096 };
static int fd = -2;
static int dev(void) { if (fd == -2) fd = open("/dev/gemv_q2k", O_RDWR | O_CLOEXEC); return fd; }

static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000U;
    int exp = (int)((v.u >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = v.u & 0x007FFFFFU;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000U;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = (mant >> shift) & 0x03FFU;
        uint32_t rem = mant & ((1U << shift) - 1U);
        uint32_t mid = 1U << (shift - 1U);
        if (rem > mid || (rem == mid && (half & 1U))) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00U | (mant ? 0x0200U : 0U));
    }
    uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFU;
    if (rem > 0x1000U || (rem == 0x1000U && (half & 1U))) half++;
    return (uint16_t)(sign | half);
}

static void q8_0(const float *x, unsigned char *q, uint32_t n) {
    for (uint32_t b = 0; b < n / 32; b++) {
        float a = 0.0f;
        for (int i = 0; i < 32; i++) {
            float z = fabsf(x[b * 32 + i]);
            if (z > a) a = z;
        }
        float d = a / 127.0f;
        uint16_t h = fp32_to_fp16(d);
        q[b * BQ8]     = (unsigned char)(h & 0xFFU);
        q[b * BQ8 + 1] = (unsigned char)(h >> 8);
        float inv_d = d ? (1.0f / d) : 0.0f;
        for (int i = 0; i < 32; i++) {
            int z = d ? (int)lrintf(x[b * 32 + i] * inv_d) : 0;
            if (z > 127) z = 127;
            if (z < -128) z = -128;
            q[b * BQ8 + 2 + i] = (unsigned char)(signed char)z;
        }
    }
}

bool gemv_q2k_try(const void *w, const float *x, float *dst, uint32_t nb, uint32_t rows) {
    if (!w || !x || !dst || !nb || nb > 1024 || dev() < 0) return false;
    unsigned char y[1024 * 8 * BQ8];
    q8_0(x, y, nb * 256);
    uint32_t per_engine = (4U * 1024U * 1024U) / (nb * BQ2K);
    if (!per_engine) return false;
    for (uint32_t r = 0; r < rows;) {
        uint32_t n = rows - r, limit = per_engine * 2;
        if (n > TILE) n = TILE;
        if (n > limit) n = limit;
        uint32_t n0 = (n + 1) / 2, n1 = n - n0;
        struct gemv_q2k_job j = {
            .weights    = (uint64_t)(uintptr_t)((const unsigned char *)w + (size_t)r * nb * BQ2K),
            .activation = (uint64_t)(uintptr_t)y,
            .output     = (uint64_t)(uintptr_t)(dst + r),
            .nb         = nb,
            .rows0      = n0,
            .rows1      = n1,
            .timeout_ms = 1000
        };
        if (ioctl(fd, GEMV_Q2K_SUBMIT, &j)) return false;
        r += n;
    }
    return true;
}
