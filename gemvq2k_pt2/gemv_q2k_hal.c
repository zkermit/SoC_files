// User-space HAL for the gemv_q2k accelerator (ABI v2).
// Linked into ggml-cpu; called from ggml_compute_forward_mul_mat on thread 0.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE          /* O_CLOEXEC under strict -std=c11 */
#endif
#include "gemv_q2k_hal.h"

#if defined(__linux__)
#include "gemv_q2k_uapi.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define FP_BYTES    32u          /* fingerprint: first + last 32 bytes       */
#define CACHE_SLOTS 8192u        /* power of two; > Q2_K tensors in a model  */

enum { SLOT_EMPTY = 0, SLOT_READY = 1, SLOT_REJECTED = -1 };

struct wentry {
    const void   *ptr;
    uint64_t      bytes;
    uint32_t      handle;
    int           state;
    unsigned char fp[2 * FP_BYTES];
};

static pthread_once_t   g_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t  g_lock = PTHREAD_MUTEX_INITIALIZER;
static bool             g_ready;    /* set once in hal_init, never changes   */
static atomic_int       g_enabled;  /* cleared on a hardware error           */
static int              g_fd = -1;
static uint32_t         g_max_blocks;
static uint64_t         g_max_rows, g_max_upload;
static uint32_t         g_min_rows = 64;
static struct wentry    g_cache[CACHE_SLOTS];
static unsigned char    g_act[GEMV_Q2K_MAX_BLOCKS * GEMV_Q2K_ACT_BLOCK_BYTES] __attribute__((aligned(64)));

static void hal_init(void) {
    const char *e = getenv("GEMV_Q2K_DISABLE");
    if (e && *e && strcmp(e, "0") != 0) return;
    e = getenv("GEMV_Q2K_MIN_ROWS");
    if (e && *e) g_min_rows = (uint32_t) strtoul(e, NULL, 10);

    int fd = open(GEMV_Q2K_DEVICE, O_RDWR | O_CLOEXEC);
    if (fd < 0) return;                                   /* no FPGA: silent CPU path */
    struct gemv_q2k_info info;
    memset(&info, 0, sizeof info);
    if (ioctl(fd, GEMV_Q2K_INFO, &info) != 0 || info.version != GEMV_Q2K_ABI_VERSION ||
        info.act_block_bytes != GEMV_Q2K_ACT_BLOCK_BYTES || info.max_blocks == 0) {
        fprintf(stderr, "gemv_q2k: driver ABI mismatch, FPGA offload disabled\n");
        close(fd);
        return;
    }
    g_max_blocks = info.max_blocks < GEMV_Q2K_MAX_BLOCKS ? info.max_blocks : GEMV_Q2K_MAX_BLOCKS;
    g_max_rows   = info.max_rows;
    g_max_upload = info.max_upload_bytes;
    g_fd = fd;
    atomic_store(&g_enabled, 1);
    g_ready = true;
    fprintf(stderr, "gemv_q2k: FPGA offload enabled (%u engines, nb<=%u, rows>=%u)\n",
            info.engines, g_max_blocks, g_min_rows);
}

bool gemv_q2k_eligible(uint32_t nb, uint64_t rows, bool weights_buffer) {
    if (!weights_buffer || nb == 0 || rows == 0) return false;
    pthread_once(&g_once, hal_init);
    /* Only constant state here: all worker threads of one node must agree even
       if thread 0 disables the device concurrently (that is checked in try). */
    if (!g_ready) return false;
    return nb <= g_max_blocks && rows <= g_max_rows && rows >= g_min_rows &&
           rows * nb * GEMV_Q2K_BLOCK_BYTES <= g_max_upload;
}

/* ggml nearest_int + quantize_row_q8_K_ref, emitted in the 272-byte HW layout. */
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof i);
    return (i & 0x007fffff) - 0x00400000;
}

static void quantize_q8k_hw(const float *x, unsigned char *out, uint32_t nb) {
    for (uint32_t b = 0; b < nb; b++) {
        const float *xb = x + 256u * b;
        unsigned char *o = out + (size_t) GEMV_Q2K_ACT_BLOCK_BYTES * b;
        float max = 0, amax = 0;
        for (int j = 0; j < 256; j++) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }
        }
        memset(o, 0, 16);
        if (amax == 0) { memset(o + 16, 0, 256); continue; }
        const float iscale = -127.f / max;
        for (int j = 0; j < 256; j++) {
            const int v = nearest_int(iscale * xb[j]);
            o[16 + j] = (unsigned char) (signed char) (v > 127 ? 127 : v);
        }
        const float d = 1 / iscale;
        memcpy(o, &d, sizeof d);                          /* little-endian host */
    }
}

static size_t slot_of(const void *p) {
    uint64_t x = (uint64_t) (uintptr_t) p;
    x ^= x >> 29;
    x *= 0x9E3779B97F4A7C15ull;
    return (size_t) (x >> 40) & (CACHE_SLOTS - 1);
}

static void fingerprint(const unsigned char *w, uint64_t bytes, unsigned char *fp) {
    memcpy(fp, w, FP_BYTES);
    memcpy(fp + FP_BYTES, w + bytes - FP_BYTES, FP_BYTES);
}

/* Resolve a device handle for these weights, uploading on first use. */
static bool ensure_uploaded(const void *w, uint64_t bytes, uint32_t *handle) {
    struct wentry *e = NULL;
    const size_t s = slot_of(w);
    for (size_t i = 0; i < CACHE_SLOTS; i++) {
        struct wentry *c = &g_cache[(s + i) & (CACHE_SLOTS - 1)];
        if (c->state == SLOT_EMPTY || c->ptr == w) { e = c; break; }
    }
    if (!e) return false;

    unsigned char fp[2 * FP_BYTES];
    fingerprint((const unsigned char *) w, bytes, fp);
    if (e->state != SLOT_EMPTY) {
        if (e->bytes == bytes && memcmp(e->fp, fp, sizeof fp) == 0) {
            if (e->state != SLOT_READY) return false;     /* known not to fit */
            *handle = e->handle;
            return true;
        }
        /* Same address, different contents (model reloaded): drop the copy. */
        if (e->state == SLOT_READY) ioctl(g_fd, GEMV_Q2K_FREE, &e->handle);
    }

    e->ptr = w;
    e->bytes = bytes;
    memcpy(e->fp, fp, sizeof fp);
    struct gemv_q2k_upload up = { .addr = (uint64_t) (uintptr_t) w, .bytes = bytes };
    if (ioctl(g_fd, GEMV_Q2K_UPLOAD, &up) != 0) {
        if (errno == ENOMEM) {
            static int warned;
            if (!warned++) fprintf(stderr, "gemv_q2k: CMA full, remaining tensors stay on CPU "
                                           "(Ultra96-V2: cma=640M or cma=768M in bootargs)\n");
        }
        e->state = SLOT_REJECTED;
        return false;
    }
    e->handle = up.handle;
    e->state = SLOT_READY;
    *handle = up.handle;
    return true;
}

bool gemv_q2k_try(const void *w, const float *x, float *dst, uint32_t nb, uint32_t rows) {
    if (!w || !x || !dst || !gemv_q2k_eligible(nb, rows, true)) return false;
    if (!atomic_load_explicit(&g_enabled, memory_order_acquire)) return false;
    const uint64_t bytes = (uint64_t) rows * nb * GEMV_Q2K_BLOCK_BYTES;
    bool ok = false;
    uint32_t handle;

    pthread_mutex_lock(&g_lock);
    if (ensure_uploaded(w, bytes, &handle)) {
        quantize_q8k_hw(x, g_act, nb);
        struct gemv_q2k_job j = {
            .activation = (uint64_t) (uintptr_t) g_act,
            .output     = (uint64_t) (uintptr_t) dst,
            .handle = handle, .nb = nb, .rows = rows, .timeout_ms = 0,
        };
        if (ioctl(g_fd, GEMV_Q2K_SUBMIT, &j) == 0) {
            ok = true;
        } else if (errno == EIO || errno == ETIMEDOUT || errno == EBUSY) {
            fprintf(stderr, "gemv_q2k: hardware error (%s, status=0x%x), FPGA offload disabled\n",
                    strerror(errno), j.status);
            atomic_store(&g_enabled, 0);
        }
    }
    pthread_mutex_unlock(&g_lock);
    return ok;
}

#else  /* !__linux__ */

bool gemv_q2k_eligible(uint32_t nb, uint64_t rows, bool weights_buffer) {
    (void) nb; (void) rows; (void) weights_buffer;
    return false;
}
bool gemv_q2k_try(const void *w, const float *x, float *dst, uint32_t nb, uint32_t rows) {
    (void) w; (void) x; (void) dst; (void) nb; (void) rows;
    return false;
}

#endif
