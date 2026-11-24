// User-space HAL for the gemv_q2k K-quant accelerator (ABI v3).
// Linked into ggml-cpu; called from ggml_compute_forward_mul_mat on thread 0.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE          /* O_CLOEXEC under strict -std=c11 */
#endif
#include "gemv_q2k_hal.h"

#if defined(__linux__)
#include "gemv_q2k_uapi.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define FP_BYTES       32u      /* fingerprint: first + last 32 bytes          */
#define CACHE_SLOTS    8192u    /* power of two; > weight tensors in a model   */
#define Q8K_BLOCK      292u     /* sizeof(block_q8_K): float d, qs[256], bsums */

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
static bool             g_ready;     /* fixed after hal_init                  */
static atomic_int       g_enabled;   /* cleared on a hardware error           */
static int              g_fd = -1;
static uint32_t         g_max_blocks, g_type_mask, g_min_rows = 16;
static uint64_t         g_max_rows, g_max_pin;
static struct wentry    g_cache[CACHE_SLOTS];
static unsigned char    g_act[GEMV_Q2K_MAX_BLOCKS * GEMV_Q2K_ACT_BLOCK_BYTES] __attribute__((aligned(64)));

static uint32_t block_bytes(int t) {
    switch (t) {
    case GEMV_Q2K_TYPE_Q2_K: return 84;
    case GEMV_Q2K_TYPE_Q3_K: return 110;
    case GEMV_Q2K_TYPE_Q4_K: return 144;
    case GEMV_Q2K_TYPE_Q5_K: return 176;
    case GEMV_Q2K_TYPE_Q6_K: return 210;
    }
    return 0;
}

static uint32_t parse_types(const char *s) {   /* "q2_k,q6_k" -> mask */
    uint32_t mask = 0;
    char buf[8];
    while (*s) {
        size_t n = 0;
        while (*s && *s != ',' && n < sizeof buf - 1) buf[n++] = (char) tolower((unsigned char) *s++);
        buf[n] = 0;
        while (*s && *s != ',') s++;
        if (*s == ',') s++;
        if (n == 4 && buf[0] == 'q' && buf[2] == '_' && buf[3] == 'k' && buf[1] >= '2' && buf[1] <= '6')
            mask |= 1u << (GEMV_Q2K_TYPE_Q2_K + (buf[1] - '2'));
    }
    return mask;
}

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
        fprintf(stderr, "gemv_q2k: driver ABI mismatch (rebuild/reload gemv_q2k.ko), offload disabled\n");
        close(fd);
        return;
    }
    g_max_blocks = info.max_blocks < GEMV_Q2K_MAX_BLOCKS ? info.max_blocks : GEMV_Q2K_MAX_BLOCKS;
    g_max_rows   = info.max_rows;
    g_max_pin    = info.max_pin_bytes;
    g_type_mask  = info.type_mask;
    e = getenv("GEMV_Q2K_TYPES");
    if (e && *e) g_type_mask &= parse_types(e);
    g_fd = fd;
    atomic_store(&g_enabled, 1);
    g_ready = true;
    fprintf(stderr, "gemv_q2k: FPGA offload enabled (%u engines, types mask 0x%x, nb<=%u, rows>=%u)\n",
            info.engines, g_type_mask, g_max_blocks, g_min_rows);
}

bool gemv_q2k_eligible(int type, uint32_t nb, uint64_t rows, bool weights_buffer) {
    if (!weights_buffer || nb == 0 || rows == 0 || type < 0 || type > 31) return false;
    pthread_once(&g_once, hal_init);
    if (!g_ready) return false;          /* constant state only, see header */
    const uint32_t B = block_bytes(type);
    return B && (g_type_mask & (1u << type)) && nb <= g_max_blocks &&
           rows >= g_min_rows && rows <= g_max_rows && rows * nb * B <= g_max_pin;
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

/* Device handle for these weights; pins them on first use (no copy). */
static bool ensure_registered(const void *w, uint64_t bytes, uint32_t *handle) {
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
            if (e->state != SLOT_READY) return false;
            *handle = e->handle;
            return true;
        }
        /* Same address, different contents (model reloaded): the old pins
           refer to the old pages, so drop them and pin the new ones. */
        if (e->state == SLOT_READY) ioctl(g_fd, GEMV_Q2K_FREE, &e->handle);
    }
    e->ptr = w;
    e->bytes = bytes;
    memcpy(e->fp, fp, sizeof fp);
    struct gemv_q2k_register r = { .addr = (uint64_t) (uintptr_t) w, .bytes = bytes };
    if (((uintptr_t) w & 15) != 0 || ioctl(g_fd, GEMV_Q2K_REGISTER, &r) != 0) {
        static int warned;
        if (!warned++) fprintf(stderr, "gemv_q2k: could not pin a weight tensor (%s); it stays on the CPU\n",
                               ((uintptr_t) w & 15) ? "unaligned" : strerror(errno));
        e->state = SLOT_REJECTED;
        return false;
    }
    e->handle = r.handle;
    e->state = SLOT_READY;
    *handle = r.handle;
    return true;
}

/* block_q8_K {float d; int8 qs[256]; int16 bsums[16]} -> 272-byte HW block */
static void pack_activation(const unsigned char *q8k, unsigned char *out, uint32_t nb) {
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *src = q8k + (size_t) Q8K_BLOCK * b;
        unsigned char *dst = out + (size_t) GEMV_Q2K_ACT_BLOCK_BYTES * b;
        memcpy(dst, src, 4);
        memset(dst + 4, 0, 12);
        memcpy(dst + 16, src + 4, 256);
    }
}

bool gemv_q2k_try(int type, const void *w, const void *q8k, float *dst, uint32_t nb, uint32_t rows) {
    if (!w || !q8k || !dst || !gemv_q2k_eligible(type, nb, rows, true)) return false;
    if (!atomic_load_explicit(&g_enabled, memory_order_acquire)) return false;
    const uint64_t bytes = (uint64_t) rows * nb * block_bytes(type);
    bool ok = false;
    uint32_t handle;

    pthread_mutex_lock(&g_lock);
    if (ensure_registered(w, bytes, &handle)) {
        pack_activation((const unsigned char *) q8k, g_act, nb);
        struct gemv_q2k_job j = {
            .activation = (uint64_t) (uintptr_t) g_act,
            .output     = (uint64_t) (uintptr_t) dst,
            .handle = handle, .type = (uint32_t) type, .nb = nb, .rows = rows,
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

bool gemv_q2k_eligible(int type, uint32_t nb, uint64_t rows, bool weights_buffer) {
    (void) type; (void) nb; (void) rows; (void) weights_buffer;
    return false;
}
bool gemv_q2k_try(int type, const void *w, const void *q8k, float *dst, uint32_t nb, uint32_t rows) {
    (void) type; (void) w; (void) q8k; (void) dst; (void) nb; (void) rows;
    return false;
}

#endif
