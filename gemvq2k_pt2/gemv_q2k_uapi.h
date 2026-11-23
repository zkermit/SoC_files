/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * gemv_q2k user/kernel ABI (version 2).
 *
 * Weights are uploaded ONCE into driver-owned DMA memory (UPLOAD) and then
 * referenced by handle. Per token only the quantised activation (nb * 272 B)
 * goes in and rows * 4 B come out, so the FPGA streams weights straight from
 * DDR instead of the CPU memcpy-ing every matrix on every token.
 *
 * Activation block format (272 B per 256 values, "Q8_K-HW"):
 *   bytes  0..3   float d       (identical to ggml block_q8_K.d)
 *   bytes  4..15  zero
 *   bytes 16..271 int8 qs[256]  (identical to ggml block_q8_K.qs)
 * bsums are recomputed in hardware. Weights are raw ggml block_q2_K (84 B).
 */
#ifndef GEMV_Q2K_UAPI_H
#define GEMV_Q2K_UAPI_H
#include <linux/ioctl.h>
#include <linux/types.h>

#define GEMV_Q2K_DEVICE          "/dev/gemv_q2k"
#define GEMV_Q2K_ABI_VERSION     2
#define GEMV_Q2K_ENGINES         4
#define GEMV_Q2K_MAX_BLOCKS      256   /* row length <= 65536 values */
#define GEMV_Q2K_BLOCK_BYTES     84    /* sizeof(block_q2_K)         */
#define GEMV_Q2K_ACT_BLOCK_BYTES 272

struct gemv_q2k_info {
	__u32 version, engines, max_blocks, act_block_bytes;
	__u64 max_rows, max_upload_bytes;
};

struct gemv_q2k_upload {
	__u64 addr;     /* in: user pointer to rows*nb*84 bytes of block_q2_K */
	__u64 bytes;    /* in */
	__u32 handle;   /* out: non-zero, valid until FREE or close()          */
	__u32 reserved;
};

struct gemv_q2k_job {
	__u64 activation;  /* in: user pointer, nb * 272 bytes         */
	__u64 output;      /* in: user pointer, rows * 4 bytes (float) */
	__u32 handle, nb, rows;
	__u32 timeout_ms;  /* 0 = driver default (2000 ms)             */
	__u32 status;      /* out: raw hardware status register        */
	__u32 reserved;
};

#define GEMV_Q2K_IOC_MAGIC 'Q'
#define GEMV_Q2K_INFO   _IOR (GEMV_Q2K_IOC_MAGIC, 2, struct gemv_q2k_info)
#define GEMV_Q2K_UPLOAD _IOWR(GEMV_Q2K_IOC_MAGIC, 3, struct gemv_q2k_upload)
#define GEMV_Q2K_FREE   _IOW (GEMV_Q2K_IOC_MAGIC, 4, __u32)
#define GEMV_Q2K_SUBMIT _IOWR(GEMV_Q2K_IOC_MAGIC, 5, struct gemv_q2k_job)

#endif
