/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * gemv_q2k user/kernel ABI (version 3): K-quant decode GEMV, zero-copy weights.
 *
 * REGISTER pins the user pages holding a weight tensor (llama.cpp's own mmap
 * or malloc buffer) and builds a page table the FPGA walks. Nothing is copied
 * and no CMA is used for weights. Pages stay pinned until FREE or close().
 *
 * SUBMIT copies the activation (nb * 272 B) in and rows * 4 B of results out.
 * Activation block (272 B per 256 values):
 *   bytes 0..3 float d, 4..15 zero, 16..271 int8 qs[256]   (== block_q8_K d, qs)
 * Weight blocks are raw ggml block_q{2,3,4,5,6}_K; `type` is the ggml_type value.
 */
#ifndef GEMV_Q2K_UAPI_H
#define GEMV_Q2K_UAPI_H
#include <linux/ioctl.h>
#include <linux/types.h>

#define GEMV_Q2K_DEVICE          "/dev/gemv_q2k"
#define GEMV_Q2K_ABI_VERSION     3
#define GEMV_Q2K_ENGINES         2
#define GEMV_Q2K_MAX_BLOCKS      256
#define GEMV_Q2K_ACT_BLOCK_BYTES 272

/* ggml_type values accepted in gemv_q2k_job.type */
#define GEMV_Q2K_TYPE_Q2_K 10
#define GEMV_Q2K_TYPE_Q3_K 11
#define GEMV_Q2K_TYPE_Q4_K 12
#define GEMV_Q2K_TYPE_Q5_K 13
#define GEMV_Q2K_TYPE_Q6_K 14

struct gemv_q2k_info {
	__u32 version, engines, max_blocks, act_block_bytes;
	__u64 max_rows;
	__u64 max_pin_bytes;  /* per registration */
	__u32 type_mask;      /* bit t set = ggml_type t supported */
	__u32 page_bytes;
};

struct gemv_q2k_register {
	__u64 addr;     /* in: user address of the tensor, 16-byte aligned */
	__u64 bytes;    /* in */
	__u32 handle;   /* out: non-zero */
	__u32 reserved;
};

struct gemv_q2k_job {
	__u64 activation;  /* in: user pointer, nb * 272 bytes         */
	__u64 output;      /* in: user pointer, rows * 4 bytes (float) */
	__u32 handle, type, nb, rows;
	__u32 timeout_ms;  /* 0 = driver default (2000 ms)             */
	__u32 status;      /* out: raw hardware status register        */
};

#define GEMV_Q2K_IOC_MAGIC 'Q'
#define GEMV_Q2K_INFO     _IOR (GEMV_Q2K_IOC_MAGIC, 0x10, struct gemv_q2k_info)
#define GEMV_Q2K_REGISTER _IOWR(GEMV_Q2K_IOC_MAGIC, 0x11, struct gemv_q2k_register)
#define GEMV_Q2K_FREE     _IOW (GEMV_Q2K_IOC_MAGIC, 0x12, __u32)
#define GEMV_Q2K_SUBMIT   _IOWR(GEMV_Q2K_IOC_MAGIC, 0x13, struct gemv_q2k_job)

#endif
