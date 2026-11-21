#ifndef GEMV_Q2K_UAPI_H
#define GEMV_Q2K_UAPI_H
#include <linux/ioctl.h>
#include <linux/types.h>
#define GEMV_Q2K_IOC_MAGIC 'Q'
/* User addresses are copied to/from coherent buffers by the kernel driver. */
struct gemv_q2k_job { __u64 weights, activation, output; __u32 nb, rows0, rows1, timeout_ms, status; };
#define GEMV_Q2K_SUBMIT _IOWR(GEMV_Q2K_IOC_MAGIC, 1, struct gemv_q2k_job)
#endif
