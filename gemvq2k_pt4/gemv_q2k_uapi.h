#ifndef GEMV_Q2K_UAPI_H
#define GEMV_Q2K_UAPI_H
#include <linux/ioctl.h>
#include <linux/types.h>
#define GEMV_Q2K_IOC_MAGIC 'K'
struct gemv_q2k_register { __u64 address, bytes; __u32 handle, reserved; };
struct gemv_q2k_submit { __u32 handle, start_block, nb, rows, timeout_ms, status; __u64 activation, output; };
#define GEMV_Q2K_REGISTER _IOWR(GEMV_Q2K_IOC_MAGIC, 1, struct gemv_q2k_register)
#define GEMV_Q2K_SUBMIT   _IOWR(GEMV_Q2K_IOC_MAGIC, 2, struct gemv_q2k_submit)
#endif
