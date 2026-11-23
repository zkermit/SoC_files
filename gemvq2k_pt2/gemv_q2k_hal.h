#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Cheap, thread-safe and deterministic: every ggml worker thread evaluates it
 * for the same node and gets the same answer (depends only on the arguments
 * and on state fixed when the device is first opened).
 * nb = Q2_K blocks per row (ne00 / 256), rows = ne01.
 * weights_buffer must be true only for tensors in a WEIGHTS buffer, because
 * uploads are cached by address and must never change behind our back.
 *
 * Env: GEMV_Q2K_DISABLE=1      turn the offload off
 *      GEMV_Q2K_MIN_ROWS=<n>   smallest matrix sent to the FPGA (default 64)
 */
bool gemv_q2k_eligible(uint32_t nb, uint64_t rows, bool weights_buffer);

/*
 * dst[r] = dot(weights row r, x) with llama.cpp's Q2_K x Q8_K numerics.
 * The first call for a weight tensor uploads it to device memory (once);
 * later calls only send the quantised activation.
 * Returns false on any failure; dst content is then undefined and the caller
 * must compute it on the CPU.
 */
bool gemv_q2k_try(const void *weights_q2k, const float *x, float *dst, uint32_t nb, uint32_t rows);

#ifdef __cplusplus
}
#endif
