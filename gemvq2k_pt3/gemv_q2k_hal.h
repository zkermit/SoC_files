#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * K-quant decode-GEMV offload (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K weights).
 *
 * gemv_q2k_eligible(): cheap, thread-safe and deterministic. Every ggml worker
 * evaluates it for the same node and gets the same answer (it depends only on
 * the arguments and on state fixed when the device is first opened).
 *   type  = ggml_type of the weights, nb = blocks per row (ne00 / 256),
 *   rows  = ne01, weights_buffer = tensor lives in a WEIGHTS buffer
 *   (registrations are cached by address, so contents must not change).
 *
 * gemv_q2k_try(): dst[r] = dot(row r, activation). `q8k` is the activation
 * already quantised by ggml as nb consecutive block_q8_K (292 bytes each), so
 * the FPGA sees exactly what the CPU path would. The first call for a tensor
 * pins its pages (zero-copy, once). Returns false on any failure; dst is then
 * undefined and the caller must compute it on the CPU.
 *
 * Env: GEMV_Q2K_DISABLE=1      turn the offload off
 *      GEMV_Q2K_MIN_ROWS=<n>   smallest matrix sent to the FPGA (default 16)
 *      GEMV_Q2K_TYPES=<list>   e.g. "q2_k,q3_k" to limit offloaded types
 */
bool gemv_q2k_eligible(int type, uint32_t nb, uint64_t rows, bool weights_buffer);
bool gemv_q2k_try(int type, const void *weights, const void *q8k, float *dst, uint32_t nb, uint32_t rows);

#ifdef __cplusplus
}
#endif
