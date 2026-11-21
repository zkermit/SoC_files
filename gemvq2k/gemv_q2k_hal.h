#pragma once
#include <stdbool.h>
#include <stdint.h>
/* nb is Q2_K blocks per row; n must equal nb*256.  Returns false on fallback. */
bool gemv_q2k_try(const void *weights_q2k, const float *x, float *dst, uint32_t nb, uint32_t rows);
