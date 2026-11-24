#pragma once
#include <stdint.h>
int gemv_q2k_u96_available(void);
int gemv_q2k_u96_run(const void *weights, const void *activation_q8k, float *output, uint32_t nb, uint32_t rows);
int gemv_q2k_u96_last_ok(void);
