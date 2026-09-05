  // preload s.f. and zeros
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    // for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale =
        *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
    /*
    if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 &&
    threadIdx.y == 0){ printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x,
    B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x,
    B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
    }
    */
    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    // Precompute negative scaled zero -> - (Z * S)
    uint4 neg_z_scaled;
    // 1. Z * S + 0
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(neg_z_scaled.x)
                 : "r"(B_loaded_zero.x), "r"(B_loaded_scale.x), "r"(ZERO));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(neg_z_scaled.y)
                 : "r"(B_loaded_zero.y), "r"(B_loaded_scale.y), "r"(ZERO));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(neg_z_scaled.z)
                 : "r"(B_loaded_zero.z), "r"(B_loaded_scale.z), "r"(ZERO));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(neg_z_scaled.w)
                 : "r"(B_loaded_zero.w), "r"(B_loaded_scale.w), "r"(ZERO));
    // 2. Negate the result
    asm volatile("neg.f16x2 %0, %1;\n"
                 : "=r"(neg_z_scaled.x)
                 : "r"(neg_z_scaled.x));
    asm volatile("neg.f16x2 %0, %1;\n"
                 : "=r"(neg_z_scaled.y)
                 : "r"(neg_z_scaled.y));
    asm volatile("neg.f16x2 %0, %1;\n"
                 : "=r"(neg_z_scaled.z)
                 : "r"(neg_z_scaled.z));
    asm volatile("neg.f16x2 %0, %1;\n"
                 : "=r"(neg_z_scaled.w)
                 : "r"(neg_z_scaled.w));

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      // B: 32 x 136 (128+8) float16
      // each warp: 32 x 4
      // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus
      // zero -> WB UINT4
      // *(uint4*)(B_shared + ((((ax0_ax1_fused_0 * 544) + (((int)threadIdx.y) *
      // 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15)
      // * 8))) = *(uint4*)(B + ((((((k_0_0 * 163840) + (ax0_ax1_fused_0 *
      // 20480)) + (((int)threadIdx.y) * 10240)) + ((((int)threadIdx.x) >> 4) *
      // 5120)) + (((int)blockIdx_y) * 128)) + ((((int)threadIdx.x) & 15) *
      // 8))); row stride in shared memory: (NWARPS * 32 * 8 / cta_N)
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

      // deq = q * scale + (-zero * scale) using a single FMA per register
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x),
                     "r"(neg_z_scaled.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y),
                     "r"(neg_z_scaled.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z),
                     "r"(neg_z_scaled.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w),
                     "r"(neg_z_scaled.w));

      /*
      if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 ==
      0 && threadIdx.x == 17 && threadIdx.y == 0){ printf("[x] %X %X %X %X\n",
      B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
      }
      */

      // write back
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) =
          B_loaded_fp16;
    }



The modified code for the branch. 
