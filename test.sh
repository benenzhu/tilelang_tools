cd testing

rm -rf ~/.tilelang/cache
N=""
N="-n 50"
pytest --verbose --color=yes --durations=0 --showlocals --cache-clear -vv \
  $N --ignore=./python/runtime --ignore=./python/transform --ignore=./python/language/test_tilelang_language_reduce.py --ignore=testing/python/language/test_tilelang_language_atomic.py::test_tile_atomic_max --ignore=testing/python/language/test_tilelang_language_int64.py::test_fill_static --ignore=testing/python/language/test_tilelang_language_eager_jit.py::test_jit2_many_annot --ignore=python/language/test_tilelang_language_cumsum.py  --ignore=python/issue/test_tilelang_issue_1106.py \
  python/
  python/amd/test_tilelang_gemm_mfma_intrinsic.py::test_assert_tl_matmul[128-128-128-float16-float16-float32-False-True-1]
  python/amd/test_tilelang_gemm_mfma_intrinsic.py
  python/amd/test_tilelang_gemm_mfma_preshuffle.py
  python/amd/test_tilelang_test_amd.py::test_gemm_f16f32f32_nt
  python/tilelibrary/test_tilelang_tilelibrary_gemm.py::test_gemm_sr[512-1024-768-False-True-float16-float16-float32-128-256-32-2-128] 
  python/amd/test_tilelang_gemm_mfma_intrinsic.py::test_assert_tl_matmul
  python/amd/test_tilelang_gemm_mfma_intrinsic.py::test_assert_tl_matmul[128-128-128-bfloat16-bfloat16-float32-False-True-1]
  python/amd/test_tilelang_gemm_mfma_intrinsic.py
  python/debug/test_tilelang_debug_print.py::test_debug_print_buffer_rocm_fp8

  python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul
  python/amd/test_tilelang_gemm_mfma_intrinsic.py::test_assert_tl_matmul[128-128-128-float8_e4m3fn-float16-float32-False-True-1]
#   ./python/
#   ./python/



# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-True-1-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-False-1-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-True-2-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-False-2-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/debug/test_tilelang_debug_print.py::test_debug_print_buffer_rocm_fp8 - RuntimeError: Compilation Failed! ['hipcc', '-std=c++17', '-fPIC', '--offload-arch=gfx950', '--shared', '/tmp/tmpo1gb83nj.cpp', '-I/root/tilelang/3rdparty/composable_kernel/include', '-I/root/tilelang/3rdparty/../src', '-o', '/tmp/tmpo1gb83nj.so']