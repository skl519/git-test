import torch
import triton
import triton.language as tl


# 定义 Triton 内核
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                        offsets=(block_start_m, k), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
        b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                        offsets=(k, block_start_n), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(0, 1))

        a_submatrix = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_submatrix = tl.load(b_block_ptr, boundary_check=(0, 1))

        accumulator += tl.dot(a_submatrix, b_submatrix)

    accumulator = leaky_relu(accumulator)

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(block_start_m, block_start_n), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# 函数封装
def matmul(a, b, BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 计算网格大小
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return c


# 测试
if __name__ == '__main__':
    # 在GPU上生成随机矩阵
    A = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    B = torch.randn(256, 192, device='cuda', dtype=torch.float32)

    # 调用matmul函数进行矩阵乘法
    C = matmul(A, B)

    print(C.shape)  # 应该输出torch.Size([128, 192])
