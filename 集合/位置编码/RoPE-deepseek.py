import torch
import math

dim: int = 64               # 嵌入的维度（需要为偶数，因为会以一对一对的方式使用）
seqlen: int = 4096 * 4      # 当前处理的序列长度
beta_fast: int = 32         # 用于计算修正范围的快速旋转参数
beta_slow: int = 1          # 用于计算修正范围的慢速旋转参数
base: float = 10000.0       # 底数，用于计算频率
factor: float = 40          # 用于调整频率的因子
original_seq_len: int = 4096  # 原始序列长度（如果当前序列长度超过这个值，则需要进行修正）

def precompute_freqs_cis() -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.
    预计算位置编码所需的复数旋转因子

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    def find_correction_dim(num_rotations, dim, base, max_seq_len): # 计算给定旋转次数的修正维度
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base)) #

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):   # 计算给定旋转次数的修正范围
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):  # 计算线性斜坡函数
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # 如果序列长度大于原始序列长度，计算修正范围和平滑因子，并调整频率值
    if seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


# ----------------------------
# 以下是测试用例
# ----------------------------

def test_case(seq_length: int, description: str):
    """
    单个测试用例：
      - 修改全局变量 seqlen 和 original_seq_len，
      - 构造输入张量，
      - 计算 freqs_cis，
      - 应用旋转位置编码，
      - 检查输出形状，并做简单数值验证。
    """
    global seqlen, original_seq_len, dim

    print("\n==============================")
    print(f"测试用例：{description}")
    print(f"使用序列长度: {seq_length}")

    # 设置全局参数（对于测试可临时修改）
    seqlen = seq_length
    # 这里 original_seq_len 可设为与 seq_length 相同或较小，以测试是否进入修正分支
    original_seq_len = seq_length if seq_length <= 16 else 16

    batch_size = 2
    n_heads = 4
    # 输入张量 x 形状：(batch, seqlen, n_heads, dim)，dim 必须与全局变量保持一致（这里为64）
    x = torch.randn(batch_size, seqlen, n_heads, dim)

    # 预计算旋转频率（复数形式），形状：(seqlen, dim//2)
    freqs_cis = precompute_freqs_cis()
    print("预计算频率张量 shape:", freqs_cis.shape)

    # 应用旋转位置编码
    y = apply_rotary_emb(x, freqs_cis)
    print("输入张量 shape:", x.shape)
    print("输出张量 shape:", y.shape)

    # 检查输出形状与输入形状是否一致
    assert y.shape == x.shape, "错误：输出张量的形状应与输入张量一致。"

    # 检查：对全零输入，输出应全为零
    x_zeros = torch.zeros_like(x)
    y_zeros = apply_rotary_emb(x_zeros, freqs_cis)
    assert torch.allclose(y_zeros, torch.zeros_like(y_zeros)), "错误：对全零输入，输出不应产生非零值。"

    print("测试用例通过！")


def run_tests():
    """
    运行所有测试用例。
    """
    # 测试用例1：序列长度较小，不触发修正（seq_length 与 original_seq_len 相同）
    test_case(seq_length=16, description="序列长度等于原始长度，不触发频率修正")

    # 测试用例2：序列长度大于 original_seq_len，触发修正逻辑
    test_case(seq_length=32, description="序列长度大于原始长度，触发频率修正")


if __name__ == '__main__':
    run_tests()
