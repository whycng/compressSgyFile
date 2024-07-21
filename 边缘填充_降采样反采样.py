
'''
对于 m*n 的数据，进行 k*k 的降采样，考虑边缘填充问题，再进行 k*k 的升采样，保证升采样之后的数据形状不变

输入： m*n 的数据，k 是降采样因子，pad_h，pad_w 是边缘填充的行数和列数，用于保证输出的尺寸不变

一般来说，3*3的卷积核，填充1，5*5的填充2，7*7的填充3
'''

import numpy as np

# 边缘填充函数
def pad_data(data, factor):
    h, w = data.shape
    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor
    padded_data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return padded_data, pad_h, pad_w

# 降采样函数
def downsample(data, factor):
    (h, w) = data.shape
    reshaped = data.reshape(h // factor, factor, w // factor, factor)
    downsampled = reshaped.mean(axis=(1, 3))
    return downsampled

# 升采样函数并裁剪填充值
def upsample(data, factor, pad_h, pad_w):
    upsampled = np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)
    # 计算裁剪的边界
    crop_h = (pad_h + 1) // 2
    crop_w = (pad_w + 1) // 2
    # 裁剪掉填充部分，使得尺寸还原到开始的形状
    upsampled = upsampled[crop_h:upsampled.shape[0] - (pad_h - crop_h), crop_w:upsampled.shape[1] - (pad_w - crop_w)]
    return upsampled

# 示例数据
data = np.random.rand(210, 2001)

# 降采样和升采样因子
factor = 5

# 填充数据
padded_data, pad_h, pad_w = pad_data(data, factor)

print("Original shape:", data.shape, "pad_h:", pad_h, "pad_w:", pad_w)
print("Padded shape:", padded_data.shape)

# 降采样
downsampled_data = downsample(padded_data, factor)
print("Downsampled shape:", downsampled_data.shape)

# 升采样并裁剪至原始形状
upsampled_data = upsample(downsampled_data, factor, pad_h, pad_w)
print("Upsampled shape:", upsampled_data.shape)