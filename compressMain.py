import numpy as np
import pandas as pd
from sklearn.decomposition import PCA,SparseCoder
import segyio
import matplotlib.pyplot as plt
from PIL import Image
import io
import zlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import butter, filtfilt, resample
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import MiniBatchDictionaryLearning
from scipy.fftpack import dct
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.ndimage import zoom
import pywt
import joblib
import cv2

'''

'''

gob_pad_h = -1
gob_pad_w = -1
gob_shape = (-1, -1)
kernel_size = 3
downsample_factor = 3
gob_data_min = 0
gob_data_max = 0
pca_model_file = r"PCA_Model.pkl"

# 量化
def quantize_data(data, num_bits=8):
    # 将数据归一化到 [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)

    # 量化数据到 [0, 2^num_bits - 1]
    quantized_data = np.round(normalized_data * (2 ** num_bits - 1)).astype(np.uint8)

    return quantized_data, data_min, data_max

def decompress_and_dequantize_data(quantized_data, data_min, data_max, num_bits=8):
    # 将量化数据还原到 [0, 1]
    normalized_data = quantized_data.astype(np.float32) / (2 ** num_bits - 1)

    # 还原数据到原始范围
    data = normalized_data * (data_max - data_min) + data_min

    return data


def load_and_reshape_binary_data(file_path, interval=2001):
    """
    从二进制文件中读取数据，每隔 interval 个数据为一列数据，最终形成形状为 (x, interval) 的数据。
    读取的数据行数 x 是动态的，取决于数据总长度是否能被 interval 整除。如果不能整除，则报错。
    """
    try:
        # 读取二进制文件中的所有数据
        data = np.fromfile(file_path, dtype=np.float32)

        # 检查数据长度是否符合预期
        total_data_points = data.size

        # 计算行数
        num_rows = total_data_points // interval

        # 如果数据长度不能被 interval 整除，则报错
        if total_data_points % interval != 0:
            raise ValueError(f"数据长度 {total_data_points} 不能被 {interval} 整除，无法整形成形状为 (x, {interval}) 的数组")

        # 重塑数据为 (x, interval)
        reshaped_data = data.reshape(num_rows, interval)

        # 返回重塑后的数据
        return reshaped_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_data_as_binary(data, file_path):
    """
    将二维numpy数组按原来的方式（每 interval 个数据为一列）写回二进制文件。
    """
    try:
        # 确保数据类型是 float32
        data = data.astype(np.float32)

        # 将数据按行存储到二进制文件中
        with open(file_path, 'wb') as f:
            data.tofile(f)

        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def pca_compress(data, n_components):
    pca = PCA(n_components=n_components)
    compressed_data = pca.fit_transform(data)
    return compressed_data, pca

def compress_segy(input_file, output_file, compression_rate):
    """
    """
    global pca_model_file,gob_shape

    # 加载数据
    data = load_and_reshape_binary_data(input_file)
    if data is not None:
        print(f"Data shape: {data.shape}")  # 打印数据的形状
    gob_shape = data.shape


    # 降采样
    def downsample2d_cv(data, factor):
        return cv2.resize(data, (data.shape[1] // factor, data.shape[0] // factor), interpolation=cv2.INTER_CUBIC)


    # 边缘填充函数
    def pad_data(data, factor):
        h, w = data.shape
        pad_h = (factor - (h % factor)) % factor
        pad_w = (factor - (w % factor)) % factor
        padded_data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        return padded_data, pad_h, pad_w

    global downsample_factor, gob_pad_h, gob_pad_w  # 降采样因子
    # 填充边缘
    data, gob_pad_h, gob_pad_w = pad_data(data, downsample_factor)

    # 降采样
    data = downsample2d_cv(data, downsample_factor)

    # 打印原始数据和平滑后数据、降采样后数据的形状
    print("降采样 data shape:", data.shape)

    # 对二维数据进行PCA压缩
    compressed_data, pca = pca_compress(data, n_components=compression_rate)
    print("pca data shape:", compressed_data.shape)

    #量化压缩后的数据
    global gob_data_min, gob_data_max
    compressed_data, gob_data_min, gob_data_max = quantize_data(compressed_data, num_bits=8)

    # 存储压缩后的数据和PCA模型
    np.savez(output_file, compressed_data=compressed_data )

    joblib.dump(pca, pca_model_file)
    print(f"Compressed data saved to {output_file}")
    print(f"PCA model saved to {pca_model_file}")


def decompress_segy(input_file, output_file, start_trace, end_trace):
    """
    解压缩 SEGY 数据。

    Args:
        input_file: 输入压缩文件文件名。
        output_file: 输出解压缩文件文件名。
        start_trace: 解压缩的起始道号。
        end_trace: 解压缩的结束道号。
    """

    global pca_model_file

    # 加载压缩后的数据和 PCA 对象
    npzfile = np.load(input_file)
    compressed_data = npzfile['compressed_data']

    # 还原量化后的数据
    global gob_data_min, gob_data_max
    compressed_data = decompress_and_dequantize_data(compressed_data, gob_data_min, gob_data_max, num_bits=8)

    # 加载压缩后的数据和PCA模型
    pca = joblib.load(pca_model_file)
    print(f"Loaded compressed data shape: {compressed_data.shape}")

    global downsample_factor
    # 提取指定范围的道
    compressed_subset = compressed_data[(start_trace // downsample_factor) : ((end_trace // downsample_factor) + 1)]
    # compressed_subset = compressed_data[start_trace:end_trace]
    print(f"Compressed subset shape: {compressed_subset.shape}",start_trace // downsample_factor, (end_trace // downsample_factor) + 1 , "end_trace:",end_trace)

    # 使用PCA模型还原指定范围的道
    reconstructed_data = pca.inverse_transform(compressed_subset)
    print(f"Reconstructed data shape: {reconstructed_data.shape}")

    # 升采样 多尺度金字塔方法
    def upsample_cv(data, factor, original_shape):
        upsampled_data = cv2.resize(data, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        return upsampled_data


    global gob_pad_h, gob_pad_w
    print(f"升采样 Reconstructed data shape: {reconstructed_data.shape}", "gob_pad_h:", gob_pad_h, "gob_pad_w:", gob_pad_w)
    #升采样
    # upsampled_data = upsample(reconstructed_data, downsample_factor, gob_pad_h, gob_pad_w)
    upsampled_data = upsample_cv(reconstructed_data, downsample_factor, gob_shape)

    # 打印恢复后数据的形状
    print("Upsampled data shape:", upsampled_data.shape)
    reconstructed_data = upsampled_data #.T

    # 将数组存储到本地文件中
    save_data_as_binary(reconstructed_data, output_file)



def plot_data(original_data, decompressed_data, start_trace, end_trace):
    """
    绘制原始数据和解压缩后的数据。

    Args:
        original_data: 原始 SEGY 数据。
        decompressed_data: 解压缩后的 SEGY 数据。
        start_trace: 起始道号。
        end_trace: 结束道号。
    """
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(original_data[start_trace:end_trace].T, aspect='auto', cmap='seismic')
    plt.title('Original SEGY Data')
    plt.xlabel('Trace Number')
    plt.ylabel('Sample Number')

    plt.subplot(1, 2, 2)
    plt.imshow(decompressed_data.T, aspect='auto', cmap='seismic')
    plt.title('Decompressed SEGY Data')
    plt.xlabel('Trace Number')
    plt.ylabel('Sample Number')

    plt.tight_layout()
    plt.show()

def fun_main(): # 主函数
    global gob_shape

    ori_file = r"E:\app\TOOLS4\virtualBoxSharedDir\原始数据\Volume_OriFile.part000"
    comp_file = 'compressed_segy_data.npz'
    compression_rate = 0.95 # 保留 95% 的特征
    compress_segy(ori_file, comp_file, compression_rate)

    output_file = r"E:\app\TOOLS4\virtualBoxSharedDir\Volume.part000"
    start_trace = 0
    end_trace =  gob_shape[0] #int(tracecount / 5)
    decompress_segy(comp_file, output_file, start_trace, end_trace)

    # 不需要，在geoeast绘图
    # 绘制原始数据和解压缩后的数据
    # plot_data(original_data, decompressed_data, start_trace, end_trace)
    print("main over-------------------")

if __name__ == "__main__":
    print("begin-----------------")
    fun_main()
