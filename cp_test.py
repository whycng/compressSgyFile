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

'''
设置卷积核，降采样，pca特征保留率
'''

tracecount = -1
gob_shape = (-1, -1)
downsample_factor = 8
pca_model_file = r"PCA_Model.pkl"

def read_grid_file(grid_file):
    """
    读取网格文件。

    Args:
        grid_file: 网格文件名。

    Returns:
        DataFrame: 包含网格信息的DataFrame。
    """
    with open(grid_file, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    headers = lines[:9]
    data = lines[9:]
    # print(data)
    # print(headers)

    # Create a DataFrame from the headers
    df_headers = pd.DataFrame(headers[1:], columns=headers[0])

    # Create a DataFrame from the data
    df_data = pd.DataFrame(data[1:], columns=data[0][1:])

    # Concatenate the two DataFrames
    # df = pd.concat([df_headers, df_data], ignore_index=True)

    return df_headers, df_data

def pca_compress(data, n_components=100):
    pca = PCA(n_components=n_components)
    compressed_data = pca.fit_transform(data)
    return compressed_data, pca

def compress_segy(input_file, output_file, compression_rate):
    """
    压缩 SEGY 数据。

    Args:
        input_file: 输入 SEGY 文件名。
        output_file: 输出压缩文件文件名。
        compression_rate: 压缩率，取值范围为 0 到 1，表示保留的特征比例。
    """
    global pca_model_file

    # 加载 SEGY 数据
    with segyio.open(input_file, ignore_geometry=True) as segy_file:
        data = segy_file.trace.raw[:] # 获取所有道的数据
        global tracecount
        tracecount = segy_file.tracecount # 获取道的总数
        samples_per_trace = segy_file.samples.size # 获取每道的样本数
        segy_format = int(segy_file.format)  # 获取SEGY文件的格式, 确保格式为整数类型
        print(f"Trace count: {tracecount}")
        print(f"Samples per trace: {samples_per_trace}")
        print(f"SEGY format: {segy_format}")
        # 打印数据的形状
        print(f"Data shape: {data.shape}")
        global gob_shape
        gob_shape = data.shape
        print("compress--- gob_shape:", gob_shape)

    # 定义卷积核
    kernel_size = 8
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    # 对数据进行2D卷积（平滑处理）
    smoothed_data = convolve2d(data, kernel, mode='same', boundary='wrap')

    # 降采样
    def downsample2d(data, factor):
        return data[::factor, ::factor]

    global downsample_factor    # 降采样因子
    downsampled_data = downsample2d(smoothed_data, downsample_factor)

    # 打印原始数据和平滑后数据、降采样后数据的形状
    print("Original data shape:", data.shape)
    print("Smoothed data shape:", smoothed_data.shape)
    print("Downsampled data shape:", downsampled_data.shape)
    data = downsampled_data

    # 对二维数据进行PCA压缩
    compressed_data, pca = pca_compress(data, n_components=85)
    print("pca data shape:", compressed_data.shape)
    # 存储压缩后的数据和PCA模型
    np.savez(output_file, compressed_data=compressed_data, components=pca.components_, mean=pca.mean_,tracecount=tracecount, samples_per_trace=samples_per_trace, segy_format=segy_format)

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
    global tracecount
    global pca_model_file

    # 加载压缩后的数据和 PCA 对象
    npzfile = np.load(input_file)
    compressed_data = npzfile['compressed_data']
    components = npzfile['components']
    mean = npzfile['mean']
    tracecount = npzfile['tracecount']
    samples_per_trace = npzfile['samples_per_trace']
    segy_format = npzfile['segy_format']

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

    # 插值恢复（升采样）
    def upsample2d(data, factor):
        return zoom(data, factor)

    def upsample(data, factor, original_shape):
        # 使用缩放因子进行插值
        upsampled_data = zoom(data, (factor, factor), order=1)

        # 修剪或填充数据以匹配原始形状
        upsampled_data = upsampled_data[:original_shape[0], :original_shape[1]]

        return upsampled_data

    # global downsample_factor
    # upsampled_data = upsample2d(reconstructed_data, downsample_factor)
    upsampled_data = upsample(reconstructed_data, downsample_factor, gob_shape)

    # 打印恢复后数据的形状
    print("Upsampled data shape:", upsampled_data.shape)
    reconstructed_data = upsampled_data

    # 创建 SEGY 文件规范
    spec = segyio.spec()
    spec.samples = range(samples_per_trace)
    spec.tracecount = tracecount
    spec.format = int(segy_format)  # 确保格式为整数类型

    # 保存解压缩后的数据
    with segyio.create(output_file, spec) as segy_file:
        segy_file.trace = reconstructed_data[start_trace:end_trace]#(210000, 2001)
        # segy_file.trace = reconstructed_data[:]
    # 保存解压缩后的数据
    # with segyio.create(output_file, spec) as segy_file:
    #     for i in range(start_trace - 1, end_trace):
    #         segy_file.trace[i - (start_trace - 1)] = reconstructed_data[i]


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
    global tracecount

    # 读取网格文件
    grid_file = r"E:\XueXiao\比赛-东方杯\2-compress\PostData_training-grid.dat"
    grid_header, grid_data = read_grid_file(grid_file)
    print(grid_header)
    print(grid_data)

    # 示例用法
    ori_file = r"E:\XueXiao\比赛-东方杯\2-compress\PostData-for-training.sgy"
    comp_file = 'compressed_segy_data.npz'
    compression_rate = 0.89 # 保留 95% 的特征
    compress_segy(ori_file, comp_file, compression_rate)

    output_file = 'decompressed_segy_data.segy'
    start_trace = 0
    end_trace = tracecount #int(tracecount / 5)
    decompress_segy(comp_file, output_file, start_trace, end_trace)

    # 加载原始数据
    with segyio.open(ori_file, ignore_geometry=True) as segy_file:
        original_data = segy_file.trace.raw[:]

    # 解压缩数据
    with segyio.open(output_file, ignore_geometry=True) as segy_file:
        decompressed_data = segy_file.trace.raw[:]

    # 绘制原始数据和解压缩后的数据
    plot_data(original_data, decompressed_data, start_trace, end_trace)
    print("main over")

if __name__ == "__main__":
    print("begin-----------------")

    fun_main()

    test = 1