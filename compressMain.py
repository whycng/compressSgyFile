import argparse

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

# downsample_factor = 3 # 降采样因子

gob_pad_h = -1 # 填充高，用于降采样和升采样
gob_pad_w = -1 # 填充宽，用于降采样和升采样
gob_shape = (-1, -1) # 压缩前数据形状，用于降采样和升采样
gob_data_min = 0 # 数据最小值，用于量化
gob_data_max = 0 # 数据最大值，用于量化

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


def process_data(input_file, interval, start_trace="start_trace", end_trace="end_trace"):
    """
    加载并处理数据，返回处理后的数据。

    参数:
        input_file (str): 输入文件路径
        interval (int): 采样点个数
        start_trace (int or str): 起始道（可以是整数或字符串 "start_trace" 表示默认值）
        end_trace (int or str): 终止道（可以是整数或字符串 "end_trace" 表示默认值）

    返回:
        data (ndarray): 处理后的数据
    """
    # 加载数据
    data = load_and_reshape_binary_data(input_file, interval=interval)

    # 设置默认的起始和终止行
    if start_trace == "start_trace":
        start_trace = 0
    if end_trace == "end_trace":
        end_trace = data.shape[0]

    # 检查起始和终止行是否越界
    if start_trace < 0 or end_trace > data.shape[0] or start_trace >= end_trace:
        raise ValueError("起始或终止行数越界或不合理")

    data = data[start_trace:end_trace]

    # 将数组存储到本地文件中
    # save_data_as_binary(data, "test_data_save.npy")

    global gob_shape
    gob_shape = data.shape

    if data is not None:
        print(f"Data shape: {data.shape}")  # 打印数据的形状

    return data,start_trace,end_trace

def pad_and_downsample(data, downsample_factor):
    """
    对数据进行边缘填充和降采样。

    参数:
        data (ndarray): 输入数据
        downsample_factor (int): 降采样因子

    返回:
        data (ndarray): 处理后的数据
        pad_h (int): 填充的行数
        pad_w (int): 填充的列数
    """
    def pad_data(data, factor):
        h, w = data.shape
        pad_h = (factor - (h % factor)) % factor
        pad_w = (factor - (w % factor)) % factor
        padded_data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        return padded_data, pad_h, pad_w

    # 填充边缘
    data, pad_h, pad_w = pad_data(data, downsample_factor)

    # 降采样
    def downsample2d_cv(data, factor):
        return cv2.resize(data, (data.shape[1] // factor, data.shape[0] // factor), interpolation=cv2.INTER_CUBIC)

    # 降采样
    data = downsample2d_cv(data, downsample_factor)

    # 打印降采样后数据的形状
    print("降采样 data shape:", data.shape)

    return data, pad_h, pad_w

def save_compressed_data_and_pca(output_file, compressed_data, pca):
    """
    存储压缩后的数据和 PCA 模型到一个文件。

    参数:
        output_file (str): 输出文件路径
        compressed_data (ndarray): 压缩后的数据
        pca (PCA): 训练好的 PCA 模型
    """
    # 使用 io.BytesIO 将 PCA 模型序列化为字节流
    pca_bytes = io.BytesIO()
    joblib.dump(pca, pca_bytes)
    pca_bytes.seek(0)

    # 调试信息
    print("压缩数据形状:", compressed_data.shape)
    print("PCA 字节流长度:", len(pca_bytes.getvalue()))

    # 将压缩数据和 PCA 模型存储在一个 npz 文件中
    np.savez(output_file, compressed_data=compressed_data, pca_model=pca_bytes.getvalue())

    # 验证文件内容
    with np.load(output_file, allow_pickle=True) as data:
        print("存储文件中的键名:", data.files)

def load_compressed_data_and_pca(input_file):
    """
    从文件中加载压缩后的数据和 PCA 模型。

    参数:
        input_file (str): 输入文件路径

    返回:
        compressed_data (ndarray): 压缩后的数据
        pca (PCA): 训练好的 PCA 模型
    """
    # 验证文件内容
    with np.load(input_file, allow_pickle=True) as data:
        print("<load_compressed_data_and_pca>存储文件中的键名:", data.files,"input_file:",input_file)

    # 从 npz 文件中加载数据
    npzfile = np.load(input_file, allow_pickle=True)
    print(npzfile.files)  # 打印存储在 npz 文件中的所有键名

    compressed_data = npzfile['compressed_data']

    # 从字节流中反序列化 PCA 模型
    pca_bytes = io.BytesIO(npzfile['pca_model'])
    pca = joblib.load(pca_bytes)

    return compressed_data, pca

def compress_segy(input_file, output_file, compression_PCArate ,start_trace,end_trace, interval, downsample_factor):
    """
    """
    #加载并处理数据
    data, start_trace, end_trace = process_data(input_file, interval, start_trace, end_trace)

    global gob_pad_h, gob_pad_w #,downsample_factor
    # 降采样
    data,  gob_pad_h, gob_pad_w = pad_and_downsample(data, downsample_factor)

    # 对二维数据进行PCA压缩
    compressed_data, pca = pca_compress(data, n_components=compression_PCArate)
    print("pca data shape:", compressed_data.shape)

    #量化压缩后的数据
    global gob_data_min, gob_data_max
    compressed_data, gob_data_min, gob_data_max = quantize_data(compressed_data, num_bits=8)

    # 保存压缩数据和 PCA 模型
    save_compressed_data_and_pca(output_file, compressed_data, pca)

    print(f"Compressed data saved to {output_file}")
    return start_trace, end_trace


def decompress_segy(input_file, output_file, start_trace, end_trace, downsample_factor):
    """
    解压缩 SEGY 数据。

    Args:
        input_file: 输入压缩文件文件名。
        output_file: 输出解压缩文件文件名。
        start_trace: 解压缩的起始道号。
        end_trace: 解压缩的结束道号。
    """

    # 加载压缩数据和 PCA 模型
    compressed_data, pca = load_compressed_data_and_pca(input_file)

    # 还原量化后的数据
    global gob_data_min, gob_data_max
    compressed_data = decompress_and_dequantize_data(compressed_data, gob_data_min, gob_data_max, num_bits=8)

    # global downsample_factor
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
    #升采样
    reconstructed_data = upsample_cv(reconstructed_data, downsample_factor, gob_shape)
    # 打印恢复后数据的形状
    print("Upsampled data shape:", reconstructed_data.shape)

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

    '''
    压缩模式1：compress1 inFileName  outFileName startTrace endTrace interval PCA压缩率 downsample_factor
    压缩模式2：compress2 inFileName  outFileName startTrace endTrace compressionRate

    '''

    ori_file = r"E:\app\TOOLS4\virtualBoxSharedDir\原始数据\Volume_OriFile.part000"
    comp_file = 'compressed_segy_data.npz'
    output_file = r"E:\app\TOOLS4\virtualBoxSharedDir\Volume.part000"

    # global downsample_factor
    downsample_factor = 3 # 降采样因子
    compression_PCArate = 0.95 # pca 保留 95% 的特征
    start_trace = 'start_trace' #起始道
    end_trace = 'end_trace' #终止道

    interval = 2001 #采样点个数

    # 压缩模式1
    start_trace, end_trace = compress_segy(ori_file, comp_file, compression_PCArate, start_trace, end_trace, interval, downsample_factor)
    # 解压缩模式1
    decompress_segy(comp_file, output_file, start_trace, end_trace, downsample_factor)

    print("---------------main over-------------------")


def main():
    '''
    compress1 inFileName  outFileName startTrace endTrace interval PCA压缩率 downsample_factor
    compress2 inFileName  outFileName startTrace endTrace compressionRate

    :return:
    '''

    parser = argparse.ArgumentParser(description="SEGY File Compression and Decompression")

    subparsers = parser.add_subparsers(dest='command')

    # 压缩模式1的命令行参数
    parser_compress1 = subparsers.add_parser('compress1', help='Compress SEGY file using PCA')
    parser_compress1.add_argument('inFileName', type=str, help='Input SEGY file name')
    parser_compress1.add_argument('outFileName', type=str, help='Output compressed file name')
    parser_compress1.add_argument('startTrace', type=int, help='Start trace number')
    parser_compress1.add_argument('endTrace', type=int, help='End trace number')
    parser_compress1.add_argument('interval', type=int, help='Number of samples per trace')
    parser_compress1.add_argument('PCArate', type=float, help='PCA compression rate')

    # 压缩模式2的命令行参数
    parser_compress2 = subparsers.add_parser('compress2', help='Compress SEGY file using a different method')
    parser_compress2.add_argument('inFileName', type=str, help='Input SEGY file name')
    parser_compress2.add_argument('outFileName', type=str, help='Output compressed file name')
    parser_compress2.add_argument('startTrace', type=int, help='Start trace number')
    parser_compress2.add_argument('endTrace', type=int, help='End trace number')
    parser_compress2.add_argument('compressionRate', type=float, help='Compression rate')

    # 解压缩模式1的命令行参数
    parser_decompress1 = subparsers.add_parser('decompress1', help='Decompress SEGY file')
    parser_decompress1.add_argument('inFileName', type=str, help='Input compressed file name')
    parser_decompress1.add_argument('outFileName', type=str, help='Output SEGY file name')
    parser_decompress1.add_argument('startTrace', type=int, help='Start trace number')
    parser_decompress1.add_argument('endTrace', type=int, help='End trace number')

    # 解压缩模式2的命令行参数
    parser_decompress2 = subparsers.add_parser('decompress2', help='Decompress SEGY file')
    parser_decompress2.add_argument('inFileName', type=str, help='Input compressed file name')
    parser_decompress2.add_argument('outFileName', type=str, help='Output SEGY file name')
    parser_decompress2.add_argument('startTrace', type=int, help='Start trace number')
    parser_decompress2.add_argument('endTrace', type=int, help='End trace number')

    while True:
        try:
            user_input = input("Enter command: ").split()
            if len(user_input) == 0:
                continue
            args = parser.parse_args(user_input)

            if args.command == 'compress1':
                compress_segy(args.inFileName, args.outFileName, args.PCArate, args.startTrace, args.endTrace,
                              args.interval)
            elif args.command == 'compress2':
                # Implement compress2 function
                pass
            elif args.command == 'decompress1':
                decompress_segy(args.inFileName, args.outFileName, args.startTrace, args.endTrace)
            elif args.command == 'decompress2':
                decompress_segy(args.inFileName, args.outFileName, args.startTrace, args.endTrace)
            else:
                print("Invalid command. Please use 'compress1', 'compress2', 'decompress1',or 'decompress2'.")
        except (SystemExit, KeyboardInterrupt):
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("------------------begin-----------------")
    fun_main()
    # main()