import argparse
import sys

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import zlib
import joblib
import cv2
import pickle
from pympler import asizeof
from scipy.fft import fft, ifft
'''

'''


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

def save_data_as_binary(data, file_path,mode='wb'):
    """
    将二维numpy数组按原来的方式（每 interval 个数据为一列）写回二进制文件。
    """
    try:
        # 确保数据类型是 float32
        data = data.astype(np.float32)

        # 将数据按行存储到二进制文件中
        with open(file_path, mode) as f:
            data.tofile(f)

        # 计算并打印字节数
        file_size_bytes = data.nbytes
        print(f"Data successfully saved to {file_path}, size: {file_size_bytes} bytes")

    except Exception as e:
        print(f"An error occurred: {e}")

def append_save_data_as_binary(data, filename):
    # 假设这个函数已经实现，它将数据保存为二进制文件
    with open(filename, 'ab') as f:
        f.write(data.tobytes())

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

    return data

def save_compressed_data_and_pca(output_file, compressed_data, pca, data_min, data_max, downsample_factor):
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
    print("压缩数据形状:", compressed_data.shape ," compressed_data bytes:", len(compressed_data.tobytes()))
    print("PCA 字节流长度:", len(pca_bytes.getvalue()))

    # 使用 zlib 压缩字节流
    pca_bytes = zlib.compress(pca_bytes.getvalue())
    # 打印压缩后的字节流长度
    compressed_length = len(pca_bytes)
    print("压缩后的 PCA 字节流长度:", compressed_length)

    # 计算 compressed_data, data_min, data_max, downsample_factor 的字节数
    compressed_data_bytes = compressed_data.nbytes
    data_min_bytes = data_min.nbytes
    data_max_bytes = data_max.nbytes
    downsample_factor_bytes = np.array(downsample_factor).nbytes

    # 计算总字节数
    total_bytes = compressed_data_bytes + len(pca_bytes) + data_min_bytes + data_max_bytes + downsample_factor_bytes
    print(f"compressed_data bytes: {compressed_data_bytes}, output_file: {output_file}, total bytes: {total_bytes}")

    # 将压缩数据和 PCA 模型存储在一个 npz 文件中
    np.savez(output_file, compressed_data=compressed_data, pca_model=pca_bytes,data_min=data_min, data_max=data_max, downsample_factor=downsample_factor)

    # # 验证文件内容
    # with np.load(output_file, allow_pickle=True) as data:
    #     print("存储文件中的键名:", data.files)

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
    data_min = npzfile['data_min']
    data_max = npzfile['data_max']
    downsample_factor = npzfile['downsample_factor']

    # 从字节流中反序列化 PCA 模型
    pca_bytes = io.BytesIO(zlib.decompress(npzfile['pca_model']))
    # pca_bytes = io.BytesIO(npzfile['pca_model'])
    pca = joblib.load(pca_bytes)

    return compressed_data, pca,data_min, data_max, downsample_factor



def resize_matrix(data, newDataShape):
    m, n = data.shape
    new_m, new_n = newDataShape

    if new_m > m:
        # 填充行
        pad_top = (new_m - m) // 2
        pad_bottom = new_m - m - pad_top
        data = np.pad(data, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    elif new_m < m:
        # 削减行
        cut_top = (m - new_m) // 2
        cut_bottom = m - new_m - cut_top
        data = data[cut_top:m-cut_bottom, :]

    if new_n > n:
        # 填充列
        pad_left = (new_n - n) // 2
        pad_right = new_n - n - pad_left
        data = np.pad(data, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    elif new_n < n:
        # 削减列
        cut_left = (n - new_n) // 2
        cut_right = n - new_n - cut_left
        data = data[:, cut_left:n-cut_right]

    return data

def compress_segy(input_file, output_file, compression_PCArate ,start_trace,end_trace, interval, downsample_factor=-1):
    """
    若启用降采样，请保证输入数据的形状大于采样核
    """
    #加载并处理数据
    data, start_trace, end_trace = process_data(input_file, interval, start_trace, end_trace)


    # 降采样
    if( -1 != downsample_factor): #若启用降采样
        data = pad_and_downsample(data, downsample_factor)
        print("填充后的 data shape:", data.shape)




    # 对二维数据进行PCA压缩
    compressed_data, pca = pca_compress(data, n_components=compression_PCArate)
    print("pca data shape:", compressed_data.shape)
    print("compressed_data before kb:", compressed_data.nbytes / 1024)

    #量化压缩后的数据

    compressed_data, data_min, data_max = quantize_data(compressed_data, num_bits=8)
    print("compressed_data after kb:", compressed_data.nbytes / 1024)
    pca_size_kb = sys.getsizeof(pca) / 1024
    pca_bytes = pca.__sizeof__()
    pca_size_exact = asizeof.asizeof(pca)
    print("pca kb:", pca_size_kb, " pca_bytes :",pca_bytes , " pca_size_exact:",pca_size_exact )

    # 保存压缩数据和 PCA 模型
    save_compressed_data_and_pca(output_file, compressed_data, pca, data_min, data_max, downsample_factor)

    print(f"Compressed data saved to {output_file}")
    return start_trace, end_trace


def decompress_segy(input_file, output_file, start_trace, end_trace ):
    """
    解压缩 SEGY 数据。

    Args:
        input_file: 输入压缩文件文件名。
        output_file: 输出解压缩文件文件名。
        start_trace: 解压缩的起始道号。
        end_trace: 解压缩的结束道号。
    """

    # 加载压缩数据和 PCA 模型
    compressed_data, pca, data_min, data_max , downsample_factor= load_compressed_data_and_pca(input_file)

    # 还原量化后的数据
    compressed_data = decompress_and_dequantize_data(compressed_data, data_min, data_max, num_bits=8)

    # 提取指定范围的道
    if (-1 != downsample_factor):
        compressed_subset = compressed_data[(start_trace // downsample_factor) : ((end_trace // downsample_factor) )]
        print(f"Compressed subset shape: {compressed_subset.shape}", start_trace // downsample_factor,
              (end_trace // downsample_factor) , "end_trace:", end_trace)
    else:
        compressed_subset = compressed_data[start_trace:end_trace]

    # 使用PCA模型还原指定范围的道
    reconstructed_data = pca.inverse_transform(compressed_subset)
    print(f"Reconstructed data shape: {reconstructed_data.shape}")

    # 升采样 多尺度金字塔方法
    def upsample_cv(data, factor): # 降采样的时候进行了边缘填充，因此升采样按理说可用除尽
        h = data.shape[0]
        w = data.shape[1]
        print("Upsampling data shape:",h * factor, w * factor )
        upsampled_data = cv2.resize(data, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
        return upsampled_data

    #升采样
    if (-1 != downsample_factor): #若启用降采样
        reconstructed_data = upsample_cv(reconstructed_data, downsample_factor)
    # 打印恢复后数据的形状
    print("Upsampled data shape:", reconstructed_data.shape)

    RszieShape = ( end_trace - start_trace, reconstructed_data.shape[1])
    # 重新调整数据的形状
    reconstructed_data = resize_matrix(reconstructed_data, RszieShape)
    # 打印恢复后数据的形状
    print("resize data shape:", reconstructed_data.shape)

    # 将数组存储到本地文件中
    save_data_as_binary(reconstructed_data, output_file)


def quantize(data_real, data_imag, num_bits=8):
    real_part = data_real
    imag_part = data_imag

    real_min, real_max = real_part.min(), real_part.max()
    imag_min, imag_max = imag_part.min(), imag_part.max()

    if real_max != real_min:
        normalized_real = (real_part - real_min) / (real_max - real_min)
    else:
        normalized_real = np.zeros_like(real_part)

    if imag_max != imag_min:
        normalized_imag = (imag_part - imag_min) / (imag_max - imag_min)
    else:
        normalized_imag = np.zeros_like(imag_part)

    quantized_real = np.round(normalized_real * (2 ** num_bits - 1)).astype(np.uint8)
    quantized_imag = np.round(normalized_imag * (2 ** num_bits - 1)).astype(np.uint8)

    return quantized_real, quantized_imag, real_min, real_max, imag_min, imag_max

def dequantize(quantized_real, quantized_imag, real_min, real_max, imag_min, imag_max, num_bits=8):
    quantized_real = quantized_real.astype(np.float32)
    quantized_imag = quantized_imag.astype(np.float32)

    normalized_real = quantized_real / (2 ** num_bits - 1)
    normalized_imag = quantized_imag / (2 ** num_bits - 1)

    print("<dequantize> normalized_real shape:", normalized_real.shape)

    if real_max != real_min:
        real_part = normalized_real * (real_max - real_min) + real_min
    else:
        real_part = np.full_like(normalized_real, real_min)

    if imag_max != imag_min:
        imag_part = normalized_imag * (imag_max - imag_min) + imag_min
    else:
        imag_part = np.full_like(normalized_imag, imag_min)
    print("<dequantize> real_part shape:", real_part.shape)
    return real_part ,imag_part



def process_data_in_chunks(input_file, interval, start_trace, end_trace, chunk_size=1000):
    """
    分块处理数据，返回一个生成器，逐块生成处理后的数据。
    """
    data, _, _ = process_data(input_file, interval, start_trace, end_trace)
    for start in range(start_trace, end_trace, chunk_size):
        end = min(start + chunk_size, end_trace)
        yield data[start:end], start, end


def compress_wf(input_file, output_file, shorld, start_trace, end_trace, interval, num_bits=8, chunk_size=1000):
    compressed_real_parts = []
    compressed_imag_parts = []
    real_mins = []
    real_maxs = []
    imag_mins = []
    imag_maxs = []
    print("chunk_size:",chunk_size)

    with open(output_file, 'wb') as f:
        # 写入 interval 变量到文件
        pickle.dump(interval, f)

        for data, start, end in process_data_in_chunks(input_file, interval, start_trace, end_trace, chunk_size):

            print("data shape",data.shape)

            # 傅里叶变换
            fft_signal = fft(data)

            # 压缩: 设定阈值，只保留主要频率成分
            threshold = shorld * np.max(np.abs(fft_signal))
            compressed_data = np.where(np.abs(fft_signal) > threshold, fft_signal, 0)

            print("compressed_data.real.tobytes():", len(compressed_data.real.tobytes()))
            print("compressed_data.imag.tobytes():", len(compressed_data.imag.tobytes()))
            print("傅里叶变换，设定阈值之后的实部 compressed_data.real.shape:", compressed_data.real.shape)

            compressed_real_pca = compressed_data.real
            compressed_imag_pca = compressed_data.imag
            # # 在量化之前对 compressed_data 进行 PCA 压缩
            # global pcaREAL, pcaIMAG, glo_pca
            # glo_pca = 10
            # pcaREAL = PCA(n_components=interval//glo_pca)  # 保留 95% 的方差
            # # pcaIMAG = PCA(n_components=interval//glo_pca)  # 保留 95% 的方差
            # compressed_real_pca = pcaREAL.fit_transform(compressed_data.real)  # 对实部进行 PCA 压缩
            # # compressed_imag_pca = pcaIMAG.fit_transform(compressed_data.imag)  # 对虚部进行 PCA 压缩
            #
            # print("pca后实部 compressed_data_pca.shape:", compressed_real_pca.shape)
            # print("compressed_data_pca.tobytes():", len(compressed_real_pca.tobytes()))

            # 量化
            quantized_real, quantized_imag, real_min, real_max, imag_min, imag_max = quantize(compressed_real_pca, compressed_imag_pca, num_bits)
            real_mins.append(real_min)
            real_maxs.append(real_max)
            imag_mins.append(imag_min)
            imag_maxs.append(imag_max)
            print("量化之后的实部 quantized_real.shape:", quantized_real.shape)

            # 压缩实部和虚部
            compressed_real = zlib.compress(quantized_real.tobytes())
            compressed_imag = zlib.compress(quantized_imag.tobytes())

            print("compressed_real:", len(compressed_real))
            print("compressed_imag:", len(compressed_imag))

            # 保存到文件
            pickle.dump((compressed_real, compressed_imag, real_min, real_max, imag_min, imag_max), f)

    print(f"Compressed data saved to {output_file}")
    return start_trace, end_trace


def decompress_wf(input_file, output_file, start_trace, end_trace, num_bits=8):

    traces_processed = 0
    chunk_n = 0

    with open(output_file, 'wb') as _:
        pass  # 清空文件内容

    with open(input_file, 'rb') as f:
        interval = pickle.load(f)
        try:
            while True :
                print("chunk_n:", chunk_n)
                chunk_n += 1

                # 加载一个块的数据
                compressed_real, compressed_imag, real_min, real_max, imag_min, imag_max = pickle.load(f)
                # print("len compressed_real:", len(compressed_real), " len compressed_imag:", len(compressed_imag))

                global glo_pca
                # 解压缩并恢复数据形状
                quantized_real = np.frombuffer(zlib.decompress(compressed_real), dtype=np.uint8).reshape(-1, interval   ) # //glo_pca
                quantized_imag = np.frombuffer(zlib.decompress(compressed_imag), dtype=np.uint8).reshape(-1, interval  ) # //100 pca ues
                print("解压出来的数据实部quantized_real.shape:", quantized_real.shape)

                traces_per_chunk = quantized_real.shape[0]

                if traces_processed + traces_per_chunk <= start_trace:
                    # 如果当前块在 start_trace 之前，跳过当前块
                    traces_processed += traces_per_chunk
                    continue
                elif traces_processed >= end_trace:
                    # 如果当前块在 end_trace 之后，停止处理
                    break

                # 确定从当前块中提取的范围
                chunk_start = max(0, start_trace - traces_processed)
                chunk_end = min(traces_per_chunk, end_trace - traces_processed + 1)

                quantized_real = quantized_real[chunk_start:chunk_end, :]
                quantized_imag = quantized_imag[chunk_start:chunk_end, :]
                print("提取出的quantized_real.shape:", quantized_real.shape,"chunk_start:", chunk_start, "chunk_end:", chunk_end,
                      " start_trace:", start_trace, "end_trace:", end_trace, "traces_processed:", traces_processed)

                # 反量化
                quantized_real,quantized_imag = dequantize(quantized_real, quantized_imag, real_min, real_max, imag_min, imag_max, num_bits)
                print("反量化之后的数据实部 dequantize quantized_real shape:", quantized_real.shape)

                # Inverse transformation to recover the original compressed data
                # global pcaREAL, pcaIMAG
                # quantized_real = pcaREAL.inverse_transform(quantized_real)
                # # data_imag = pcaIMAG.inverse_transform(data.imag)
                data = quantized_real + 1j * quantized_imag
                print("逆变换后的数据实部 data real shape:", data.real.shape)

                # 逆傅里叶变换恢复信号
                signal_chunk = ifft(data).real

                # 将数据块追加到文件中
                append_save_data_as_binary(signal_chunk, output_file)

                traces_processed += traces_per_chunk

        except EOFError:
            pass

    print(f"Decompressed data saved to {output_file}")


def plot_waveforms(original_data, decompressed_data, start_trace, end_trace):
    """
    绘制原始数据和解压缩后的数据的波形图。

    Args:
        original_data: 原始 SEGY 数据。
        decompressed_data: 解压缩后的 SEGY 数据。
        start_trace: 起始道号。
        end_trace: 结束道号。
    """
    plt.figure(figsize=(14, 7))

    num_traces = end_trace - start_trace
    time_samples = original_data.shape[1]

    plt.subplot(1, 2, 1)
    for i in range(num_traces):
        plt.plot(original_data[start_trace + i] + i * 2, label=f'Trace {start_trace + i}')
    plt.title('Original SEGY Data Waveforms')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    for i in range(num_traces):
        plt.plot(decompressed_data[i] + i * 2, label=f'Trace {start_trace + i}')
    plt.title('Decompressed SEGY Data Waveforms')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

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
    压缩模式2：compress2 inFileName  outFileName startTrace endTrace interval PCA压缩率

    '''

    ori_file = r"E:\app\TOOLS4\virtualBoxSharedDir\原始数据\Volume_OriFile.part000"
    comp_file = 'compressed_segy_data.npz'
    output_file = r"E:\app\TOOLS4\virtualBoxSharedDir\Volume.part000"

    downsample_factor = 3 # 降采样因子
    compression_PCArate = 0.95 # pca 保留 95% 的特征
    start_trace = 0 #起始道
    end_trace = 2100  #终止道
    interval = 2001 #采样点个数


    # # 压缩模式1
    # start_trace, end_trace = compress_segy(ori_file, comp_file, compression_PCArate, start_trace, end_trace, interval, downsample_factor)
    # # 解压缩模式1
    # decompress_segy(comp_file, output_file, start_trace, end_trace)

    chunk_size = end_trace // 10
    shorld = 0.1
    # 压缩模式2
    start_trace, end_trace = compress_wf(ori_file, comp_file, shorld, start_trace, end_trace, interval , num_bits=8, chunk_size=chunk_size)
    # 解压缩模式2
    decompress_wf(comp_file, output_file, start_trace, end_trace )
    # decompress_wf(comp_file, output_file, end_trace // 4, end_trace // 2)


    print("---------------main over-------------------")


def main():
    '''
    compress1 inFileName outFileName startTrace endTrace interval PCArate downsample_factor
    compress2 inFileName outFileName startTrace endTrace shorld interval chunksize
    decompress1 inFileName outFileName startTrace endTrace downsample_factor
    decompress2 inFileName outFileName startTrace endTrace
    '''

    while True:
        try:
            user_input = input("Enter command: ").split()
            if len(user_input) == 0:
                continue

            command = user_input[0]
            args = user_input[1:]

            if command == 'compress1' and len(args) == 7:
                inFileName, outFileName, startTrace, endTrace, interval, PCArate, downsample_factor = args
                compress_segy(inFileName, outFileName, float(PCArate), int(startTrace), int(endTrace), int(interval),
                              int(downsample_factor))
            elif command == 'compress2' and len(args) == 7:
                inFileName, outFileName, startTrace, endTrace, shorld, interval, chunksize = args
                compress_wf(inFileName, outFileName, float(shorld), int(startTrace), int(endTrace), int(interval),
                            int(chunksize))
            elif command == 'decompress1' and len(args) == 4:
                inFileName, outFileName, startTrace, endTrace = args
                decompress_segy(inFileName, outFileName, int(startTrace), int(endTrace) )
            elif command == 'decompress2' and len(args) == 4:
                inFileName, outFileName, startTrace, endTrace = args
                decompress_wf(inFileName, outFileName, int(startTrace), int(endTrace))
            else:
                print("Invalid command or incorrect number of arguments.")
                print("Usage:")
                print("  compress1 inFileName outFileName startTrace endTrace interval PCArate downsample_factor")
                print("  compress2 inFileName outFileName startTrace endTrace shorld interval chunksize")
                print("  decompress1 inFileName outFileName startTrace endTrace  ")
                print("  decompress2 inFileName outFileName startTrace endTrace")

        except (SystemExit, KeyboardInterrupt):
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")



if __name__ == "__main__":
    print("------------------begin-----------------")
    fun_main()
    # main()
'''
 compress1 E:\app\TOOLS4\virtualBoxSharedDir\原始数据\Volume_OriFile.part000 compressed_segy_data.npz 0 210000 2001 0.95 3
 decompress1 compressed_segy_data.npz E:\app\TOOLS4\virtualBoxSharedDir\Volume.part000 0 210000  
 compress2 E:\app\TOOLS4\virtualBoxSharedDir\原始数据\Volume_OriFile.part000 compressed_segy_data.npz 0 210000 0.1 2001 2100
 decompress2 compressed_segy_data.npz E:\app\TOOLS4\virtualBoxSharedDir\Volume.part000 52500 105000


'''


# def main():
#     '''
#     compress1 inFileName  outFileName startTrace endTrace interval PCA压缩率 downsample_factor
#     compress2 inFileName  outFileName startTrace endTrace compressionRate
#     decompress1 inFileName  outFileName startTrace endTrace
#     decompress2 inFileName  outFileName startTrace endTrace
#     '''
#
#     parser = argparse.ArgumentParser(description="SEGY File Compression and Decompression")
#
#     subparsers = parser.add_subparsers(dest='command')
#
#     # 压缩模式1的命令行参数
#     parser_compress1 = subparsers.add_parser('compress1', help='Compress SEGY file using PCA')
#     parser_compress1.add_argument('inFileName', type=str, help='Input SEGY file name')
#     parser_compress1.add_argument('outFileName', type=str, help='Output compressed file name')
#     parser_compress1.add_argument('startTrace', type=int, help='Start trace number')
#     parser_compress1.add_argument('endTrace', type=int, help='End trace number')
#     parser_compress1.add_argument('interval', type=int, help='Number of samples per trace')
#     parser_compress1.add_argument('PCArate', type=float, help='PCA compression rate')
#     parser_compress1.add_argument('downsample_factor', type=int, help='Downsample factor')
#
#     # 压缩模式2的命令行参数
#     parser_compress2 = subparsers.add_parser('compress2', help='Compress SEGY file using a different method')
#     parser_compress2.add_argument('inFileName', type=str, help='Input SEGY file name')
#     parser_compress2.add_argument('outFileName', type=str, help='Output compressed file name')
#     parser_compress2.add_argument('startTrace', type=int, help='Start trace number')
#     parser_compress2.add_argument('endTrace', type=int, help='End trace number')
#     parser_compress2.add_argument('shorld', type=float, help='fft shorld')
#     parser_compress2.add_argument('interval', type=int, help='Number of samples per trace')
#     parser_compress2.add_argument('chunksize', type=int, help='chunksize for fft')
#
#     # 解压缩模式1的命令行参数
#     parser_decompress1 = subparsers.add_parser('decompress1', help='Decompress SEGY file')
#     parser_decompress1.add_argument('inFileName', type=str, help='Input compressed file name')
#     parser_decompress1.add_argument('outFileName', type=str, help='Output SEGY file name')
#     parser_decompress1.add_argument('startTrace', type=int, help='Start trace number')
#     parser_decompress1.add_argument('endTrace', type=int, help='End trace number')
#     parser_decompress1.add_argument('downsample_factor', type=int, help='downsample_factor')
#
#     # 解压缩模式2的命令行参数
#     parser_decompress2 = subparsers.add_parser('decompress2', help='Decompress SEGY file')
#     parser_decompress2.add_argument('inFileName', type=str, help='Input compressed file name')
#     parser_decompress2.add_argument('outFileName', type=str, help='Output SEGY file name')
#     parser_decompress2.add_argument('startTrace', type=int, help='Start trace number')
#     parser_decompress2.add_argument('endTrace', type=int, help='End trace number')
#
#     while True:
#         try:
#             user_input = input("Enter command: ").split()
#             if len(user_input) == 0:
#                 continue
#             args = parser.parse_args(user_input)
#
#             if args.command == 'compress1':
#                 # print(" test compress1")
#                 compress_segy(args.inFileName, args.outFileName, args.PCArate, args.startTrace, args.endTrace,
#                               args.interval, args.downsample_factor)
#             elif args.command == 'compress2':
#                 compress_wf(args.inFileName, args.outFileName, args.shorld, args.startTrace, args.endTrace,
#                             args.interval, chunk_size=args.chunksize)
#             elif args.command == 'decompress1':
#                 decompress_segy(args.inFileName, args.outFileName, args.startTrace, args.endTrace,args.downsample_factor)
#             elif args.command == 'decompress2':
#                 decompress_wf(args.inFileName, args.outFileName, args.startTrace, args.endTrace )
#             else:
#                 print("Invalid command. Please use 'compress1', 'compress2', 'decompress1', or 'decompress2'.")
#
#         except (SystemExit, KeyboardInterrupt):
#             print("Exiting...")
#             break
#         except Exception as e:
#             print(f"Error: {e}")



# def quantize(data, num_bits=8):
#     """
#     量化数据的实部和虚部。
#     """
#     real_part = data.real
#     imag_part = data.imag
#
#     real_min, real_max = real_part.min(), real_part.max()
#     imag_min, imag_max = imag_part.min(), imag_part.max()
#
#     normalized_real = (real_part - real_min) / (real_max - real_min)
#     normalized_imag = (imag_part - imag_min) / (imag_max - imag_min)
#
#     quantized_real = np.round(normalized_real * (2 ** num_bits - 1)).astype(np.uint8)
#     quantized_imag = np.round(normalized_imag * (2 ** num_bits - 1)).astype(np.uint8)
#
#     return quantized_real, quantized_imag, real_min, real_max, imag_min, imag_max
#
# def dequantize(quantized_real, quantized_imag, real_min, real_max, imag_min, imag_max, num_bits=8):
#     """
#     反量化数据的实部和虚部。
#     """
#     normalized_real = quantized_real / (2 ** num_bits - 1)
#     normalized_imag = quantized_imag / (2 ** num_bits - 1)
#
#     real_part = normalized_real * (real_max - real_min) + real_min
#     imag_part = normalized_imag * (imag_max - imag_min) + imag_min
#     print("real_part.shape:", real_part.shape,"imag_part.shape:", imag_part.shape)
#
#     return real_part + 1j * imag_part


#
# def decompress_wf2(input_file, output_file, start_trace, end_trace, interval, num_bits=8):
#     with open(output_file, 'wb') as _:
#         pass  # 清空文件内容
#
#     with open(input_file, 'rb') as f:
#         try:
#             while True:
#                 compressed_real, compressed_imag, real_min, real_max, imag_min, imag_max = pickle.load(f)
#                 print("len compressed_real:", len(compressed_real) , " len compressed_imag:", len(compressed_imag))
#                 # compressed_real, compressed_imag = pickle.load(f)
#
#                 quantized_real = np.frombuffer(zlib.decompress(compressed_real), dtype=np.uint8 ).reshape(-1, interval)
#                 quantized_imag = np.frombuffer(zlib.decompress(compressed_imag), dtype=np.uint8 ).reshape(-1, interval)
#
#                 # 反量化
#                 data = dequantize(quantized_real, quantized_imag, real_min, real_max, imag_min, imag_max, num_bits)
#
#                 # 逆傅里叶变换恢复信号
#                 signal_chunk = ifft(data).real
#
#                 # 将数据块追加到文件中
#                 append_save_data_as_binary(signal_chunk, output_file)
#         except EOFError:
#             pass
#
#     print(f"Decompressed data saved to {output_file}")
# def decompress_wf3(input_file, output_file, start_trace, end_trace, interval, num_bits=8):
#     # 从 npz 文件加载压缩数据和量化参数
#     npzfile = np.load(input_file,allow_pickle=True)
#     print(npzfile.keys())
#
#     compressed_real_parts = npzfile['compressed_real']
#     compressed_imag_parts = npzfile['compressed_imag']
#     # real_mins = npzfile['real_mins']
#     # real_maxs = npzfile['real_maxs']
#     # imag_mins = npzfile['imag_mins']
#     # imag_maxs = npzfile['imag_maxs']
#
#     print("len compressed_real_parts:", len(compressed_real_parts))
#     print("len compressed_imag_parts:", len(compressed_imag_parts))
#     # print("len real_mins:", len(real_mins))
#     # print("len real_maxs:", len(real_maxs))
#     # print("len imag_mins:", len(imag_mins))
#     # print("len imag_maxs:", len(imag_maxs))
#
#     decompressed_data = []
#
#     for i in range(len(compressed_real_parts)):
#         # Decompress real parts
#         zobj_real = zlib.decompressobj()
#         dataZ_real = zobj_real.decompress(compressed_real_parts[i])
#         dataZ_real += zobj_real.flush()  # Ensure all data is processed
#         quantized_real = np.frombuffer(dataZ_real, dtype=np.uint8).reshape(-1, interval)
#
#         # Decompress imaginary parts
#         zobj_imag = zlib.decompressobj()
#         dataZ_imag = zobj_imag.decompress(compressed_imag_parts[i])
#         dataZ_imag += zobj_imag.flush()  # Ensure all data is processed
#         quantized_imag = np.frombuffer(dataZ_imag, dtype=np.uint8).reshape(-1, interval)
#
#         # quantized_real = np.frombuffer(zlib.decompress(compressed_real_parts[i]), dtype=np.uint8).reshape(-1, interval)
#         # quantized_imag = np.frombuffer(zlib.decompress(compressed_imag_parts[i]), dtype=np.uint8).reshape(-1, interval)
#
#         data = quantized_real + 1j * quantized_imag
#
#         # 反量化
#         # data = dequantize(quantized_real, quantized_imag, real_mins[i], real_maxs[i], imag_mins[i], imag_maxs[i], num_bits)
#         decompressed_data.append(data)
#
#         # 释放不再需要的数据块内存
#         del quantized_real, quantized_imag, data
#
#     decompressed_data = np.concatenate(decompressed_data, axis=0)
#
#     # 逆傅里叶变换恢复信号
#     signal = ifft(decompressed_data).real
#     # 将数组存储到本地文件中
#     save_data_as_binary(signal, output_file)
#
#     global testData
#     # plot_waveforms(testData, signal, start_trace, end_trace)
#
#     print(f"Decompressed data saved to {output_file}")
# def compress_wf(input_file, output_file, shorld ,start_trace,end_trace, interval ,num_bits=8):
#
#     #加载并处理数据
#     data, start_trace, end_trace = process_data(input_file, interval, start_trace, end_trace)
#
#     global testData
#     testData = data
#
#     # 傅里叶变换
#     fft_signal = fft(data)
#     # 压缩: 设定阈值，只保留主要频率成分
#     threshold = shorld * np.max(np.abs(fft_signal))
#     compressed_data = np.where(np.abs(fft_signal) > threshold, fft_signal, 0)
#
#     # 分别处理实部和虚部
#     real_part = compressed_data.real
#     imag_part = compressed_data.imag
#
#     print("compressed_data shape:", compressed_data.shape)
#     # 将压缩数据和量化参数存储在一个 npz 文件中
#     np.savez("ffz_real.ff", compressed_data=zlib.compress(real_part.astype(np.float32).tobytes()))
#     np.savez("ffz_imag.ff", compressed_data=zlib.compress(imag_part.astype(np.float32).tobytes()))
#
#     print(f"Compressed data saved to {output_file}")
#     return start_trace, end_trace
#
# def decompress_wf(input_file, output_file, start_trace, end_trace, interval, num_bits=8):
#     # Load real and imaginary parts
#     npzff_real = np.load("ffz_real.ff.npz")
#     npzff_imag = np.load("ffz_imag.ff.npz")
#
#     # Decompress and extract real and imaginary parts
#     real_part_loaded = np.frombuffer(zlib.decompress(npzff_real['compressed_data']), dtype=np.float32).reshape(-1, interval)
#     imag_part_loaded = np.frombuffer(zlib.decompress(npzff_imag['compressed_data']), dtype=np.float32).reshape(-1, interval)
#
#     data = real_part_loaded + 1j * imag_part_loaded
#
#     # 逆傅里叶变换恢复信号
#     signal = ifft(data).real
#     # 将数组存储到本地文件中
#     save_data_as_binary(signal, output_file)
#
#     global testData
#     # plot_waveforms(testData, signal, start_trace, end_trace)
#
#     print(f"Decompressed data saved to {output_file}")
