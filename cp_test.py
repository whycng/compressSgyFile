import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import segyio
import matplotlib.pyplot as plt

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

def compress_segy(input_file, output_file, compression_rate):
    """
    压缩 SEGY 数据。

    Args:
        input_file: 输入 SEGY 文件名。
        output_file: 输出压缩文件文件名。
        compression_rate: 压缩率，取值范围为 0 到 1，表示保留的特征比例。
    """

    # 加载 SEGY 数据
    with segyio.open(input_file, ignore_geometry=True) as segy_file:
        data = segy_file.trace.raw[:] # 获取所有道的数据
        tracecount = segy_file.tracecount # 获取道的总数
        samples_per_trace = segy_file.samples.size # 获取每道的样本数
        segy_format = int(segy_file.format)  # 获取SEGY文件的格式, 确保格式为整数类型
        print(f"Trace count: {tracecount}")
        print(f"Samples per trace: {samples_per_trace}")
        print(f"SEGY format: {segy_format}")
        # 打印数据的形状
        print(f"Data shape: {data.shape}")


    # 使用 PCA 进行降维
    pca = PCA(n_components=compression_rate)
    compressed_data = pca.fit_transform(data)

    # 保存压缩后的数据和 PCA 对象
    np.savez(output_file, compressed_data=compressed_data, components=pca.components_, mean=pca.mean_, tracecount=tracecount, samples_per_trace=samples_per_trace, segy_format=segy_format)

def decompress_segy(input_file, output_file, start_trace, end_trace):
    """
    解压缩 SEGY 数据。

    Args:
        input_file: 输入压缩文件文件名。
        output_file: 输出解压缩文件文件名。
        start_trace: 解压缩的起始道号。
        end_trace: 解压缩的结束道号。
    """

    # 加载压缩后的数据和 PCA 对象
    npzfile = np.load(input_file)
    compressed_data = npzfile['compressed_data']
    components = npzfile['components']
    mean = npzfile['mean']
    tracecount = npzfile['tracecount']
    samples_per_trace = npzfile['samples_per_trace']
    segy_format = npzfile['segy_format']

    # 使用 PCA 进行重建
    pca = PCA()
    pca.components_ = components
    pca.mean_ = mean
    reconstructed_data = pca.inverse_transform(compressed_data)
    print("reconstructed_data.shape: ", reconstructed_data.shape)

    # 创建 SEGY 文件规范
    spec = segyio.spec()
    spec.samples = range(samples_per_trace)
    spec.tracecount = tracecount
    spec.format = int(segy_format)  # 确保格式为整数类型

    # 保存解压缩后的数据
    with segyio.create(output_file, spec) as segy_file:
        segy_file.trace = reconstructed_data[start_trace:end_trace]#(210000, 2001)

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



# 读取网格文件
grid_file = r"E:\XueXiao\比赛-东方杯\2-compress\PostData_training-grid.dat"
grid_header, grid_data = read_grid_file(grid_file)
print(grid_header)
print(grid_data)

# 示例用法
ori_file = r"E:\XueXiao\比赛-东方杯\2-compress\PostData-for-training.sgy"
comp_file = 'compressed_segy_data.npz'
compression_rate = 0.7  # 保留 95% 的特征
compress_segy(ori_file, comp_file, compression_rate)

output_file = 'decompressed_segy_data.segy'
start_trace = 1
end_trace = 4000
decompress_segy(comp_file, output_file, start_trace, end_trace)

# 加载原始数据
with segyio.open(ori_file, ignore_geometry=True) as segy_file:
    original_data = segy_file.trace.raw[:]

# 解压缩数据
with segyio.open(output_file, ignore_geometry=True) as segy_file:
    decompressed_data = segy_file.trace.raw[:]

# 绘制原始数据和解压缩后的数据
plot_data(original_data, decompressed_data, start_trace, end_trace)