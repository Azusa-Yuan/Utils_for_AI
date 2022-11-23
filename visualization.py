import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import load_file as load
from itertools import product
import logging
# set log file
logging.basicConfig(level=logging.INFO, filename='log.txt',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# pycharm 需要的额外步骤，来让日记显示在控制台

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger = logging.getLogger()
logger.addHandler(ch)

"""
    deal_data:
    traffic_standard_transform()
    normalize()
    visualization:
    plot_grid() 绘制多个子图的曲线图
    visualize_traffic_data() 
"""

def normalize(x, mean, std):
    return (x - mean) / std

def standard_transform(data: np.array, output_dir: str):
    record_len, sensor_num, type_num = data.shape
    data_norm = np.zeros_like(data)

    for i in range(type_num):
        mean, std = data[..., i].mean(), data[..., i].std()
        logging.info(f'mean:{mean}, std:{std}')
        data_norm[..., i] = normalize(data[..., i], mean, std)

    return data_norm


#  default:choose 8 sensors, show all timeslice
def visualize_traffic_data(train_data, df_test=None, plot_random=True, save_path='test', title = 'METR-LA', lenth = -1, normalization_flag = False):
    """Plots multiple time series."""
    # add channel (record_len, sensor_num, type_num)
    if train_data.ndim == 2:
        train_data = train_data[:, :, np.newaxis]

    train_data = train_data[:lenth,...]
    record_len, sensor_num, type_num = train_data.shape

    # normalization_flag
    if normalization_flag:
        train_data = standard_transform(train_data, output_dir=save_path)

    assert sensor_num >= 8, "Must provide at least 8 sensor"
    # random choose sensors
    if plot_random:
        unique_sensor = random.sample(range(sensor_num), k=8)
    else:
        unique_sensor = range(sensor_num)[:8]

    y_label = ["type:"+str(i+1) for i in range(type_num)]
    fig, axes = plt.subplots(4, 2, figsize=(24, 14))
    # product usage: https://blog.csdn.net/happyday_d/article/details/86005024
    for uid, (idx, idy) in zip(unique_sensor, product(range(4), range(2))):
        sub_train_data = train_data[:, uid, :]
        axes[idx, idy].plot(sub_train_data, label=y_label)
        # if df_test is not None:
        #     max_ds = train_uid['ds'].max()
        #     test_uid = df_test.query('unique_id == @uid')
        #     axes[idx, idy].plot(test_uid['ds'], test_uid['y'], c='black', label='True')
        #     axes[idx, idy].plot(test_uid['ds'], test_uid['y_5'], c='blue', alpha=0.3)
        #     axes[idx, idy].plot(test_uid['ds'], test_uid['y_50'], c='blue', label='p50')
        #     axes[idx, idy].plot(test_uid['ds'], test_uid['y_95'], c='blue', alpha=0.3)
        #     axes[idx, idy].fill_between(x=test_uid['ds'],
        #                                 y1=test_uid['y_5'],
        #                                 y2=test_uid['y_95'],
        #                                 alpha=0.2, label='p5-p95')
        axes[idx, idy].set_title(f'{title}: {uid}')
        axes[idx, idy].set_xlabel('Timestamp [t]')
        axes[idx, idy].set_ylabel('Target')
        axes[idx, idy].legend(loc='upper left')
        # 设置最大的间隔数 为了让图片更美观
        axes[idx, idy].xaxis.set_major_locator(plt.MaxNLocator(20))
        # 暂时只找到这个了，改变啊x轴刻度的角度  https://blog.csdn.net/weixin_39190382/article/details/109312056
        for tick in axes[idx, idy].get_xticklabels():
            tick.set_rotation(-30)
        axes[idx, idy].grid()
    fig.subplots_adjust(hspace=0.5)

    plt.savefig(f'{save_path}.png')
    # ：在 plt.show() 后调用了 plt.savefig() ，在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），这时候你再 plt.savefig() 就会保存这个新生成的空白图片
    #  因此需要将plt.savefig()放在plt.show()之前，又或者是使用plt.gcf()固定图片，再保存
    plt.show()

# 曲线图 由nhit里得到的  他的数据结构里略怪
def plot_grid(df_train, df_test=None, plot_random=True, save_path='test', title = 'METR-LA', lenth = -1):
    """Plots multiple time series."""
    fig, axes = plt.subplots(4, 2, figsize=(24, 14))

    unique_ids = df_train['unique_id'].unique()

    assert len(unique_ids) >= 8, "Must provide at least 8 ts"

    if plot_random:
        unique_ids = random.sample(list(unique_ids), k=8)
    else:
        unique_ids = unique_ids[:8]
    # product的用法 https://blog.csdn.net/happyday_d/article/details/86005024
    for uid, (idx, idy) in zip(unique_ids, product(range(4), range(2))):
        train_uid = df_train.query('unique_id == @uid')
        axes[idx, idy].plot(train_uid['ds'][:lenth], train_uid['y'][:lenth], label='y_train', c='black')
        if df_test is not None:
            max_ds = train_uid['ds'].max()
            test_uid = df_test.query('unique_id == @uid')
            axes[idx, idy].plot(test_uid['ds'], test_uid['y'], c='black', label='True')
            axes[idx, idy].plot(test_uid['ds'], test_uid['y_5'], c='blue', alpha=0.3)
            axes[idx, idy].plot(test_uid['ds'], test_uid['y_50'], c='blue', label='p50')
            axes[idx, idy].plot(test_uid['ds'], test_uid['y_95'], c='blue', alpha=0.3)
            axes[idx, idy].fill_between(x=test_uid['ds'],
                                        y1=test_uid['y_5'],
                                        y2=test_uid['y_95'],
                                        alpha=0.2, label='p5-p95')
        axes[idx, idy].set_title(f'{title}: {uid}')
        axes[idx, idy].set_xlabel('Timestamp [t]')
        axes[idx, idy].set_ylabel('Target')
        axes[idx, idy].legend(loc='upper left')
        axes[idx, idy].xaxis.set_major_locator(plt.MaxNLocator(20))
        # 暂时只找到这个了，改变啊x轴刻度的角度  https://blog.csdn.net/weixin_39190382/article/details/109312056
        for tick in axes[idx, idy].get_xticklabels():
            tick.set_rotation(-30)
        axes[idx, idy].grid()
    fig.subplots_adjust(hspace=0.5)

    plt.savefig(f'{save_path}.png')
    # ：在 plt.show() 后调用了 plt.savefig() ，在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），这时候你再 plt.savefig() 就会保存这个新生成的空白图片
    #  因此需要将plt.savefig()放在plt.show()之前，又或者是使用plt.gcf()固定图片，再保存
    plt.show()


if __name__ == '__main__':
    data = load.Load_csv("PeMS_04/PeMS04.npz")
    data = data['data']
    visualize_traffic_data(data, lenth=12, title="PeMS_04", normalization_flag=True)
