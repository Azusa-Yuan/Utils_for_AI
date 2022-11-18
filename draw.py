import matplotlib.pyplot as plt
import random
import pandas as pd
import file
from itertools import product
"""
    plot_grid() 绘制多个子图的曲线图
"""


# 曲线图 由nhit里得到的
def plot_grid(df_train, df_test=None, plot_random=True, save_path='test', title = 'METR-LA', lenth = -1):
    """Plots multiple time series."""
    fig, axes = plt.subplots(4, 2, figsize=(24, 14))

    unique_ids = df_train['unique_id'].unique()

    assert len(unique_ids) >= 8, "Must provide at least 8 ts"

    if plot_random:
        unique_ids = random.sample(list(unique_ids), k=8)
    else:
        unique_uids = unique_ids[:8]
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


Y_df = pd.read_csv('df_y.csv', )
plot_grid(Y_df, lenth=12)
