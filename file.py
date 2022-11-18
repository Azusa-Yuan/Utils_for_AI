file_path = 'METR-LA.H5'
import pandas as pd
import numpy as np
import pickle
import h5py as h5

#------------------------------------------------------
"""
    实现的功能
    结构方面：
    dataframe create creat_dataframe()
    dataframe处理类  frame_process
    
    I/O方面:
    process_h5()
    Process_HDF()、Process_HDF_2()
    Process_p()
    Process_pkl()
    
    
"""
#------------------------------------------------------


def creat_dataframe():
    df = pd.DataFrame(
        data=np.random.randint(
            0, 10, (6, 4)),
        columns=["a", "b", "c", "d"])
    print(df)


# 专门针对DataFrame的处理
class frame_process:
    # 返回index值
    def index(self, df, value=True):
        data = df.index
        if value:
            data = data.values
        return data

    # 返回列属性的值
    def columns(self, df, value=True):
        data = df.columns
        if value:
            data = data.values
        return data

# h5文件 由key value组成,但不是dict https://www.cnblogs.com/yld321/p/14851388.html
def process_h5():
    with h5.File(file_path, "r") as f:
        groups = []
        data = []
        print(type(f))
        for k in f.keys():
            if isinstance(f[k], h5.Dataset):
                print(f[k].value)
            else:
                print(f[k].name)
                groups.append(f[k].name)
        for i in groups:
            print(f[i].values)
            data.append(f[i].values)


# HDF文件，文件后缀名也是h5 https://zhuanlan.zhihu.com/p/352437247
# HDFStore 是一个类似 dict 的对象，它使用 PyTables 库并以高性能的 HDF5 格式来读写 pandas 对象。
def Process_HDF():
    with pd.HDFStore(file_path) as store:
        for key in store.keys():
            print(key)
            print(store[key])
            print(type(store[key]))
        data = store['/df']
        deal = frame_process()
        # 获取行索引
        data_index = deal.index(data)
        # 插入新的列， 列的位置  名字 值
        data.insert(loc=0, column='ds', value=data_index)
        # 重置索引 drop=True 为删除原索引
        data.reset_index(drop=True)
        print(data)
        # 保存文件
        data.to_csv('METR-LA.csv')


        # 获取列索引
        # print(data.columns.values)


# 第二种方法  该方法格式不是df.frame
def Process_HDF_2():
    df = pd.read_hdf(file_path)
    data = df.values
    print(data)
    print(data.shape)

    data = data[..., [0]]
    print(data)


#.p文件的读取与存储 p文件是m文件的加密格式，一般是为了防止算法暴露而转化的，在函数调用的时候优先于m文件。
def Process_p():
    data = pickle.load(open('METR-LA_720/NHITS/num_1/trials.p', 'rb'))
    print(dir(data))
    print(data.losses())
    print(data.best_trial)


def Process_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Process_p()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



