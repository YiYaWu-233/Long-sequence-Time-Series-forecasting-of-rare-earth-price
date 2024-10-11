import os
import numpy as np
import pandas as pd
from tools.Deco_model import deco_method


class data_load(object):
    def __init__(self, args):
        self.scaler = None
        self.args = args
        self.data = args.data
        self.data_path = args.data_path
        self.img_path = args.img_path
        self.data_name = args.data_name
        self.target = args.target
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.test_len = args.test_len
        self.scale = args.scale
        self.Deco_model = args.Deco_model
        self.checkpoints = args.checkpoints

    def get_data(self):
        dataset = pd.read_csv(os.path.join(self.data_path, self.data_name))
        dataset['日期'] = pd.to_datetime(dataset['日期'])
        if self.data == '稀土价格':
            dataset = dataset[dataset.日期 <= '2024/3/14'].copy()
            dataset = pd.DataFrame(dataset, columns=['日期', '氧化镧平均价', '氧化铈平均价', '氧化镨平均价', '氧化钕平均价'])
            pd_features = pd.DataFrame(dataset, columns=[self.target])
        else:
            pd_features = pd.DataFrame(dataset[-2500:], columns=[self.target])  # 提取预测目标
        print('dataSet-shape:', dataset.shape)
        print('PD_features-shape:', pd_features.shape)
        return [dataset, pd_features]

    def data_processing(self, dataset):
        if self.scale:
            self.scaler = StandardScaler()
            train_data = dataset[:-self.test_len]
            self.scaler.fit(train_data.values)
            dataset = self.scaler.transform(dataset.values)
            dataset = pd.DataFrame(dataset, columns=[self.target])  # 提取预测目标
        if self.Deco_model:
            t_components = deco_method(dataset, args=self.args)
            return dataset, t_components
        else:
            return dataset, dataset

    def split_data(self, dataset, label):
        numpy_data = dataset.values
        numpy_label = label.values
        input_sequence = []
        target_sequence = []

        L = len(numpy_data)
        for i in range(0, L - self.test_len - self.label_len - self.pred_len):
            # 第一个[i:i+label_len, :]表示input_seq的长度和从numpy_data中复制的范围, 长度为i+label_len，复制所有特征
            start = i + self.label_len
            end = i + self.label_len + self.pred_len
            input_data = numpy_data[i:start, :]
            target_data = numpy_label[start:end, :]
            input_sequence.append(input_data)
            target_sequence.append(target_data)
        # 将列表转换为单个 numpy.ndarray
        train_data = np.array(input_sequence)
        train_label = np.array(target_sequence)
        input_sequence = []
        target_sequence = []
        # 测试集起始位置，测试集终止位置，预测步长
        for i in range(L - self.test_len - self.label_len, L - self.label_len - self.pred_len + 1, self.pred_len):
            start = i + self.label_len
            end = i + self.label_len + self.pred_len
            input_data = numpy_data[i:start, :]
            target_data = numpy_label[start:end, :]
            input_sequence.append(input_data)
            target_sequence.append(target_data)
        test_data = np.array(input_sequence)
        test_label = np.array(target_sequence)
        return [train_data, train_label, test_data, test_label]

    @staticmethod
    def seq_normal(dataset):
        row_mean = dataset.mean(axis=1)  # 获取行均值
        new_dataset = dataset.subtract(row_mean, axis=0)  # 将序列均值置零
        return new_dataset

    def restore(self, predict):
        predict = predict.reshape(-1, 1)  # (10, 30)->(300,1)
        result = self.scaler.inverse_transform(predict)
        result = pd.DataFrame(result, columns=['predict'])  # numpy->dataframe
        result.to_csv({self.checkpoints}+f'{self.target}-result.csv', index=True)
        return result


class StandardScaler(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = self.mean
        std = self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = self.mean
        std = self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
