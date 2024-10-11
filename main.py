import pickle
import argparse
import numpy as np
import pandas as pd
from time import time

from RandomForest.RandomForestR import RandomForestRegressor
from tools.dataloader import data_load
from tools.evaluator import evaluation

parser = argparse.ArgumentParser(description='[SRF] Long sequence Time-Series Forecasting')
# 选择模型 ['SRF', 'VMD-SRF', 'SSA-SRF']
parser.add_argument('--model', type=str, default='VMD-SRF')
# 数据上级目录
parser.add_argument('--data_path', type=str, default='./data/')
# 数据名称['稀土价格', 'ECL', 'WTH', 'ETTh1']
parser.add_argument('--data', type=str, default='稀土价格')
# 文件名称
parser.add_argument('--data_name', type=str, default='稀土价格.csv')
# 数据中要预测的标签列 ['氧化镧平均价', '氧化铈平均价', '氧化镨平均价', '氧化钕平均价', 'MT_320', 'WetBulbCelsius', 'OT']
parser.add_argument('--target', type=str, default='氧化镨平均价')
# 数据上级目录
parser.add_argument('--img_path', type=str, default='./img/')
# 模型保存位置
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
# 结果保存位置
parser.add_argument('--result', type=str, default='./result/')
# 是否进行频谱分解['VMD', 'SSA']
parser.add_argument('--Deco_model', type=str, default='VMD')
# 时间序列相似度计算方法['DTW', 'LCSS']
parser.add_argument('--sim_method', type=str, default='DTW')
# 数据输入维度
parser.add_argument('--input_dim', type=int, default=1)
# 数据输出维度
parser.add_argument('--output_dim', type=int, default=1)
# 先验序列长度
parser.add_argument('--label_len', type=int, default=30)
# 预测序列长度
parser.add_argument('--pred_len', type=int, default=30)
# 测试集长度
parser.add_argument('--test_len', type=int, default=300)
# 是否标准化
parser.add_argument('--scale', action='store_true', default=False)
# 取参数值
args = parser.parse_args()


def train(input_data, output_data):
    multi_model = []
    for col in range(output_data.shape[2]):
        data = input_data[:, :, col]  # 获取当前预测对象的序列数据
        target = output_data[:, :, col]  # 获取当前预测对象的辅助特征

        data = pd.DataFrame(data, columns=range(1, args.label_len + 1))
        target = pd.DataFrame(target, columns=range(1, args.pred_len + 1))

        forest = RandomForestRegressor(TSCM=args.sim_method, n_estimators=20, sample_sz=0.8, min_samples_leaf=10,
                                       min_samples_split=20, n_jobs=20, max_depth=9, random_state=2023)
        start = time()
        forest.fit(data, target)
        end = time()
        print(f'calc time: {(end - start):.6f} sec, {(end - start) / 60:.6f} min')
        multi_model.append(forest)
    return multi_model


def forcast(multi_model, input_data):
    # 维度转换三维->二维
    col_pred = []
    for col in range(len(multi_model)):
        data_seq = input_data[:, :, col]
        data_seq = pd.DataFrame(data_seq.reshape(-1, args.pred_len), columns=range(1, args.pred_len + 1))
        model = multi_model[col]  # 从字典中取出模型
        predict = model.predict(data_seq)
        col_pred.append(predict)  # 将预测结果放进列表中
    pred_seq = np.array(col_pred)
    pred_seq = np.squeeze(pred_seq)  # 数组中移除维度为1的条目
    return pred_seq


def main(args):
    print(f'预测对象: {args.target}')
    dl = data_load(args)
    data_set, pd_features = dl.get_data()
    date_ready, t_components = dl.data_processing(pd_features)
    train_data, train_label, test_data, test_label = dl.split_data(t_components, t_components)

    start = time()
    multi_model = train(train_data, train_label)
    end = time()
    with open(args.checkpoints + f'{args.target}预测模型.pkl', 'wb') as f:
        pickle.dump(multi_model, f)  # 保存模型
    print(f'calc time: {(end - start):.6f} sec, {(end - start) / 60:.6f} min')
    predict = forcast(multi_model, test_data)
    if args.Deco_model:
        predict = np.sum(predict, axis=0)  # 在第一个维度求和，将频率分量还原为原始信号
    result = predict.reshape(-1, 1)  # (7, 30)->(300,30)
    result = pd.DataFrame(result, columns=['predict'])  # numpy->dataframe
    true_file = date_ready.tail(args.test_len)
    true_file = true_file.reset_index(drop=True)
    evo = evaluation(args)
    evo.access(result, true_file)


if __name__ == "__main__":
    main(args)
