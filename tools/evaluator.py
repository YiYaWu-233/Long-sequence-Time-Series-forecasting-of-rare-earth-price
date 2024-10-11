import os
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 风格设置: ['white', 'whitegrid', 'darkgrid', 'dark', 'ticks']
sns.set_style("white")
mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示中文字体
mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号


class evaluation(object):
    def __init__(self, args):
        self.model = args.model
        self.data = args.data
        self.data_path = args.data_path
        self.img_path = args.img_path
        self.result = args.result
        self.target = args.target

    @staticmethod
    def compute_loss(predict, target):  # 误差函数计算
        MAPE = np.mean(np.abs((predict - target) / target)) * 100  # 平均绝对百分比误差
        MSE = mean_squared_error(target, predict)  # 均方误差
        RMSE = np.sqrt(mean_squared_error(target, predict))
        MAE = mean_absolute_error(target, predict)  # 平均绝对误差
        R2 = r2_score(target, predict)
        return [MAPE, MSE, RMSE, MAE, R2]

    def access(self, predict, true):
        result_compare = pd.DataFrame()
        result_compare['true'] = true[self.target].copy()
        result_compare['predict'] = predict['predict'].copy()
        result_compare.to_csv(self.result + f'{self.model}-{self.target}.csv', index=False)
        pre_MAPE, pre_MSE, pre_RMSE, pre_MAE, pre_R2 = self.compute_loss(predict['predict'], true[self.target])
        print(f'predict  MAPE: {pre_MAPE:.2f}')
        print(f'predict  MSE: {pre_MSE:.2f}')
        print(f'predict  RMSE: {pre_RMSE:.2f}')
        print(f'predict  MAE: {pre_MAE:.2f}')
        print(f'predict  R2: {pre_R2:.2f}')

        plt.figure(figsize=(10, 6))
        g = sns.lineplot(data=true, x=true.index, y=true[self.target], label='True Values')
        g = sns.lineplot(data=predict, x=predict.index, y=predict['predict'], linestyle='dashdot',
                         label='Predict Values')
        g.set_ylabel('value', fontsize=20)
        g.set_xlabel('time', fontsize=20)
        plt.xticks(fontsize=8)  # x轴字体大小
        plt.yticks(fontsize=8)  # y轴字体大小
        g.set_title(f'{self.model}-{self.target}', fontsize=20)
        plt.savefig(self.img_path + f'{self.model}-{self.target}-预测结果与真实值的比较', dpi=600)
        plt.show()
