import numpy as np
import pandas as pd
import seaborn as sns
from pylab import mpl
from vmdpy import VMD
import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis

# 风格设置: ['white', 'whitegrid', 'darkgrid', 'dark', 'ticks']
sns.set_style("white")
mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示中文字体
mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号


def deco_method(dataset, args):
    if args.Deco_model == 'VMD':
        t_u, _, _ = VMD(dataset.values, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7)
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 9))  # 绘制分解信号图
        axes[0] = sns.lineplot(dataset.values[:, 0], color='red', ax=axes[0])
        axes[0].set_xlabel('Original time series', fontsize=15)
        for i in range(3):
            axes[i + 1] = sns.lineplot(t_u[i, :], linewidth=1, ax=axes[i + 1])
            axes[i + 1].set_xlabel(f'IMF{i + 1}', fontsize=15)
        plt.tight_layout()
        plt.savefig(args.img_path + f'VMD-{args.target} decomposition results', dpi=600)
        plt.show()
        t_components = pd.DataFrame(t_u.T)  # numpy数组转换成dataframe格式
    elif args.Deco_model == 'SSA':
        window_size = 100
        groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]
        # Singular Spectrum Analysis
        ssa = SingularSpectrumAnalysis(window_size=window_size, groups=groups)
        sequence = dataset.values.reshape(1, -1)
        t_u = ssa.fit_transform(sequence)
        t_u = t_u.squeeze()
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 9))  # 绘制分解信号图
        axes[0] = sns.lineplot(sequence[0, :], color='red', ax=axes[0])
        axes[0].set_xlabel('Original time series', fontsize=15)
        for i in range(3):
            axes[i + 1] = sns.lineplot(t_u[i, :], linewidth=1, ax=axes[i + 1])
            axes[i + 1].set_xlabel(f'IMF{i + 1}', fontsize=15)
        plt.tight_layout()
        plt.savefig(args.img_path + f'SSA-{args.target} decomposition results', dpi=600)
        plt.show()
        t_components = pd.DataFrame(t_u.T)  # numpy数组转换成dataframe格式
    else:
        raise ValueError(f"Method {args.Deco_model} is not valid. It must be VMD or SSA.")
    return t_components
