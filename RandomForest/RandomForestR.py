import numpy as np
import pandas as pd
from time import time
from dtaidistance import dtw
from tslearn.metrics import lcss
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from RandomForest.DecisionTree import DecisionTreeRegressor
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


class RandomForestRegressor(object):
    """ Random Forest Regressor  """

    def __init__(self, TSCM, n_estimators, sample_sz, max_features=None, min_samples_leaf=1, min_samples_split=2,
                 n_jobs=None, max_depth=None, random_state=None):
        self._n_trees = n_estimators  # 森林中决策树的数量.
        self._sample_sz = sample_sz  # 每个决策树的训练样本数(0~1).
        self._max_features = max_features
        self._min_samples_leaf = min_samples_leaf  # 决策树中每个叶子节点的最小样本数.
        self._min_samples_split = min_samples_split  # 决策树中节点可分的最小样本数
        self._n_jobs = n_jobs  # 可以并行计算的线程数.
        self._max_depth = max_depth  # 每个决策树的最大深度.
        self._random_state = random_state
        self.tscm = TSCM
        self._trees = [self._create_tree() for i in range(self._n_trees)]

    # 为每个基本决策树生成训练数据
    def _get_sample_data(self, random_state, bootstrap=True):
        """
        参数
        ----------
        bootstrap: boolean value, True/False, 默认值为True,它将从输入训练数据中进行Bootstrap采样. 如果为False,将执行独占采样.

        返回
        -------
        idxs: array-like object, 返回输入训练数据中采样数据的索引.
        """
        num_samples = int(len(self._X) * self._sample_sz)  # 获取训练样本数.
        if bootstrap:
            np.random.seed(random_state)
            time_series_indices = np.random.choice(len(self._X), size=num_samples, replace=False)
            train_data = self._X.loc[time_series_indices]
            train_label = self._y.loc[time_series_indices]
            dist_mat = self.dist_mat[np.ix_(time_series_indices, time_series_indices)]
        else:
            train_data = self._X.head(num_samples)
            train_label = self._y.head(num_samples)
            dist_mat = self.dist_mat[:num_samples, :num_samples]
        if self._max_features:
            # 获取当前决策树训练特征
            sub_col_index = np.random.choice(self._X.columns.tolist(), self._max_features, replace=False)
            train_data = train_data.loc[:, sub_col_index]
        return [train_data, train_label, dist_mat]

    # 建立决策树
    def _create_tree(self):
        """
        返回
        -------
        DecisionTreeRegressor : 决策树对象.
        """
        return DecisionTreeRegressor(self._min_samples_leaf, self._min_samples_split, self._max_depth)

    # 拟合单个基本决策树
    def _single_tree_fit(self, tree, random_state):
        """
        参数
        ----------
        tree : 决策树对象.
        random_state : 随机数种子，用于确定随机采样

        返回
        -------
        tree : 决策树对象.
        """
        train_data, train_label, dist_mat = self._get_sample_data(random_state)  # 获取随机采样索引
        # new_sequences = self._seq_normal(train_label)
        # dist_mat = self._calc_dist_matrix(new_sequences.values)  # 计算样本标签之间的距离矩阵
        return tree.fit(train_data, train_label, dist_mat)

    # 从训练集(x,y)训练树木的森林回归器
    def fit(self, x, y):
        """
        参数
        ----------
        x : DataFrame, 训练输入样本.
        y : Series or array-like object, 目标值.
        """
        print(f'训练参数:n_trees={self._n_trees},n_jobs={self._n_jobs},max_features={self._max_features},'
              f'sample_sz={self._sample_sz},min_samples_leaf={self._min_samples_leaf},'
              f'min_samples_split={self._min_samples_split},max_depth={self._max_depth}')
        print(f'train_data shape:{x.shape}; train_label shape:{y.shape}')
        self._X = x
        self._y = y
        print('input-data:', self._X.shape, 'output-data:', self._y.shape)
        if self._random_state:
            np.random.seed(self._random_state)
        random_state_stages = np.random.choice(range(10 * self._n_trees), self._n_trees)  # 为每颗决策树生成随机数种子
        # 确定特征采样方式
        if self._max_features == 'sqrt':
            self._max_features = int(np.sqrt(self._X.shape[1]))
        elif self._max_features == 'log2':
            self._max_features = int(np.log2(self._X.shape[1]))
        # 计算序列相似性
        start = time()
        self.dist_mat = self._calc_dist_matrix(self._y.values)
        end = time()
        print(f'{self.tscm} similarity calc time: {(end - start):.6f} sec, {(end - start) / 60:.6f} min')
        if self._n_jobs:
            self._trees = self._parallel(self._trees, self._single_tree_fit, random_state_stages, self._n_jobs)
        else:
            for i, tree in enumerate(self._trees):
                self._single_tree_fit(tree, random_state_stages[i])

    def predict(self, x):
        """
        使用训练模型预测目标值
        参数
        ---------
        x : DataFrame or array-like object, 输入样本.

        返回
        -------
        ypreds : array-like object, 预测目标值.
        """
        all_tree_preds = np.stack([tree.predict(x) for tree in self._trees])
        return np.mean(all_tree_preds, axis=0)

    # 并行作业执行
    def _parallel(self, trees, fn, random_state_stages, n_jobs):
        """
        参数
        ----------
        trees : list-like object, 类似列表的对象包含所有决策树.
        fn : function-like object, 适应度函数.
        n_jobs : integer, 线程数.

        返回
        -------
        result : list-like object, 映射函数fn的每次调用都有一个类似列表的结果对象.
        """
        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 1
        if n_jobs > 0:
            workers = min(n_jobs, workers)
        print(f'训练参数:n_jobs={workers}')
        # 并行建立多棵决策树
        result = Parallel(workers, verbose=0, backend="threading")(
            delayed(fn)(trees[i], random_state) for i, random_state in enumerate(random_state_stages))
        return result

    def _seq_normal(self, sequences):
        row_mean = sequences.mean(axis=1)  # 获取行均值
        new_sequences = sequences.subtract(row_mean, axis=0)  # 将序列均值置零
        return new_sequences

    # 计算序列距离
    def _calc_dist_matrix(self, sequences):
        """
        参数
        ----------
        sequence:2-D numpy_array, 节点序列样本集合

        返回
        ----------
        dist_similarity:float, 序列内聚度
        cent_seq_index:int, 中心序列索引
        """
        if self.tscm == 'DTW':
            dist_mat = dtw.distance_matrix(sequences, parallel=True, use_c=True)
        elif self.tscm == 'LCSS':
            # 序列标准化
            scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
            # scaler = StandardScaler()
            sequences = scaler.fit_transform(sequences)
            sequences = sequences.squeeze()
            len_dataset = sequences.shape[0]
            # 初始化一个 len_dataset * len_dataset 的数组来存储相似度
            dist_mat = np.zeros((len_dataset, len_dataset))
            # 计算每对时间序列之间的 LCSS 相似度
            for i in range(len_dataset):
                for j in range(i, len_dataset):
                    sim = lcss(sequences[i], sequences[j], eps=0.05)
                    dist_mat[i, j] = 1 - sim
                    dist_mat[j, i] = 1 - sim  # 利用对称性减少计算量
        else:
            raise ValueError(f"Method {self.tscm} is not valid. It must be DTW or LCSS.")
        return dist_mat

    @property
    def feature_importances_(self):
        """
        计算特征重要性

        返回
        -------
        self._feature_importances : array-like object, 每个特征的重要性得分.
        """
        if not hasattr(self, '_feature_importances'):
            norm_imp = np.zeros(len(self._X.columns))
            for tree in self._trees:
                t_imp = tree.calc_feature_importance()
                norm_imp = norm_imp + t_imp / np.sum(t_imp)
            self._feature_importances = norm_imp / self._n_trees
        return self._feature_importances

    @property
    def feature_importances_extra(self):
        """
        另一种计算特征重要性的方法
        """
        norm_imp = np.zeros(len(self._X.columns))
        for tree in self._trees:
            t_imp = tree.calc_feature_importance_extra()
            norm_imp = norm_imp + t_imp / np.sum(t_imp)
        norm_imp = norm_imp / self._n_trees
        imp = pd.DataFrame({'col': self._X.columns, 'imp': norm_imp}).sort_values('imp', ascending=False)
        return imp
