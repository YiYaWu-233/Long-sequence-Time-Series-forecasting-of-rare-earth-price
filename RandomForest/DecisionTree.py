import numpy as np
import pandas as pd
from dtaidistance import dtw
from sklearn.metrics import r2_score


class DecisionTreeRegressor(object):

    def __init__(self, min_samples_leaf, min_samples_split, max_depth=None):
        self.min_samples_leaf = min_samples_leaf  # 叶子节点含有的最少样本
        self.min_samples_split = min_samples_split  # 节点可分的最小样本数
        self._split_point = 0  # 当前节点的切分值.
        self._split_col_idx = 0  # 当前节点切分特征的索引.
        self._score = float('inf')  # 当前节点的不纯度度量(划分的质量指标),初始设置为正无穷大.
        self._sample_sz = 0  # 当前节点的样本数量.
        self._left_child_tree = None
        self._right_child_tree = None
        self._feature_importances = []  # 特征重要性列表.
        self._node_importance = 0  # 节点的重要性度量.
        if max_depth is not None:
            max_depth = max_depth - 1
        self._max_depth = max_depth

    # 用输入数据 x 和目标值 y 训练决策树模型
    def fit(self, x, y, dist_mat):
        self._X = x
        self._y = y
        self._X.reset_index(drop=True, inplace=True)  # 重置索引
        self._y.reset_index(drop=True, inplace=True)
        self._col_names = self._X.columns
        self._feature_importances = np.zeros(96)
        self._sample_sz = len(self._X)  # 样本数量
        _, cent_seq_index = self._calc_dist_similarity(dist_mat)  # 获取中心序列索引
        self._val = self._y.iloc[cent_seq_index]
        # acceleration_weight = int(self._sample_sz * 0.2)  # 加速权重，用来调整min_samples_leaf
        if self._sample_sz < self.min_samples_split:
            return self
        if self._max_depth is not None and self._max_depth < 2:
            return self
        if len(self._col_names) < 2:
            return self
        self._find_best_split(dist_mat)
        return self

    # 递归计算序列不纯度
    def _calc_dist_impurity(self, sequences):
        """
        参数
        ----------
        sequence:2-D numpy_array, 节点序列样本集合

        返回
        ----------
        dist_similarity:float, 序列内聚度
        cent_seq_index:int, 中心序列索引
        """
        mean_seq = np.mean(sequences, axis=0)
        sequences = np.vstack((mean_seq, sequences))
        dtw_distances = dtw.distance_matrix(sequences, parallel=True, use_c=True, block=((0, 1), (0, len(sequences))),
                                            only_triu=True, compact=True)
        # dist_mat = dtw.distance_matrix(sequence, parallel=True, use_c=True)
        dist_impurity = np.sum(dtw_distances, axis=0) / len(dtw_distances)
        return dist_impurity

    # 递归计算序列相似度
    def _calc_dist_similarity(self, dist_mat):
        """
        参数
        ----------
        sequence:2-D numpy_array, 节点序列样本集合

        返回
        ----------
        dist_similarity:float, 序列内聚度
        cent_seq_index:int, 中心序列索引
        """
        sample_cohesion = np.sum(dist_mat, axis=1)  # 计算每一行的和,
        cent_seq_index = np.argmin(sample_cohesion)  # 行和最小所在即为中心序列
        dist_similarity = sample_cohesion[cent_seq_index] / len(sample_cohesion)
        return dist_similarity, cent_seq_index

    # 寻找当前节点的最佳特征，用于树的构建
    def _find_best_split(self, dist_mat):
        for col_idx in self._col_names:
            self._find_col_best_split_point(col_idx, dist_mat)  # 寻找col_idx的最佳切分值
        # self._split_col_idx 记录了当前节点使用的最佳特征的索引
        self._feature_importances[self._split_col_idx - 1] = self._node_importance

        if self.is_leaf:
            return

        left_child_sample_idxs = np.nonzero(self.split_col <= self.split_point)[0]
        right_child_sample_idxs = np.nonzero(self.split_col > self.split_point)[0]
        _X = self._X.drop(self._split_col_idx, axis=1)
        # 左右子树样本划分
        sample_x_l = _X.iloc[left_child_sample_idxs]
        sample_x_r = _X.iloc[right_child_sample_idxs]
        sample_y_l = self._y.iloc[left_child_sample_idxs]
        sample_y_r = self._y.iloc[right_child_sample_idxs]
        dist_mat_l = dist_mat[np.ix_(left_child_sample_idxs, left_child_sample_idxs)]
        dist_mat_r = dist_mat[np.ix_(right_child_sample_idxs, right_child_sample_idxs)]
        self._left_child_tree = (
            DecisionTreeRegressor(self.min_samples_leaf, self.min_samples_split, self._max_depth).fit(
                sample_x_l, sample_y_l, dist_mat_l))
        self._right_child_tree = (
            DecisionTreeRegressor(self.min_samples_leaf, self.min_samples_split, self._max_depth).fit(
                sample_x_r, sample_y_r, dist_mat_r))

    # 寻找分裂特征的最佳切分值，用于树的构建
    def _find_col_best_split_point(self, col_idx, dist_mat):
        x_col = self._X[col_idx].values  # dataFrame->numpy, 取出col_idx列
        sorted_idxs = np.argsort(x_col)  # 返回x_col中从小到大元素索引
        sorted_x_col = x_col[sorted_idxs]  # 将x_col从大到小排序
        sorted_y = self._y.values[sorted_idxs]  # 将self._y也变更为对应顺序
        sorted_dist_mat = dist_mat[sorted_idxs]  # 将dist_mat也变更为对应顺序

        for i in range(self.min_samples_leaf, self._sample_sz - self.min_samples_leaf):
            xi, yi = sorted_x_col[i], sorted_y[i]  # 获取特征值x和目标值y
            # 如果当前特征值与下一个特征值相同，则跳过当前特征值
            if xi == sorted_x_col[i + 1]:
                continue
            lchild_samples = i  # 左子节点的样本数量
            rchild_samples = self._sample_sz - i  # 右子节点的样本数量
            # 如果父节点分裂后，左右子树样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if lchild_samples < self.min_samples_leaf or rchild_samples < self.min_samples_leaf:
                continue

            lchild_impurity, _ = self._calc_dist_similarity(sorted_dist_mat[0:lchild_samples, 0:lchild_samples])
            rchild_impurity, _ = self._calc_dist_similarity((sorted_dist_mat[-rchild_samples:, -rchild_samples:]))
            split_score = (lchild_samples * lchild_impurity + rchild_samples * rchild_impurity) / self._sample_sz

            if split_score < self._score:
                self._score = split_score
                self._split_point = xi  # 记录更优切分值xi
                self._split_col_idx = col_idx  # 记录更优切分特征索引col_idx
                self._node_importance = (
                        self._sample_sz * (self._calc_dist_impurity(sorted_y) - split_score))

    def predict(self, x):
        if type(x) == pd.DataFrame:
            x = x.values
        return np.array([self._predict_row(row) for row in x])

    def _predict_row(self, row):
        if self.is_leaf:
            step = row[-1] - self._val.iloc[0]
            result = self._val + step
            return result
        tree = (self._left_child_tree if row[self._split_col_idx - 1] <= self.split_point else self._right_child_tree)
        return tree._predict_row(row)

    def __repr__(self):
        pr = f'sample: {self._sample_sz}, value: {self._val}\r\n'
        if not self.is_leaf:
            pr += f'split column: {self.split_name}, \
                split point: {self.split_point}, score: {self._score} '
        return pr

    def calc_feature_importance(self):
        if self.is_leaf:
            return self._feature_importances
        return (self._feature_importances
                + self._left_child_tree.calc_feature_importance()
                + self._right_child_tree.calc_feature_importance()
                )

    #  通过计算 R^2 分数的减小来计算特征重要性
    def calc_feature_importance_extra(self):
        imp = []
        o_preds = self.predict(self._X.values)
        o_r2 = r2_score(self._y, o_preds)
        for col in self._col_names:
            tmp_x = self._X.copy()
            shuffle_col = tmp_x[col].values
            np.random.shuffle(shuffle_col)
            tmp_x.loc[:, col] = shuffle_col
            tmp_preds = self.predict(tmp_x.values)
            tmp_r2 = r2_score(self._y, tmp_preds)
            imp.append((o_r2 - tmp_r2))
        imp = imp / np.sum(imp)
        return imp

    @property
    def split_name(self):
        return self._split_col_idx

    @property
    def split_col(self):
        return self._X[self._split_col_idx]

    @property
    def is_leaf(self):
        return self._score == float('inf')

    @property
    def split_point(self):
        return self._split_point
