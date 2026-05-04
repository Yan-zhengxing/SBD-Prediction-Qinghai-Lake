"""
空间交叉验证模块 - 包含多种方法
功能：实现自适应网格分块、K-means分块、环境分层分块以及DEM分层分块
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import os
import warnings
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# 导入配置
try:
    import config

    COORD_COLS = config.COORD_COLS if hasattr(config, 'COORD_COLS') else ['POINT_X', 'POINT_Y']
except ImportError:
    COORD_COLS = ['POINT_X', 'POINT_Y']


class DemStratifiedSpatialCV:
    """DEM分层+K-means空间交叉验证类"""

    def __init__(self, n_strata=5, min_samples_per_block=10,
                 test_blocks_per_stratum=1, random_state=42):
        """
        初始化DEM分层空间CV

        Parameters:
        -----------
        n_strata : int
            DEM分层数量，默认为5层
        min_samples_per_block : int
            每个块的最小样本数
        test_blocks_per_stratum : int
            每层中作为测试集的块数（1或2）
        random_state : int
            随机种子
        """
        self.n_strata = n_strata
        self.min_samples_per_block = min_samples_per_block
        self.test_blocks_per_stratum = test_blocks_per_stratum
        self.random_state = random_state
        np.random.seed(random_state)

    def extract_coordinates(self, coords_df):
        """
        提取坐标数据（兼容多种列名格式）
        """
        # 检查是否是DataFrame
        if not isinstance(coords_df, pd.DataFrame):
            # 如果是数组或Series，返回前两列作为坐标
            if hasattr(coords_df, 'shape'):
                if len(coords_df.shape) == 1:
                    # 一维数组，可能是Series
                    print("  警告: 坐标数据是一维的，尝试转换为二维")
                    coords_df = pd.DataFrame({'coord': coords_df})
                # 如果是二维数组，取前两列
                if len(coords_df.shape) >= 2 and coords_df.shape[1] >= 2:
                    return coords_df[:, 0], coords_df[:, 1], ('x', 'y')
                else:
                    print("  警告: 坐标数据维度不足，使用索引")
                    return (np.arange(len(coords_df)),
                            np.arange(len(coords_df)),
                            ('x_index', 'y_index'))

        # 如果是DataFrame，首先尝试使用全局配置的坐标列
        if COORD_COLS[0] in coords_df.columns and COORD_COLS[1] in coords_df.columns:
            return (coords_df[COORD_COLS[0]].values,
                    coords_df[COORD_COLS[1]].values,
                    tuple(COORD_COLS))

        # 尝试多种可能的坐标列名
        coord_patterns = [
            ('POINT_X', 'POINT_Y'),
            ('x_coord', 'y_coord'),
            ('x', 'y'),
            ('Longitude', 'Latitude'),
            ('lon', 'lat'),
            ('X', 'Y'),
            ('X_COORD', 'Y_COORD')
        ]

        for x_col, y_col in coord_patterns:
            if x_col in coords_df.columns and y_col in coords_df.columns:
                return coords_df[x_col].values, coords_df[y_col].values, (x_col, y_col)

        # 如果找不到标准列名，尝试查找包含关键字的列
        x_cols = [col for col in coords_df.columns
                  if any(keyword in col.lower() for keyword in ['x', 'lon', 'long'])]
        y_cols = [col for col in coords_df.columns
                  if any(keyword in col.lower() for keyword in ['y', 'lat'])]

        if x_cols and y_cols:
            return (coords_df[x_cols[0]].values,
                    coords_df[y_cols[0]].values,
                    (x_cols[0], y_cols[0]))

        # 如果还是找不到，检查列数
        if coords_df.shape[1] >= 2:
            # 使用前两列作为坐标
            print(f"  警告: 未找到标准坐标列，使用前两列作为坐标: {coords_df.columns[0]}, {coords_df.columns[1]}")
            return (coords_df.iloc[:, 0].values,
                    coords_df.iloc[:, 1].values,
                    (coords_df.columns[0], coords_df.columns[1]))

        # 最后尝试，使用索引作为坐标
        print("⚠️  警告: 未找到坐标列，使用索引作为坐标")
        return (np.arange(len(coords_df)),
                np.arange(len(coords_df)),
                ('x_index', 'y_index'))

    def dem_stratified_kmeans_blocking(self, coords_df, dem_values):
        """
        DEM分层+K-means空间分块方法

        Parameters:
        -----------
        coords_df : DataFrame
            包含坐标的数据框
        dem_values : array
            DEM值数组

        Returns:
        --------
        block_labels : array
            块标签数组
        block_info : dict
            块信息
        """
        print("=" * 60)
        print("执行DEM分层+K-means空间分块")
        print("=" * 60)

        # 提取坐标
        x_coords, y_coords, coord_cols = self.extract_coordinates(coords_df)
        n_samples = len(x_coords)

        print(f"  使用的坐标列: {coord_cols}")
        print(f"  总样本数: {n_samples}")
        print(f"  DEM值范围: {dem_values.min():.1f} - {dem_values.max():.1f}")

        # 1. 按DEM值分层（使用分位数分层）
        print(f"\n  步骤1: 按DEM值分{self.n_strata}层")

        # 计算分位数
        strata_bins = np.percentile(dem_values,
                                    np.linspace(0, 100, self.n_strata + 1))

        # 确保边界值唯一
        strata_bins = np.unique(strata_bins)
        n_strata_actual = len(strata_bins) - 1

        if n_strata_actual < self.n_strata:
            print(f"  ⚠️  注意: 由于DEM值重复，实际分为 {n_strata_actual} 层")
            self.n_strata = n_strata_actual

        # 分配层标签
        stratum_labels = np.digitize(dem_values, strata_bins[1:-1])

        # 统计每层样本数
        stratum_stats = []
        for i in range(self.n_strata):
            mask = stratum_labels == i
            n_samples_in_stratum = mask.sum()
            dem_min = dem_values[mask].min() if n_samples_in_stratum > 0 else np.nan
            dem_max = dem_values[mask].max() if n_samples_in_stratum > 0 else np.nan
            dem_mean = dem_values[mask].mean() if n_samples_in_stratum > 0 else np.nan

            stratum_stats.append({
                'stratum': i + 1,
                'n_samples': n_samples_in_stratum,
                'dem_range': f"{dem_min:.1f}-{dem_max:.1f}",
                'dem_mean': dem_mean
            })

        print("  各层统计信息:")
        for stat in stratum_stats:
            print(f"    第{stat['stratum']}层: {stat['n_samples']}个样本, "
                  f"DEM范围: {stat['dem_range']}, 平均DEM: {stat['dem_mean']:.1f}")

        # 2. 每层内进行K-means空间分块
        print(f"\n  步骤2: 每层内进行K-means空间聚类分块")

        # 初始化块标签数组
        block_labels = np.full(n_samples, -1, dtype=int)
        block_counter = 0

        # 存储每层的块信息
        stratum_block_info = []

        for i in range(self.n_strata):
            mask = stratum_labels == i
            n_samples_in_stratum = mask.sum()

            if n_samples_in_stratum < self.min_samples_per_block * 2:
                print(f"    第{i + 1}层: {n_samples_in_stratum}个样本过少，不进行分块")
                block_labels[mask] = block_counter
                stratum_block_info.append({
                    'stratum': i + 1,
                    'n_blocks': 1,
                    'block_range': f"{block_counter}",
                    'samples_per_block': [n_samples_in_stratum]
                })
                block_counter += 1
                continue

            # 提取该层的坐标
            stratum_coords = np.column_stack([x_coords[mask], y_coords[mask]])

            # 确定该层的K-means聚类数
            # 规则：至少2个块，最多不超过层内样本数/最小样本数
            max_blocks = max(2, n_samples_in_stratum // self.min_samples_per_block)
            # 对于稀疏样本，限制最大块数
            max_blocks = min(6, max_blocks)

            # 根据样本数动态确定块数
            if n_samples_in_stratum < 30:
                n_clusters = 2
            elif n_samples_in_stratum < 60:
                n_clusters = 3
            elif n_samples_in_stratum < 100:
                n_clusters = 4
            else:
                n_clusters = min(5, max_blocks)

            print(f"    第{i + 1}层: {n_samples_in_stratum}个样本，分为{n_clusters}个空间块")

            # 标准化坐标
            scaler = StandardScaler()
            stratum_coords_scaled = scaler.fit_transform(stratum_coords)

            # K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            stratum_block_ids = kmeans.fit_predict(stratum_coords_scaled)

            # 统计每个块的样本数
            unique_blocks, counts = np.unique(stratum_block_ids, return_counts=True)
            block_counts = counts.tolist()

            print(f"      块大小: {min(block_counts)}-{max(block_counts)}个样本")

            # 分配全局块标签
            for j, local_block_id in enumerate(stratum_block_ids):
                idx = np.where(mask)[0][j]
                block_labels[idx] = block_counter + local_block_id

            stratum_block_info.append({
                'stratum': i + 1,
                'n_blocks': n_clusters,
                'block_range': f"{block_counter}-{block_counter + n_clusters - 1}",
                'samples_per_block': block_counts
            })

            block_counter += n_clusters

        # 3. 验证块标签
        unique_blocks, counts = np.unique(block_labels[block_labels >= 0], return_counts=True)
        n_blocks_total = len(unique_blocks)

        print(f"\n  步骤3: 分块结果统计")
        print(f"    总块数: {n_blocks_total}")
        print(f"    每块样本数范围: {min(counts)}-{max(counts)}")
        print(f"    平均每块样本数: {np.mean(counts):.1f}")

        # 检查是否有样本未分配块
        unassigned = np.sum(block_labels < 0)
        if unassigned > 0:
            print(f"  ⚠️  警告: 有{unassigned}个样本未分配到任何块")
            # 将未分配的样本分配到最近的块
            block_labels = self.assign_unassigned_samples(x_coords, y_coords, block_labels)

        # 4. 准备块信息
        block_info = {
            'n_strata': self.n_strata,
            'n_blocks': n_blocks_total,
            'min_samples': min(counts),
            'max_samples': max(counts),
            'mean_samples': np.mean(counts),
            'method': 'dem_stratified_kmeans',
            'test_blocks_per_stratum': self.test_blocks_per_stratum,
            'stratum_stats': stratum_stats,
            'stratum_block_info': stratum_block_info,
            'coord_cols': coord_cols
        }

        return block_labels, block_info

    def assign_unassigned_samples(self, x_coords, y_coords, block_labels):
        """
        将未分配的样本分配到最近的块
        """
        unassigned_mask = block_labels < 0
        assigned_mask = block_labels >= 0

        if not np.any(unassigned_mask):
            return block_labels

        unassigned_indices = np.where(unassigned_mask)[0]
        assigned_indices = np.where(assigned_mask)[0]

        coords_unassigned = np.column_stack([x_coords[unassigned_mask],
                                             y_coords[unassigned_mask]])
        coords_assigned = np.column_stack([x_coords[assigned_mask],
                                           y_coords[assigned_mask]])
        labels_assigned = block_labels[assigned_mask]

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords_assigned)
        distances, indices = nbrs.kneighbors(coords_unassigned)

        for i, idx in enumerate(unassigned_indices):
            nearest_idx = assigned_indices[indices[i][0]]
            block_labels[idx] = block_labels[nearest_idx]

        print(f"    已将{len(unassigned_indices)}个未分配样本分配到最近的块")
        return block_labels

    def create_cv_folds(self, block_labels, dem_values, n_splits=5):
        """
        创建交叉验证折

        Parameters:
        -----------
        block_labels : array
            块标签数组
        dem_values : array
            DEM值数组
        n_splits : int
            折数

        Returns:
        --------
        folds : list
            每折的(训练索引, 测试索引)列表
        fold_info : list
            每折的信息
        """
        print(f"\n  步骤4: 创建交叉验证折 (目标: {n_splits}折)")

        # 获取唯一的块标签
        unique_blocks = np.unique(block_labels)
        n_blocks = len(unique_blocks)

        # 将块按DEM值排序（按块内平均DEM）
        block_dem_means = []
        for block in unique_blocks:
            mask = block_labels == block
            block_dem_means.append(dem_values[mask].mean())

        # 按DEM平均值排序
        sorted_blocks = [block for _, block in sorted(zip(block_dem_means, unique_blocks))]

        # 对于DEM分层方法，我们希望每折的测试集包含各层的样本
        # 策略：每折从每层中抽取1-2个块作为测试集

        # 首先确定每层有哪些块
        stratum_blocks = {}
        for i in range(self.n_strata):
            # 获取该层的样本索引
            stratum_mask = self.get_stratum_mask(dem_values, i)
            # 获取该层的块
            blocks_in_stratum = np.unique(block_labels[stratum_mask])
            blocks_in_stratum = blocks_in_stratum[blocks_in_stratum >= 0]  # 排除-1
            stratum_blocks[i] = blocks_in_stratum.tolist()

        print(f"  各层块分布:")
        for i in range(self.n_strata):
            if i in stratum_blocks and stratum_blocks[i]:
                print(f"    第{i + 1}层: {len(stratum_blocks[i])}个块 - {stratum_blocks[i]}")
            else:
                print(f"    第{i + 1}层: 0个块")

        # 确定每层要抽取的测试块数
        test_blocks_per_stratum = self.test_blocks_per_stratum

        # 计算最大可能的折数
        max_folds = min([len(blocks) // test_blocks_per_stratum
                         for blocks in stratum_blocks.values() if blocks])

        if max_folds < n_splits:
            print(f"  ⚠️  警告: 最大可能折数为{max_folds}，将n_splits调整为{max_folds}")
            n_splits = max_folds

        if n_splits < 2:
            print(f"  ❌ 错误: 无法创建至少2折交叉验证")
            return [], []

        print(f"  最终折数: {n_splits}")

        # 为每层准备测试块序列
        test_block_sequences = {}
        for i in range(self.n_strata):
            if i in stratum_blocks and stratum_blocks[i]:
                blocks = stratum_blocks[i]
                # 随机打乱块顺序
                np.random.shuffle(blocks)

                # 将块分组，每组test_blocks_per_stratum个
                n_groups = len(blocks) // test_blocks_per_stratum
                groups = []
                for j in range(n_groups):
                    start = j * test_blocks_per_stratum
                    end = start + test_blocks_per_stratum
                    if end <= len(blocks):
                        groups.append(blocks[start:end])

                # 如果组数不够，用循环方式填充
                while len(groups) < n_splits:
                    groups.append(groups[len(groups) % len(groups)])

                test_block_sequences[i] = groups[:n_splits]

        # 创建折
        folds = []
        fold_info = []

        for fold in range(n_splits):
            test_blocks = []

            # 从每层收集测试块
            for i in range(self.n_strata):
                if i in test_block_sequences:
                    test_blocks.extend(test_block_sequences[i][fold])

            # 创建测试掩码
            test_mask = np.isin(block_labels, test_blocks)
            train_mask = ~test_mask

            # 检查是否有样本
            if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
                print(f"  ⚠️  警告: 第{fold + 1}折训练集或测试集为空，跳过")
                continue

            folds.append((train_mask, test_mask))

            # 收集折信息
            fold_info.append({
                'fold': fold + 1,
                'test_blocks': test_blocks,
                'n_train': np.sum(train_mask),
                'n_test': np.sum(test_mask),
                'train_ratio': np.sum(train_mask) / len(block_labels)
            })

            print(f"    第{fold + 1}折: 训练集{np.sum(train_mask)}样本, "
                  f"测试集{np.sum(test_mask)}样本, "
                  f"{len(test_blocks)}个测试块")

        return folds, fold_info

    def get_stratum_mask(self, dem_values, stratum_idx):
        """
        获取指定层的掩码
        """
        # 计算DEM分位数
        strata_bins = np.percentile(dem_values,
                                    np.linspace(0, 100, self.n_strata + 1))
        strata_bins = np.unique(strata_bins)

        if stratum_idx == 0:
            mask = dem_values <= strata_bins[1]
        elif stratum_idx == self.n_strata - 1:
            mask = dem_values >= strata_bins[-2]
        else:
            mask = (dem_values > strata_bins[stratum_idx]) & (dem_values <= strata_bins[stratum_idx + 1])

        return mask

    def get_stratum_labels(self, dem_values):
        """获取每层的标签"""
        strata_bins = np.percentile(dem_values, np.linspace(0, 100, self.n_strata + 1))
        strata_bins = np.unique(strata_bins)
        return np.digitize(dem_values, strata_bins[1:-1])


def dem_stratified_spatial_cv_with_predictions(model, X, y, coords_df, dem_values,
                                               n_strata=5, n_splits=5,
                                               verbose=True, output_dir=None):
    """
    DEM分层空间交叉验证

    参数:
        model: 模型对象
        X: 特征数据
        y: 目标数据
        coords_df: 坐标数据框
        dem_values: DEM值数组
        n_strata: DEM分层数
        n_splits: 交叉验证折数
        verbose: 是否显示详细信息
        output_dir: 输出目录

    返回:
        交叉验证结果和预测值
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"执行DEM分层空间交叉验证")
        print(f"  分层数: {n_strata}, 折数: {n_splits}")
        print(f"{'=' * 60}")

    try:
        # 1. 创建DEM分层空间块
        spatial_cv = DemStratifiedSpatialCV(
            n_strata=n_strata,
            min_samples_per_block=10,
            test_blocks_per_stratum=1,
            random_state=42
        )

        block_labels, block_info = spatial_cv.dem_stratified_kmeans_blocking(
            coords_df, dem_values
        )

        # 2. 准备坐标数据
        x_coords, y_coords, coord_cols = spatial_cv.extract_coordinates(coords_df)

        # 3. 创建交叉验证折
        folds, fold_info = spatial_cv.create_cv_folds(block_labels, dem_values, n_splits)

        if not folds:
            print("  ❌ 无法创建有效的交叉验证折")
            return None, None, None

        # 4. 进行交叉验证
        print(f"\n  步骤5: 进行交叉验证")

        # 初始化结果存储
        y_true_all = []
        y_pred_all = []
        fold_labels_all = []
        x_coords_all = []
        y_coords_all = []
        block_ids_all = []

        fold_results = []

        for fold_idx, (train_mask, test_mask) in enumerate(folds):
            # 检查样本数
            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train == 0 or n_test == 0:
                if verbose:
                    print(f"  Fold {fold_idx + 1}: 训练集或测试集为空，跳过")
                continue

            if verbose:
                print(f"  Fold {fold_idx + 1}: 训练集 {n_train} 样本, "
                      f"测试集 {n_test} 样本")

            # 提取数据
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_mask], X.iloc[test_mask]
                y_train, y_test = y.iloc[train_mask], y.iloc[test_mask]
            else:
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]

            try:
                # 克隆模型
                if hasattr(model, '__class__') and model.__class__.__name__ == 'CubistWrapper':
                    # CubistWrapper不能直接clone，需要特殊处理
                    try:
                        import copy
                        model_copy = copy.deepcopy(model)
                    except:
                        print(f"  ⚠️ 警告: 无法克隆{model.__class__.__name__}，跳过当前fold")
                        continue
                else:
                    # 标准sklearn模型，使用clone
                    from sklearn.base import clone
                    try:
                        model_copy = clone(model)
                    except Exception as e:
                        print(f"  ⚠️ 警告: 克隆模型失败，尝试深拷贝: {e}")
                        import copy
                        model_copy = copy.deepcopy(model)

                # 训练模型
                model_copy.fit(X_train, y_train)

                # 预测
                y_pred = model_copy.predict(X_test)

                # 收集结果
                y_true_all.extend(y_test if hasattr(y_test, 'values') else y_test)
                y_pred_all.extend(y_pred)
                fold_labels_all.extend([fold_idx] * len(y_test))
                x_coords_all.extend(x_coords[test_mask])
                y_coords_all.extend(y_coords[test_mask])
                block_ids_all.extend(block_labels[test_mask])

                # 计算指标
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                fold_results.append({
                    'fold': fold_idx + 1,
                    'train_samples': n_train,
                    'test_samples': n_test,
                    'test_blocks': len(np.unique(block_labels[test_mask])),
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                })

                if verbose:
                    print(f"    R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

            except Exception as e:
                if verbose:
                    print(f"  Fold {fold_idx + 1} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                continue

        if not y_true_all:
            if verbose:
                print("  所有fold都失败，空间交叉验证无法完成")
            return None, None, None

        # 计算整体指标
        overall_r2 = r2_score(y_true_all, y_pred_all)
        overall_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        overall_mae = mean_absolute_error(y_true_all, y_pred_all)

        if verbose:
            print(f"\n  整体结果: R²={overall_r2:.3f}, "
                  f"RMSE={overall_rmse:.3f}, MAE={overall_mae:.3f}")

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'actual_value': y_true_all,
            'predicted_value': y_pred_all,
            'fold': fold_labels_all,
            'block_id': block_ids_all,
            COORD_COLS[0]: x_coords_all,
            COORD_COLS[1]: y_coords_all
        })

        # 添加通用的坐标列名
        results_df['x_coord'] = x_coords_all
        results_df['y_coord'] = y_coords_all

        # 如果原始坐标列名与标准不同，也添加它们
        if coord_cols[0] != COORD_COLS[0] or coord_cols[1] != COORD_COLS[1]:
            results_df[coord_cols[0]] = x_coords_all
            results_df[coord_cols[1]] = y_coords_all

        # 添加DEM值（如果可用）
        if dem_values is not None:
            # 获取测试样本的索引
            test_indices = []
            for _, test_mask in folds:
                test_indices.extend(np.where(test_mask)[0])
            results_df['DEM'] = dem_values[test_indices]

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # 保存预测结果
            output_file = os.path.join(output_dir, 'dem_stratified_cv_predictions.csv')
            results_df.to_csv(output_file, index=False)

            # 保存fold结果
            fold_results_df = pd.DataFrame(fold_results)
            fold_results_file = os.path.join(output_dir, 'dem_stratified_cv_fold_results.csv')
            fold_results_df.to_csv(fold_results_file, index=False)

            # 保存块信息
            block_info_file = os.path.join(output_dir, 'dem_stratified_cv_block_info.pkl')
            with open(block_info_file, 'wb') as f:
                pickle.dump(block_info, f)

            # 保存折信息
            fold_info_df = pd.DataFrame(fold_info)
            fold_info_file = os.path.join(output_dir, 'dem_stratified_cv_fold_info.csv')
            fold_info_df.to_csv(fold_info_file, index=False)

            # 可视化空间分块
            visualize_dem_stratification(
                pd.DataFrame({
                    'x': x_coords,
                    'y': y_coords,
                    'dem': dem_values,
                    'stratum': spatial_cv.get_stratum_labels(dem_values),
                    'block': block_labels
                }),
                f'{model.__class__.__name__}_dem_stratified',
                output_dir,
                show_plot=False
            )

            if verbose:
                print(f"\n  空间CV预测结果已保存到: {output_file}")
                print(f"  Fold结果已保存到: {fold_results_file}")
                print(f"  块信息已保存到: {block_info_file}")

        # 返回结果
        cv_results = {
            'method': 'dem_stratified',
            'overall_r2': overall_r2,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'fold_results': fold_results,
            'n_folds_completed': len(fold_results),
            'n_blocks': block_info['n_blocks'],
            'n_strata': n_strata,
            'block_info': block_info,
            'coord_cols': COORD_COLS
        }

        return cv_results, results_df, block_labels

    except Exception as e:
        if verbose:
            print(f"  ❌ DEM分层空间交叉验证失败: {e}")
            import traceback
            traceback.print_exc()
        return None, None, None


def visualize_dem_stratification(data_df, title, output_dir=None, show_plot=False):
    """
    可视化DEM分层结果

    参数:
        data_df: 包含坐标、DEM、层标签和块标签的数据框
        title: 图表标题
        output_dir: 输出目录
        show_plot: 是否显示图表
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. DEM值分布
        ax1 = axes[0, 0]
        ax1.hist(data_df['dem'].values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('DEM值')
        ax1.set_ylabel('频率')
        ax1.set_title('DEM值分布')
        ax1.grid(True, alpha=0.3)

        # 2. 分层结果
        ax2 = axes[0, 1]
        unique_strata = np.unique(data_df['stratum'].values)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_strata)))

        for i, stratum in enumerate(unique_strata):
            mask = data_df['stratum'].values == stratum
            ax2.scatter(data_df['x'].values[mask], data_df['y'].values[mask],
                        c=[colors[i]], s=30, alpha=0.7,
                        label=f'层 {stratum}')

        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title('DEM分层结果')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. 空间分块结果
        ax3 = axes[1, 0]
        unique_blocks = np.unique(data_df['block'].values)
        unique_blocks = unique_blocks[unique_blocks >= 0]

        if len(unique_blocks) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_blocks)))
            for i, block in enumerate(unique_blocks):
                mask = data_df['block'].values == block
                ax3.scatter(data_df['x'].values[mask], data_df['y'].values[mask],
                            c=[colors[i]], s=30, alpha=0.7,
                            label=f'块 {block}')

            if len(unique_blocks) <= 10:
                ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
        else:
            # 块太多，不显示图例
            scatter = ax3.scatter(data_df['x'].values, data_df['y'].values,
                                  c=data_df['block'].values, s=30, alpha=0.7,
                                  cmap='tab20')
            plt.colorbar(scatter, ax=ax3, label='块标签')

        ax3.set_xlabel('X坐标')
        ax3.set_ylabel('Y坐标')
        ax3.set_title('空间分块结果')
        ax3.grid(True, alpha=0.3)

        # 4. DEM与空间位置关系
        ax4 = axes[1, 1]
        scatter = ax4.scatter(data_df['x'].values, data_df['y'].values,
                              c=data_df['dem'].values, s=30, alpha=0.7,
                              cmap='terrain')
        plt.colorbar(scatter, ax=ax4, label='DEM值')
        ax4.set_xlabel('X坐标')
        ax4.set_ylabel('Y坐标')
        ax4.set_title('DEM空间分布')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{title}\nDEM分层空间分块可视化', fontsize=14)
        plt.tight_layout()

        # 保存图像
        if output_dir:
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = os.path.join(output_dir, f'dem_stratification_{safe_title}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  DEM分层可视化图已保存到: {filename}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    except Exception as e:
        print(f"  ⚠️  DEM分层可视化失败: {e}")
        return None


# 主函数，用于测试
if __name__ == "__main__":
    print("DEM分层空间交叉验证模块 - 通用优化版")
    print("=" * 60)
    print("主要特点:")
    print("1. 按DEM值分5层，保证环境梯度覆盖")
    print("2. 每层内使用K-means进行空间聚类分块")
    print("3. 每层抽取1-2个块作为测试集，确保空间独立")
    print("4. 适用于地形复杂、样本稀疏的研究区域")
    print("=" * 60)
