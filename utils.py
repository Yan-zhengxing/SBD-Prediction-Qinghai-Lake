"""
工具函数模块 - 空间预测通用工具集
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')

# 导入配置文件
import config


def load_and_preprocess_data(filepath, sheet_name=0, encode_year=True):
    """
    加载和预处理数据

    参数:
        filepath: 数据文件路径
        sheet_name: Excel工作表名称
        encode_year: 是否编码Year变量
    """
    print(f"加载数据: {filepath}")

    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    # 根据文件扩展名选择加载方法
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")

    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")

    # 编码Year变量
    if encode_year and 'Year' in df.columns:
        original_years = df['Year'].unique()
        print(f"原始Year值: {sorted(original_years)}")

        # 映射Year值
        df['Year'] = df['Year'].map(config.YEAR_ENCODING)

        # 检查映射结果
        encoded_years = df['Year'].unique()
        valid_encoded = encoded_years[~np.isnan(encoded_years)]
        print(f"编码后Year值: {sorted(valid_encoded)}")

        # 统计各年份样本数
        print(f"各年份样本统计:")
        year_mapping_rev = {v: k for k, v in config.YEAR_ENCODING.items()}
        for year_code in valid_encoded:
            year_original = year_mapping_rev.get(int(year_code), "未知")
            count = (df['Year'] == year_code).sum()
            print(f"  Year={int(year_code)} ({year_original}): {count}个样本")

    # 检查目标列是否存在
    missing_targets = [col for col in config.TARGET_COLS if col not in df.columns]
    if missing_targets:
        print(f"警告: 目标列 {missing_targets} 不存在")

    # 检查坐标列是否存在
    missing_coords = [col for col in config.COORD_COLS if col not in df.columns]
    if missing_coords:
        print(f"警告: 坐标列 {missing_coords} 不存在")

    # 检查固定协变量是否存在
    for covariate in config.FIXED_COVARIATES:
        if covariate not in df.columns:
            # 植被指数兼容替换
            if covariate == config.VEG_INDEX_CONFIG['evi_col'] and config.VEG_INDEX_CONFIG['ndvi_col'] in df.columns:
                print(f"注意: 使用NDVI替代EVI")
            else:
                print(f"警告: 固定协变量 {covariate} 不存在")

    # 检查土壤属性列是否存在
    for depth, properties in config.SOIL_PROPERTIES.items():
        for prop in properties:
            if prop not in df.columns:
                print(f"警告: 土壤属性列 {prop} 不存在")

    # 显示数据基本信息
    print(f"\n数据基本信息:")
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")
    print(f"  目标变量: {config.TARGET_COLS}")
    print(f"  坐标列: {config.COORD_COLS}")

    # 检查各深度目标变量的有效样本数
    for target_col in config.TARGET_COLS:
        if target_col in df.columns:
            n_valid = df[target_col].notna().sum()
            print(f"  {target_col}: {n_valid}个有效样本")

    return df


def get_coordinates(df, coord_cols):
    """
    提取坐标数据
    """
    # 检查坐标列是否存在
    missing_cols = [col for col in coord_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"坐标列不存在: {missing_cols}")

    # 提取坐标
    coords = df[coord_cols].copy()

    # 重命名列名以便后续处理
    if coord_cols == ['POINT_X', 'POINT_Y']:
        coords.columns = ['x_coord', 'y_coord']

    return coords


def check_required_columns(df, required_cols):
    """
    检查必需列是否存在
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"警告: 以下必需列不存在: {missing_cols}")
        print(f"可用列: {list(df.columns)}")
        return False

    return True


def split_train_test_data(df, target_col, feature_cols, coords_df, test_size=0.2, random_state=42,
                          stratify_by_year=False, verbose=True):
    """
    划分训练集和测试集

    参数:
        df: 完整数据框
        target_col: 目标变量列名
        feature_cols: 特征列名列表
        coords_df: 坐标数据框
        test_size: 测试集比例
        random_state: 随机种子
        stratify_by_year: 是否按年份分层抽样
        verbose: 是否输出详细信息
    """
    # 获取有目标值的样本
    mask = df[target_col].notna()
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, target_col]
    coords = coords_df.loc[mask]

    if len(X) < 10:
        if verbose:
            print(f"  样本太少 ({len(X)}个)，无法划分")
        return None

    # 准备分层抽样参数
    stratify = None
    if stratify_by_year and 'Year' in X.columns:
        # 按年份分层抽样
        stratify = X['Year'].values
        if verbose:
            print(f"  按Year分层抽样: 分布 {np.unique(stratify, return_counts=True)}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True
    )

    # 获取对应的坐标
    coords_train = coords.loc[X_train.index]
    coords_test = coords.loc[X_test.index]

    if verbose:
        print(f"  训练集: {len(X_train)}个样本，测试集: {len(X_test)}个样本")
        print(f"  划分比例: 训练集 {len(X_train) / len(X):.1%}，测试集 {len(X_test) / len(X):.1%}")

        # 显示Year分布
        if 'Year' in X_train.columns:
            year_mapping_rev = {v: k for k, v in config.YEAR_ENCODING.items()}
            for year_code in np.unique(X_train['Year']):
                year_original = year_mapping_rev.get(int(year_code), "未知")
                train_count = (X_train['Year'] == year_code).sum()
                test_count = (X_test['Year'] == year_code).sum()
                print(f"  Year={int(year_code)} ({year_original}): 训练集 {train_count}，测试集 {test_count}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'coords_train': coords_train,
        'coords_test': coords_test,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_cols': feature_cols
    }


def calculate_regression_stats(y_true, y_pred):
    """
    计算回归统计指标
    """
    # 确保是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 移除NaN值
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    if len(y_true_valid) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # 计算R²
    r2 = r2_score(y_true_valid, y_pred_valid)

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))

    # 计算MAE
    mae = mean_absolute_error(y_true_valid, y_pred_valid)

    # 计算回归线斜率和截距
    if len(y_true_valid) > 1:
        slope, intercept = np.polyfit(y_true_valid, y_pred_valid, 1)
    else:
        slope, intercept = np.nan, np.nan

    return r2, rmse, mae, slope, intercept


def create_timestamp():
    """
    创建时间戳
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model(model, filename):
    """
    保存模型
    """
    try:
        joblib.dump(model, filename)
        print(f"模型已保存到: {filename}")
        return True
    except Exception as e:
        print(f"保存模型失败: {e}")
        return False


def load_model(filename):
    """
    加载模型
    """
    try:
        model = joblib.load(filename)
        print(f"模型已从 {filename} 加载")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def export_regression_results(y_true, y_pred, filename, coords=None, fold_info=None, year_info=None):
    """
    导出回归结果到Excel

    参数:
        y_true: 真实值
        y_pred: 预测值
        filename: 输出文件名
        coords: 坐标数据
        fold_info: 交叉验证的fold信息
        year_info: 年份信息
    """
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'actual_value': y_true,
        'predicted_value': y_pred
    })

    # 计算残差
    results_df['residual'] = results_df['actual_value'] - results_df['predicted_value']
    results_df['absolute_error'] = np.abs(results_df['residual'])
    results_df['relative_error'] = results_df['residual'] / results_df['actual_value']

    # 添加坐标信息
    if coords is not None:
        for col in coords.columns:
            results_df[col] = coords[col].values if len(coords) == len(results_df) else np.nan

    # 添加fold信息
    if fold_info is not None:
        results_df['fold'] = fold_info.values if len(fold_info) == len(results_df) else np.nan

    # 添加年份信息
    if year_info is not None:
        results_df['Year'] = year_info.values if len(year_info) == len(results_df) else np.nan
        # 动态映射原始年份
        year_mapping_rev = {v: k for k, v in config.YEAR_ENCODING.items()}
        results_df['Year_name'] = results_df['Year'].map(lambda x: year_mapping_rev.get(int(x), "未知") if pd.notna(x) else np.nan)

    # 保存到Excel
    results_df.to_excel(filename, index=False)

    # 计算统计指标
    r2, rmse, mae, slope, intercept = calculate_regression_stats(y_true, y_pred)

    print(f"回归结果已保存到: {filename}")
    print(f"  样本数: {len(y_true)}")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # 按年份分析（如果年份信息可用）
    if year_info is not None and 'Year' in results_df.columns:
        print(f"  按年份分析:")
        year_mapping_rev = {v: k for k, v in config.YEAR_ENCODING.items()}
        for year_code in np.unique(results_df['Year'].dropna()):
            mask = results_df['Year'] == year_code
            year_samples = mask.sum()
            year_original = year_mapping_rev.get(int(year_code), "未知")
            if year_samples > 0:
                year_r2 = r2_score(results_df.loc[mask, 'actual_value'],
                                   results_df.loc[mask, 'predicted_value'])
                year_rmse = np.sqrt(mean_squared_error(results_df.loc[mask, 'actual_value'],
                                                       results_df.loc[mask, 'predicted_value']))
                print(f"    Year={int(year_code)} ({year_original}): {year_samples}样本, R²={year_r2:.4f}, RMSE={year_rmse:.4f}")

    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'slope': slope,
        'intercept': intercept,
        'filename': filename
    }


def filter_data_by_year(df, year_option='both'):
    """
    根据年份选项过滤数据

    参数:
        df: 数据框
        year_option: 'both' / 原始年份值

    返回:
        过滤后的数据框
    """
    if 'Year' not in df.columns:
        print("警告: 数据中没有Year列")
        return df.copy()

    year_mapping = config.YEAR_ENCODING
    year_mapping_rev = {v: k for k, v in year_mapping.items()}

    if year_option == 'both':
        filtered_df = df.copy()
        count_info = ", ".join([f"{k}年{v}样本" for k, v in df['Year'].value_counts().to_dict().items()])
        print(f"使用所有年份数据: {count_info}")
    else:
        # 匹配编码后的年份
        target_code = year_mapping.get(year_option, None)
        if target_code is not None:
            filtered_df = df[df['Year'] == target_code].copy()
            print(f"使用{year_option}年数据: {len(filtered_df)}样本")
        else:
            print(f"警告: 未知年份选项，使用所有数据")
            filtered_df = df.copy()

    return filtered_df
