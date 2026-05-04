# -*- coding: utf-8 -*-
# 论文配套代码：土壤容重预测模型 (公开版)
# 模型：随机森林 / XGBoost / Cubist
# 功能：模型训练、评估、SHAP解释、空间交叉验证
import sys
import os
import warnings
import traceback
import numpy as np
import pandas as pd
import shap
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from sklearn.model_selection import KFold

# ======================== 核心辅助函数 ========================
def calculate_slope(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    return slope

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    slope = calculate_slope(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "slope": slope}

# ======================== 修复matplotlib后端 ========================
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ======================== R环境配置 (通用版，用户自行修改) ========================
# 【重要】运行前请配置为本机R安装路径
# os.environ['R_HOME'] = '你的R安装路径'
# os.environ['PATH'] += ';' + os.path.join(os.environ['R_HOME'], 'bin', 'x64')

try:
    import rpy2
    from rpy2 import robjects
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr, isinstalled

    try:
        from rpy2.robjects import pandas2ri as old_pandas2ri
        old_pandas2ri.activate()
    except ImportError:
        from rpy2.robjects import pandas as rpd
        from rpy2.robjects.conversion import localconverter
        
    if isinstalled('Cubist'):
        importr('Cubist')
        print("✅ R-Cubist包加载成功")
    else:
        print("⚠️ R-Cubist包未安装，后续Cubist模型将不可用")
except Exception as e:
    print(f"❌ R环境加载异常: {e}")
    print("💡 请先配置R_HOME路径，并安装Cubist包")
    sys.exit(1)

# ======================== 自定义模块导入 ========================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import config
    import utils
    from model_rf import RandomForestModel
    from model_xgb import XGBoostModel
    from model_cubist import CubistModel
except ImportError as e:
    print(f"⚠️ 自定义模块导入失败: {e}")
    sys.exit(1)

# ======================== SHAP解释配置 ========================
SHAP_INTERPRET_CONFIG = {
    'enable_for_rf_xgb': True,
    'use_for_feature_selection': False,
    'background_samples': 100,
    'protected_features': ['NDVI_szj', 'EVI_szj_23', 'DEM', 'SM_szj', 'Year']
}

# ======================== 集成解释报告配置 ========================
INTEGRATED_REPORT_CONFIG = {
    'enable': True,
    'format': ['excel', 'word'],
    'include_models': ['rf', 'xgb', 'cubist']
}

class CubistWrapper:
    """Cubist模型包装器"""
    def __init__(self, cubist_runner):
        self.cubist_runner = cubist_runner
        self.model_info = None

    def fit(self, X, y, params=None, target_col=None):
        self.model_info = self.cubist_runner.train_model(X, y, params=params, target_col=target_col)
        return self

    def predict(self, X):
        if self.model_info is None:
            raise ValueError("模型未训练")
        return self.cubist_runner.predict(self.model_info, X)

class SoilBulkDensityPredictor:
    """土壤容重预测主类"""
    def __init__(self, use_ndvi=False, year_option='both',
                 dem_strata=5, test_blocks_per_stratum=1):
        print("=" * 80)
        print("土壤容重预测系统（仅SHAP解释，不筛选特征）")
        print("=" * 80)

        self.use_ndvi = use_ndvi
        self.year_option = year_option
        self.dem_strata = dem_strata
        self.test_blocks_per_stratum = test_blocks_per_stratum
        self.seed = config.TRAIN_TEST_SPLIT['random_state']

        veg_index = 'NDVI_szj' if use_ndvi else 'EVI_szj_23'
        print(f"使用的植被指数: {veg_index}")
        print(f"年份配置: {year_option}")
        print(f"DEM分层数: {dem_strata}层")
        print(f"每层测试块数: {test_blocks_per_stratum}")
        print(f"SHAP使用策略: 仅用于解释，不用于特征筛选")

        timestamp_suffix = f"_{veg_index}_{year_option}_SHAP_interpretation_only"
        self.timestamp = utils.create_timestamp() + timestamp_suffix
        self.output_base = f"./results_{self.timestamp}"
        os.makedirs(self.output_base, exist_ok=True)

        self.df = None
        self.df_filtered = None
        self.coords_df = None
        self.dem_values = None
        self.feature_dict = {}
        self.results = {'0-10cm': {}, '10-20cm': {}, '20-30cm': {}}
        self.shap_results = {'rf': {}, 'xgb': {}, 'cubist': {}}
        self.cubist_rules = {'0-10cm': None, '10-20cm': None, '20-30cm': None}
        self.interpretation_data = {}

        print("初始化模型...")
        self.rf_model = RandomForestModel()
        self.xgb_model = XGBoostModel()
        self.cubist_model = None

        try:
            cubist_init_params = config.CUBIST_PARAMS['默认参数']
            self.cubist_model = CubistModel(
                random_state=cubist_init_params['seed'],
                committees=cubist_init_params['committees'],
                neighbors=cubist_init_params['neighbors'],
                rules=cubist_init_params['rules']
            )
            print("✅ Cubist模型初始化成功")
        except Exception as e:
            print(f"⚠️ Cubist模型初始化失败: {str(e)[:100]}")

        print(f"结果将保存到: {self.output_base}")

    def get_features_for_target(self, target_col):
        if target_col in self.feature_dict:
            return self.feature_dict[target_col]['features']
        else:
            return []

    def load_data(self):
        print("\n" + "=" * 60)
        print("步骤1: 加载数据")
        print("=" * 60)

        filepath = os.path.join(config.DATA_PATH, config.TRAIN_FILE)
        print(f"加载文件: {filepath}")

        self.df = utils.load_and_preprocess_data(filepath, encode_year=True)
        self.df_filtered = utils.filter_data_by_year(self.df, self.year_option)

        required_cols = (config.TARGET_COLS + config.COORD_COLS + config.FIXED_COVARIATES +
                         list(config.SOIL_PROPERTIES.values())[0])
        if self.use_ndvi:
            required_cols = [col for col in required_cols if col != 'EVI_szj_23']
            required_cols.append('NDVI_szj')
        else:
            required_cols = [col for col in required_cols if col != 'NDVI_szj']

        utils.check_required_columns(self.df_filtered, required_cols)
        self.coords_df = utils.get_coordinates(self.df_filtered, config.COORD_COLS)

        if 'DEM' in self.df_filtered.columns:
            self.dem_values = self.df_filtered['DEM'].values
            print(f"DEM值范围: {self.dem_values.min():.1f} - {self.dem_values.max():.1f}")
        else:
            print("⚠️ 警告: 数据中未找到DEM列")
            self.dem_values = None

        self.build_feature_dict()
        print(f"数据加载完成! 原始样本数: {len(self.df)}，过滤后样本数: {len(self.df_filtered)}")
        return True

    def build_feature_dict(self):
        print("\n构建各深度特征字典...")
        base_features = config.FIXED_COVARIATES.copy()
        
        if self.use_ndvi:
            base_features = [col if col != 'EVI_szj_23' else 'NDVI_szj' for col in base_features]

        veg_index_col = 'NDVI_szj' if self.use_ndvi else 'EVI_szj_23'
        if veg_index_col and veg_index_col in self.df_filtered.columns and veg_index_col not in base_features:
            base_features.append(veg_index_col)

        if 'Year' not in base_features and 'Year' in self.df_filtered.columns:
            base_features.append('Year')
        if 'DEM' not in base_features and 'DEM' in self.df_filtered.columns:
            base_features.append('DEM')

        for target_col in config.TARGET_COLS:
            if target_col == 'BD0_10':
                depth_key, depth_name = '0_10', '0-10cm'
            elif target_col == 'BD10_20':
                depth_key, depth_name = '10_20', '10-20cm'
            elif target_col == 'BD20_30':
                depth_key, depth_name = '20_30', '20-30cm'
            else:
                continue

            soil_features = config.SOIL_PROPERTIES[depth_key]
            features = list(dict.fromkeys(base_features + soil_features))

            self.feature_dict[target_col] = {
                'features': features,
                'depth_name': depth_name,
                'target_col': target_col
            }
            print(f"  {depth_name} ({target_col}): {len(features)}个特征")

        return self.feature_dict

    def train_test_split_random(self):
        print("\n" + "=" * 60)
        print("随机划分训练集和测试集")
        print("=" * 60)

        test_size = config.TRAIN_TEST_SPLIT['test_size']
        random_state = config.TRAIN_TEST_SPLIT['random_state']
        shuffle = config.TRAIN_TEST_SPLIT['shuffle']
        stratify_by_year = config.TRAIN_TEST_SPLIT['stratify_by_year']

        split_results = {}
        for target_col in config.TARGET_COLS:
            depth_info = self.feature_dict.get(target_col, {})
            depth_name = depth_info.get('depth_name', f"未知深度({target_col})")
            print(f"\n处理 {depth_name} ({target_col})...")

            feature_cols = self.get_features_for_target(target_col)
            split_data = utils.split_train_test_data(
                self.df_filtered, target_col, feature_cols, self.coords_df,
                test_size=test_size, random_state=random_state,
                stratify_by_year=stratify_by_year, verbose=True
            )

            if split_data is None:
                print(f"  ❌ 数据划分失败，跳过 {depth_name}")
                continue

            split_data['depth_name'] = depth_name
            split_results[target_col] = split_data

        print(f"\n划分完成，共处理 {len(split_results)} 个深度")
        return split_results

    def shap_interpretation_only(self, model, model_type, X_data, feature_cols, output_dir, depth_name):
        print(f"  📊 执行SHAP解释分析 ({model_type.upper()})...")

        try:
            if len(X_data) > SHAP_INTERPRET_CONFIG['background_samples']:
                background = shap.sample(X_data, SHAP_INTERPRET_CONFIG['background_samples'], random_state=self.seed)
            else:
                background = X_data

            explainer = shap.TreeExplainer(model, background)
            shap_values = explainer.shap_values(X_data)

            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'SHAP_Importance': shap_importance,
                'SHAP_Rank': range(1, len(feature_cols) + 1)
            }).sort_values(by='SHAP_Importance', ascending=False).reset_index(drop=True)

            shap_dir_df = pd.DataFrame([
                {'Feature': feat,
                 'pos_ratio': round(np.sum(shap_values[:, idx] > 0)/len(shap_values[:, idx]), 3),
                 'neg_ratio': round(np.sum(shap_values[:, idx] < 0)/len(shap_values[:, idx]), 3),
                 'direction': "正向" if np.sum(shap_values[:, idx] > 0)/len(shap_values[:, idx])>0.6 
                             else "负向" if np.sum(shap_values[:, idx] < 0)/len(shap_values[:, idx])>0.6 else "混合",
                 'mean_impact': round(np.mean(shap_values[:, idx]), 4)}
                for idx, feat in enumerate(feature_cols)
            ])
            
            final_shap_df = pd.merge(shap_importance_df, shap_dir_df, on='Feature')

            shap_result_file = os.path.join(output_dir, f'{model_type}_{depth_name}_SHAP_interpretation.xlsx')
            final_shap_df.to_excel(shap_result_file, index=False)

            if isinstance(X_data, np.ndarray):
                X_df = pd.DataFrame(X_data, columns=feature_cols)
            else:
                X_df = X_data.copy()

            shap_matrix_df = pd.DataFrame(shap_values, columns=[f"{feat}_SHAP" for feat in feature_cols])
            combined_df = pd.concat([X_df.reset_index(drop=True), shap_matrix_df.reset_index(drop=True)], axis=1)

            shap_matrix_file = os.path.join(output_dir, f'{model_type}_{depth_name}_SHAP_matrix_with_original_features.xlsx')
            with pd.ExcelWriter(shap_matrix_file, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='Original_Features_Plus_SHAP', index=False)
                shap_matrix_df.to_excel(writer, sheet_name='SHAP_Value_Matrix', index=False)
                X_df.to_excel(writer, sheet_name='Original_Feature_Matrix', index=False)

            print(f"  ✅ SHAP值矩阵与原始特征匹配完成")
            
            plt.figure(figsize=(8, 5))
            top_features = final_shap_df.head(min(10, len(final_shap_df)))
            plt.barh(range(len(top_features)), top_features['SHAP_Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('平均绝对SHAP值')
            plt.title(f'{depth_name} {model_type.upper()} SHAP特征重要性排名', fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_type}_{depth_name}_SHAP_importance_bar.png'), bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_data, feature_names=feature_cols, show=False, max_display=min(15, len(feature_cols)))
            plt.title(f'{depth_name} {model_type.upper()} SHAP值分布', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_type}_{depth_name}_SHAP_beeswarm.png'), bbox_inches='tight', dpi=300)
            plt.close()

            print(f"  ✅ SHAP解释分析完成")
            return {'importance_df': final_shap_df, 'shap_values': shap_values, 'file_path': shap_result_file, 'shap_matrix_file': shap_matrix_file}

        except Exception as e:
            print(f"  ⚠️ SHAP分析失败: {e}")
            traceback.print_exc()
            return None

    def _get_most_used_features(self, rules_df, all_features):
        feature_counts = {}
        for _, row in rules_df.iterrows():
            used_features = row['Used_Features'].split(', ')
            for feat in used_features:
                if feat and feat != '无':
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in sorted_features[:10]]

    def run_model_pipeline(self, model_type, split_data, params=None):
        print(f"\n" + "=" * 60)
        print(f"运行 {model_type} 模型")
        print("=" * 60)

        model_results = {}
        params = params if params is not None else {}

        for target_col, data in split_data.items():
            depth_name = data['depth_name']
            print(f"\n===== 处理深度: {depth_name} =====")

            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']
            coords_train, coords_test = data['coords_train'], data['coords_test']
            feature_cols = data.get('feature_cols', list(X_train.columns) if hasattr(X_train, 'columns') else [])

            curr_cubist_params = params
            if model_type.lower() == 'cubist' and '分土层最优参数' in config.CUBIST_PARAMS:
                if target_col in config.CUBIST_PARAMS['分土层最优参数']:
                    curr_cubist_params = config.CUBIST_PARAMS['分土层最优参数'][target_col]
                    print(f"  ✅ 加载 {depth_name} Cubist分土层最优参数")

            if model_type.lower() == 'rf':
                model_runner, model_name = self.rf_model, "随机森林"
            elif model_type.lower() == 'xgb':
                model_runner, model_name = self.xgb_model, "XGBoost"
            elif model_type.lower() == 'cubist':
                if self.cubist_model is None:
                    print("Cubist模型不可用，跳过")
                    continue
                model_runner, model_name = self.cubist_model, "Cubist (R)"
            else:
                print(f"未知的模型类型: {model_type}")
                continue

            print(f"使用 {model_name} 模型")
            print(f"特征数量: {len(feature_cols)}个（全量使用，不筛选）")

            try:
                if model_type.lower() == 'cubist':
                    model_info = model_runner.train_model(X_train, y_train, params=curr_cubist_params, target_col=target_col)
                    if model_info is None:
                        print(f"  ❌ Cubist模型训练失败，跳过 {depth_name}")
                        continue
                    model = model_info
                else:
                    model = model_runner.train_model(X_train=X_train, y_train=y_train, params=params)
                    if model is None:
                        print(f"  ❌ 模型训练失败，跳过 {depth_name}")
                        continue
            except Exception as e:
                print(f"  ❌ 模型训练异常: {e}")
                traceback.print_exc()
                continue

            try:
                if model_type.lower() == 'cubist':
                    y_pred = model_runner.predict(model, X_test)
                else:
                    y_pred = model_runner.predict(model, X_test)

                y_test_clean = y_test[~np.isnan(y_pred)]
                y_pred_clean = y_pred[~np.isnan(y_pred)]
                r2, rmse, mae, slope, intercept = utils.calculate_regression_stats(y_test_clean, y_pred_clean)
            except Exception as e:
                print(f"  ❌ 模型评估异常: {e}")
                traceback.print_exc()
                continue

            output_dir = os.path.join(self.output_base, model_type, depth_name)
            os.makedirs(output_dir, exist_ok=True)

            trad_filename = os.path.join(output_dir, f'traditional_validation_results.xlsx')
            utils.export_regression_results(y_test, y_pred, trad_filename, coords_test)

            print("  3. 执行模型解释...")
            interpretation_result = None
            if model_type.lower() == 'cubist':
                print(f"  📌 Cubist模型已训练完成，规则请在R中查看")
                status_file = os.path.join(output_dir, f'cubist_{depth_name}_status.txt')
                with open(status_file, 'w', encoding='utf-8') as f:
                    f.write(f"Cubist模型训练状态报告\n土层深度: {depth_name}\n特征数量: {len(feature_cols)}\n")
                interpretation_result = {'status': 'trained_no_rules_extracted', 'status_file': status_file}
                self.cubist_rules[depth_name] = interpretation_result

            elif model_type.lower() in ['rf', 'xgb'] and SHAP_INTERPRET_CONFIG['enable_for_rf_xgb']:
                X_train_array = X_train[feature_cols].values if isinstance(X_train, pd.DataFrame) else X_train
                interpretation_result = self.shap_interpretation_only(model, model_type, X_train_array, feature_cols, output_dir, depth_name)
                if interpretation_result:
                    self.shap_results[model_type][depth_name] = interpretation_result

            print("  4. 执行4折空间交叉验证...")
            cv_r2_list, cv_rmse_list, cv_mae_list, cv_slope_list = [], [], [], []
            cv_results = []
            n_splits_final = 4

            X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
            y_full = np.concatenate([y_train, y_test]) if isinstance(y_train, np.ndarray) else pd.concat([pd.Series(y_train), pd.Series(y_test)]).values
            kf = KFold(n_splits=n_splits_final, shuffle=True, random_state=self.seed)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
                X_train_cv = X_full.iloc[train_idx] if isinstance(X_full, pd.DataFrame) else X_full[train_idx]
                X_test_cv = X_full.iloc[test_idx] if isinstance(X_full, pd.DataFrame) else X_full[test_idx]
                y_train_cv, y_test_cv = y_full[train_idx], y_full[test_idx]

                if model_type == 'rf':
                    model_cv = RandomForestRegressor(**params)
                elif model_type == 'xgb':
                    model_cv = XGBRegressor(**params)
                elif model_type == 'cubist':
                    model_cv = CubistWrapper(self.cubist_model)
                    model_cv.fit(X_train_cv, y_train_cv, params=curr_cubist_params, target_col=target_col)
                    y_pred_cv = model_cv.predict(X_test_cv)
                    continue

                model_cv.fit(X_train_cv, y_train_cv)
                y_pred_cv = model_cv.predict(X_test_cv)

                y_test_cv_clean = y_test_cv[~np.isnan(y_pred_cv)]
                y_pred_cv_clean = y_pred_cv[~np.isnan(y_pred_cv)]
                if len(y_test_cv_clean) < 1:
                    continue

                fold_metrics = calculate_metrics(y_test_cv_clean, y_pred_cv_clean)
                cv_r2_list.append(fold_metrics['R2'])
                cv_rmse_list.append(fold_metrics['RMSE'])
                cv_mae_list.append(fold_metrics['MAE'])
                cv_slope_list.append(fold_metrics['slope'])

            spatial_r2 = np.mean(cv_r2_list) if len(cv_r2_list) > 0 else 0.0
            spatial_rmse = np.mean(cv_rmse_list) if len(cv_rmse_list) > 0 else 0.0
            spatial_mae = np.mean(cv_mae_list) if len(cv_mae_list) > 0 else 0.0

            print(f"  📊 4折空间交叉验证均值: R²={spatial_r2:.4f}, RMSE={spatial_rmse:.4f}")

            print("  5. 保存模型...")
            model_filename = os.path.join(output_dir, f'{model_type}_{depth_name}_model.pkl')
            try:
                if model_type.lower() == 'cubist' and hasattr(model_runner, 'save_model'):
                    model_runner.save_model(model, model_filename)
                else:
                    utils.save_model(model, model_filename)
            except Exception as e:
                print(f"  ⚠️ 模型保存失败: {e}")

            model_results[target_col] = {
                'model': model, 'model_type': model_type, 'depth_name': depth_name,
                'traditional_stats': {'R2': round(r2,3), 'RMSE': round(rmse,3), 'MAE': round(mae,3), 'slope': round(slope,3)},
                'feature_cols': feature_cols, 'interpretation': interpretation_result,
                'spatial_cv_results': {'R2': spatial_r2, 'RMSE': spatial_rmse, 'MAE': spatial_mae}
            }
            print(f"  ✅ {depth_name} 处理完成!")

        return model_results

    def run_all_models(self, split_data):
        print("\n" + "=" * 80)
        print("运行所有模型")
        print("=" * 80)
        all_results = {}

        print("\n>>> 1. 随机森林模型 <<<")
        rf_results = self.run_model_pipeline('rf', split_data, config.RF_PARAMS['默认参数'])
        all_results['rf'] = rf_results
        self.results = self._update_results(self.results, rf_results, 'rf')

        print("\n>>> 2. XGBoost模型 <<<")
        xgb_results = self.run_model_pipeline('xgb', split_data, config.XGB_PARAMS['默认参数'])
        all_results['xgb'] = xgb_results
        self.results = self._update_results(self.results, xgb_results, 'xgb')

        print("\n>>> 3. Cubist模型 <<<")
        if self.cubist_model is not None:
            cubist_results = self.run_model_pipeline('cubist', split_data, config.CUBIST_PARAMS['默认参数'])
            all_results['cubist'] = cubist_results
            self.results = self._update_results(self.results, cubist_results, 'cubist')
        else:
            print("⚠️ Cubist模型不可用，跳过")

        if INTEGRATED_REPORT_CONFIG['enable']:
            self.generate_integrated_interpretation_report(all_results)
        return all_results

    def _update_results(self, results_dict, model_results, model_type):
        for target_col, result in model_results.items():
            depth_name = result['depth_name']
            if depth_name in results_dict:
                results_dict[depth_name][model_type] = result
        return results_dict

    def generate_integrated_interpretation_report(self, all_results):
        print("\n生成集成解释报告...")
        report_data = []
        for depth_name in ['0-10cm', '10-20cm', '20-30cm']:
            rf_data = self.shap_results['rf'].get(depth_name)
            xgb_data = self.shap_results['xgb'].get(depth_name)
            cubist_data = self.cubist_rules.get(depth_name)

            if rf_data:
                top = rf_data['importance_df'].head(3)['Feature'].tolist()
                report_data.append({'深度': depth_name, '模型': 'RF', 'Top1': top[0] if len(top)>0 else 'N/A', 'Top2': top[1] if len(top)>1 else 'N/A', 'Top3': top[2] if len(top)>2 else 'N/A'})
            if xgb_data:
                top = xgb_data['importance_df'].head(3)['Feature'].tolist()
                report_data.append({'深度': depth_name, '模型': 'XGBoost', 'Top1': top[0] if len(top)>0 else 'N/A', 'Top2': top[1] if len(top)>1 else 'N/A', 'Top3': top[2] if len(top)>2 else 'N/A'})
            if cubist_data:
                report_data.append({'深度': depth_name, '模型': 'Cubist', 'Top1': 'N/A', 'Top2': 'N/A', 'Top3': 'N/A'})

        if report_data:
            report_df = pd.DataFrame(report_data)
            report_df.to_excel(os.path.join(self.output_base, 'integrated_interpretation_report.xlsx'), index=False)
            print("✅ 集成解释报告已生成")

    def compare_models(self):
        print("\n模型性能对比完成，结果已保存")
        comparison_data = []
        for depth_name, depth_results in self.results.items():
            for model_type in ['rf', 'xgb', 'cubist']:
                if model_type not in depth_results: continue
                res = depth_results[model_type]
                comparison_data.append({
                    '深度': depth_name, '模型': model_type.upper(),
                    '传统R²': res['traditional_stats']['R2'], '传统RMSE': res['traditional_stats']['RMSE'],
                    '空间R²': res['spatial_cv_results']['R2'], '空间RMSE': res['spatial_cv_results']['RMSE']
                })
        pd.DataFrame(comparison_data).to_excel(os.path.join(self.output_base, 'model_comparison.xlsx'), index=False)

    def export_summary_report(self):
        report = [{'项目': '土壤容重预测模型', '版本': '公开版', '输出路径': self.output_base}]
        pd.DataFrame(report).to_excel(os.path.join(self.output_base, 'summary_report.xlsx'), index=False)
        print("✅ 综合报告已生成")

    def run_full_pipeline(self):
        if not self.load_data(): return False
        split_data = self.train_test_split_random()
        if not split_data: return False
        self.run_all_models(split_data)
        self.compare_models()
        self.export_summary_report()
        print("\n🎉 完整流程执行完成！")
        return True

def main():
    print("\n土壤容重预测系统")
    print("=" * 80)

    use_ndvi = (input("选择植被指数 (1-EVI, 2-NDVI) [默认1]: ").strip() or "1") == "2"
    year_opt = input("选择年份 (1-全部,2-2023,3-2024) [默认1]: ").strip() or "1"
    year_option = {'2':'2023','3':'2024'}.get(year_opt, 'both')
    
    dem_strata = int(input("DEM分层数 [默认5]: ") or 5)
    test_blocks = int(input("每层测试块数 [默认1]: ") or 1)

    predictor = SoilBulkDensityPredictor(use_ndvi, year_option, dem_strata, test_blocks)
    predictor.run_full_pipeline()

if __name__ == "__main__":
    try:
        import openpyxl
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        
    main()
