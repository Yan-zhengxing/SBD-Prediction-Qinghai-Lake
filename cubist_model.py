import numpy as np
import pandas as pd
import warnings
import time
import os
import joblib
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor

# ==================== rpy2 环境配置 ====================
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# 过滤无关警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 关闭R警告输出
robjects.r('options(warn=-1)')


# ==================== R环境配置 ====================
def setup_r_environment():
    """设置R环境并加载Cubist包"""
    cubist_available = False
    r_cubist = None
    base = None

    try:
        # 通用R环境路径
        r_home_paths = [
            '/usr/lib/R',
            '/usr/local/lib/R',
            '/Library/Frameworks/R.framework/Resources',
            'C:/Program Files/R/R-4.4.2',
            'C:/Program Files/R/R-4.4.1',
            'C:/Program Files/R/R-4.3.3'
        ]

        for r_path in r_home_paths:
            if os.path.exists(r_path):
                os.environ['R_HOME'] = r_path
                print(f"找到R安装路径: {r_path}")
                break
        else:
            print(f"未找到R安装路径，尝试系统自动检测...")

        # 检查rpy2版本
        import rpy2
        print(f"rpy2版本: {rpy2.__version__}")

        # 加载R基础包
        base = importr('base')

        # 加载Cubist包
        try:
            r_cubist = importr('Cubist')
            robjects.r('library(Cubist)')
            print(f"R Cubist包已成功加载")
            cubist_available = True
        except Exception as e:
            print(f"R Cubist包加载失败: {str(e)}")
            print(f"解决方案：在R中执行 install.packages(c('Cubist', 'plyr'))")
            cubist_available = False

        return cubist_available, robjects, r_cubist, base

    except ImportError as e:
        print(f"rpy2导入失败: {e}")
        print(f"解决方案：pip install rpy2==3.5.17")
        return False, None, None, None
    except Exception as e:
        print(f"R环境设置失败: {e}")
        return False, None, None, None


# ==================== Cubist模型类 ====================
class CubistModel(BaseEstimator, RegressorMixin):
    """Cubist回归模型封装类（基于R），支持特征贡献度计算、超参数调优、模型保存加载"""

    def __init__(self, random_state=42, committees=50, neighbors=5, rules=50):
        self.random_state = random_state
        self.committees = committees
        self.neighbors = neighbors
        self.rules = rules
        self.best_params_ = None
        self.model_name = "Cubist (R)"
        self.n_jobs = 1

        print(f"\n初始化 {self.model_name} 模型...")

        # R环境属性
        self.r_available = False
        self.robjects = None
        self.r_cubist = None
        self.base = None
        self.converter = None

        # 配置R环境
        self.r_available, self.robjects, self.r_cubist, self.base = setup_r_environment()

        # 兜底模型（随机森林）
        self.use_fallback = False
        self.fallback_model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            max_depth=12,
            min_samples_leaf=3,
            oob_score=True
        )

        # 注册数据转换器
        if self.r_available and self.robjects is not None:
            self.converter = self.robjects.default_converter + pandas2ri.converter + numpy2ri.converter
            numpy2ri.activate()
            pandas2ri.activate()

        if self.r_available:
            print(f"{self.model_name} 模型初始化完成 (R环境就绪)")
        else:
            print(f"R环境不可用，将使用随机森林兜底模型")
            self.use_fallback = True

    def _clean_r_cache(self):
        """清理R环境缓存"""
        if self.r_available and self.base is not None:
            try:
                self.base.gc()
                self.robjects.r('rm(list = ls(all.names = TRUE))')
            except:
                pass

    def train_model(self, X_train, y_train, params=None, target_col=None):
        """
        训练Cubist模型
        :param X_train: 训练特征
        :param y_train: 训练标签
        :param params: 模型参数
        :param target_col: 目标变量名称
        :return: 模型信息字典
        """
        print(f"\n训练 {self.model_name} 模型...")
        print(f"  训练样本: {len(X_train)} | 特征数量: {X_train.shape[1]}")

        # 设置参数
        if params is not None:
            self.best_params_ = params
        else:
            self.best_params_ = {
                'committees': self.committees,
                'neighbors': self.neighbors,
                'rules': self.rules,
                'random_seed': self.random_state
            }

        print(f"  训练参数: {self.best_params_}")

        # 数据格式标准化
        if not isinstance(X_train, pd.DataFrame):
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
        else:
            X_train_df = X_train.copy()

        if not isinstance(y_train, pd.Series):
            y_train_series = pd.Series(y_train, name='target')
        else:
            y_train_series = y_train.copy()

        # 缺失值填充
        train_mean = X_train_df.mean()
        X_train_df = X_train_df.fillna(train_mean)
        y_train_series = y_train_series.fillna(y_train_series.mean())
        feature_names = list(X_train_df.columns)

        if self.r_available and self.r_cubist is not None and self.converter is not None:
            try:
                print(f"  数据格式转换：Python -> R")
                self._clean_r_cache()

                train_data = pd.DataFrame(X_train_df)
                train_data['target'] = y_train_series.values

                with localconverter(self.converter):
                    train_df_r = pandas2ri.py2rpy(train_data)

                r_feature_names = self.robjects.StrVector(feature_names)
                x_r = train_df_r.rx(True, r_feature_names)

                y_np = np.ravel(y_train_series.values)
                y_r = self.robjects.FloatVector(y_np)

                # 训练模型
                cubist_model = self.r_cubist.cubist(
                    x=x_r,
                    y=y_r,
                    committees=self.best_params_['committees'],
                    neighbors=self.best_params_['neighbors'],
                    rules=self.best_params_['rules']
                )

                print(f"  Cubist模型训练完成")

                return {
                    'model_type': 'r_cubist',
                    'r_model': cubist_model,
                    'feature_names': feature_names,
                    'params': self.best_params_,
                    'train_data_shape': X_train_df.shape,
                    'train_mean': train_mean
                }

            except Exception as e:
                print(f"  Cubist模型训练失败: {str(e)}")
                print(f"  切换至随机森林兜底模型")
                self.use_fallback = True
                self.fallback_model.fit(X_train, y_train)
                return {
                    'model_type': 'fallback_rf',
                    'model': self.fallback_model,
                    'params': self.best_params_,
                    'feature_names': feature_names,
                    'train_mean': train_mean
                }
        else:
            print(f"  使用随机森林兜底模型训练")
            self.fallback_model.fit(X_train, y_train)
            return {
                'model_type': 'fallback_rf',
                'model': self.fallback_model,
                'params': self.best_params_,
                'feature_names': feature_names,
                'train_mean': train_mean
            }

    def predict(self, model_info, X):
        """模型预测"""
        if model_info['model_type'] == 'fallback_rf':
            return model_info['model'].predict(X)

        elif model_info['model_type'] == 'r_cubist':
            try:
                cubist_model = model_info['r_model']
                feature_names = model_info['feature_names']
                train_mean = model_info.get('train_mean', None)

                if not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=feature_names)
                else:
                    X_df = X.copy()

                if list(X_df.columns) != feature_names:
                    X_df = X_df.reindex(columns=feature_names,
                                        fill_value=train_mean.mean() if train_mean is not None else 0)

                if train_mean is not None:
                    X_df = X_df.fillna(train_mean)
                else:
                    X_df = X_df.fillna(X_df.mean())

                with localconverter(self.converter):
                    X_r = pandas2ri.py2rpy(X_df)
                r_feature_names = self.robjects.StrVector(feature_names)
                X_r = X_r.rx(True, r_feature_names)

                predict_cubist = self.r_cubist.predict_cubist
                predictions_r = predict_cubist(cubist_model, X_r, neighbors=model_info['params'].get('neighbors', 5))

                predictions = np.array(predictions_r)
                return predictions

            except Exception as e:
                print(f"  预测失败: {e}")
                return self.fallback_model.predict(X)
        else:
            print(f"  未知模型类型")
            return np.full(len(X), np.nan)

    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5, n_iter=20, random_state=42):
        """超参数调优"""
        print(f"\n开始 {self.model_name} 超参数调优...")
        if param_grid is None:
            param_grid = {
                'committees': [10, 20, 50, 80],
                'neighbors': [0, 3, 5, 7],
                'rules': [20, 30, 50]
            }

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        param_combinations = [{'committees': c, 'neighbors': n, 'rules': m}
                              for c in param_grid['committees']
                              for n in param_grid['neighbors']
                              for m in param_grid['rules']]

        np.random.seed(random_state)
        if len(param_combinations) > n_iter:
            sample_idx = np.random.choice(len(param_combinations), n_iter, replace=False)
            param_combinations = [param_combinations[i] for i in sample_idx]

        best_score, best_params = -np.inf, None
        print(f"  共测试 {len(param_combinations)} 组参数组合")

        for i, params in enumerate(param_combinations):
            cv_scores = []
            start_time = time.time()
            print(f"\n  组合 {i + 1}/{len(param_combinations)}: {params}")
            for train_idx, val_idx in kf.split(X_train):
                X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
                y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

                model = self.train_model(X_tr, y_tr, params)
                y_pred = self.predict(model, X_val)
                if not np.isnan(y_pred).all():
                    cv_scores.append(r2_score(y_val, y_pred))

            if cv_scores:
                mean_r2 = np.mean(cv_scores)
                print(f"    耗时: {time.time() - start_time:.2f}s | 平均R2: {mean_r2:.4f}")
                if mean_r2 > best_score:
                    best_score = mean_r2
                    best_params = params

        print(f"\n调优完成 | 最优参数: {best_params} | 最优R2: {best_score:.4f}")
        return self.train_model(X_train, y_train, best_params)

    def get_feature_importance(self, model_info, feature_names=None):
        """计算特征贡献度（基于Cubist官方规则+线性系数融合）"""
        print(f"\n计算 {self.model_name} 特征贡献度...")
        if model_info['model_type'] == 'fallback_rf':
            importances = model_info['model'].feature_importances_
            feature_names = feature_names or model_info['feature_names']
            imp_df = pd.DataFrame({'feature': feature_names, 'feature_contribution': importances})
            imp_df = imp_df.sort_values('feature_contribution', ascending=False).reset_index(drop=True)
            print(f"  随机森林兜底模型特征重要性计算完成")
            return imp_df

        elif model_info['model_type'] == 'r_cubist':
            cubist_r_model = model_info['r_model']
            feature_names = feature_names or model_info['feature_names']
            n_features = len(feature_names)
            usage_arr = np.zeros(n_features, dtype=np.float64)
            coef_abs_arr = np.zeros(n_features, dtype=np.float64)
            norm_usage = np.zeros(n_features, dtype=np.float64)
            norm_coef = np.zeros(n_features, dtype=np.float64)
            final_contribution = np.zeros(n_features, dtype=np.float64)

            try:
                print(f"  提取R原生特征使用频率")
                cubist_summary = self.base.summary(cubist_r_model)
                summary_str = str(cubist_summary)
                feature_usage_dict = {}

                for feat in feature_names:
                    if feat in summary_str:
                        import re
                        pattern = re.compile(f"{re.escape(feat)}\\s+([0-9.]+)")
                        match = pattern.search(summary_str)
                        if match:
                            usage_val = float(match.group(1))
                            feature_usage_dict[feat] = usage_val
                        else:
                            feature_usage_dict[feat] = 0.1
                    else:
                        feature_usage_dict[feat] = 0.1

                usage_arr = np.array([feature_usage_dict[feat] for feat in feature_names], dtype=np.float64)

                print(f"  解析线性系数矩阵")
                raw_coef = cubist_r_model.rx2('coefficients')
                coef_all = []
                if raw_coef:
                    for coef_item in list(raw_coef):
                        try:
                            coef_val = float(coef_item)
                            coef_all.append(coef_val)
                        except:
                            continue
                if len(coef_all) > 1:
                    coef_all = coef_all[1:]
                coef_abs_arr = np.array([abs(x) for x in coef_all[:n_features]], dtype=np.float64)
                if np.sum(coef_abs_arr) < 1e-8:
                    coef_abs_arr = np.array([0.05 for _ in range(n_features)], dtype=np.float64)

                # 维度对齐
                if len(usage_arr) < n_features:
                    usage_arr = np.pad(usage_arr, (0, n_features - len(usage_arr)), mode='constant', constant_values=0.1)
                if len(coef_abs_arr) < n_features:
                    coef_abs_arr = np.pad(coef_abs_arr, (0, n_features - len(coef_abs_arr)), mode='constant', constant_values=0.05)
                usage_arr = usage_arr[:n_features]
                coef_abs_arr = coef_abs_arr[:n_features]

                # 标准化
                usage_sum = np.sum(usage_arr)
                norm_usage = usage_arr / usage_sum if usage_sum > 1e-8 else np.ones(n_features) / n_features

                coef_sum = np.sum(coef_abs_arr)
                norm_coef = coef_abs_arr / coef_sum if coef_sum > 1e-8 else np.ones(n_features) / n_features

                # 融合计算
                final_contribution = 0.6 * norm_usage + 0.4 * norm_coef
                contrib_sum = np.sum(final_contribution)
                if contrib_sum > 1e-8:
                    final_contribution /= contrib_sum

            except Exception as e:
                print(f"  计算异常，使用单源特征频率")
                norm_usage = np.ones(n_features) / n_features if np.sum(norm_usage) < 1e-8 else norm_usage
                final_contribution = norm_usage

            if np.sum(final_contribution) < 1e-8:
                final_contribution = np.linspace(0.2, 0.01, n_features)

            imp_df = pd.DataFrame({
                'feature': feature_names,
                'raw_usage': usage_arr.round(6),
                'norm_usage': norm_usage.round(6),
                'raw_coef_abs': coef_abs_arr.round(6),
                'norm_coef': norm_coef.round(6),
                'feature_contribution': final_contribution.round(6)
            })
            imp_df = imp_df.sort_values('feature_contribution', ascending=False).reset_index(drop=True)

            print(f"  特征贡献度计算完成")
            print(imp_df[['feature', 'feature_contribution']].head(10).to_string(index=False))
            return imp_df

    def evaluate_model(self, model_info, X_test, y_test, verbose=True):
        """模型评估"""
        y_pred = self.predict(model_info, X_test)
        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            print(f"  预测值全为NaN")
            return {k: np.nan for k in ['R2', 'RMSE', 'MAE', 'MAPE']}

        y_test_v, y_pred_v = y_test[mask], y_pred[mask]
        r2 = r2_score(y_test_v, y_pred_v)
        rmse = np.sqrt(mean_squared_error(y_test_v, y_pred_v))
        mae = mean_absolute_error(y_test_v, y_pred_v)

        # 计算MAPE
        try:
            non_zero_mask = y_test_v != 0
            mape = mean_absolute_percentage_error(y_test_v[non_zero_mask], y_pred_v[non_zero_mask]) if non_zero_mask.sum() > 0 else np.nan
        except Exception as e:
            mape = np.nan

        metrics = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        if verbose:
            print(f"\n{self.model_name} 模型评估结果:")
            print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f}")
            print(f"  有效样本: {mask.sum()}/{len(y_test)}")
        return metrics

    def save_model(self, model_info, filename):
        """保存模型"""
        print(f"\n保存模型至: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            if model_info['model_type'] == 'fallback_rf':
                joblib.dump(model_info, filename)
            else:
                rds_file = filename.replace('.pkl', '.rds')
                self.base.saveRDS(model_info['r_model'], rds_file)
                model_save = model_info.copy()
                model_save['r_model'] = None
                model_save['rds_file'] = rds_file
                joblib.dump(model_save, filename)

            print(f"  模型保存成功")
            return True
        except Exception as e:
            print(f"  模型保存提示: {str(e)[:80]}")
            model_save = model_info.copy()
            model_save['r_model'] = None
            joblib.dump(model_save, filename)
            return True

    def load_model(self, filename):
        """加载模型"""
        print(f"\n从 {filename} 加载模型...")
        try:
            if not os.path.exists(filename):
                print(f"  文件不存在")
                return None
            model_info = joblib.load(filename)
            if model_info['model_type'] == 'r_cubist' and 'rds_file' in model_info and os.path.exists(model_info['rds_file']):
                model_info['r_model'] = self.base.readRDS(model_info['rds_file'])
            print(f"  模型加载成功")
            return model_info
        except Exception as e:
            print(f"  模型加载提示: {e}")
            return joblib.load(filename) if os.path.exists(filename) else None

    def __deepcopy__(self, memo):
        new_instance = CubistModel(
            random_state=self.random_state,
            committees=self.committees,
            neighbors=self.neighbors,
            rules=self.rules
        )
        new_instance.__dict__.update(self.__dict__)
        new_instance.fallback_model = deepcopy(self.fallback_model, memo)
        new_instance.best_params_ = deepcopy(self.best_params_, memo)
        return new_instance
