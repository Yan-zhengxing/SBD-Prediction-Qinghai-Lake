"""
XGBoost回归模型模块
"""

import numpy as np

try:
    import xgboost as xgb

    XGB_INSTALLED = True
except ImportError:
    print("警告: xgboost未安装，请运行: pip install xgboost")
    XGB_INSTALLED = False
    xgb = None

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class XGBoostModel:
    """XGBoost回归模型封装类"""

    def __init__(self, n_jobs=-1, random_state=42):
        """
        初始化XGBoost模型

        参数:
            n_jobs: 并行作业数
            random_state: 随机种子
        """
        if not XGB_INSTALLED:
            raise ImportError("xgboost未安装，请先安装: pip install xgboost")

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params_ = None
        self.feature_importances_ = None
        self.model_name = "XGBoost"

        print(f"初始化 {self.model_name} 模型")

    def train_model(self, X_train, y_train, params=None):
        """
        训练XGBoost模型

        参数:
            X_train: 训练特征
            y_train: 训练目标
            params: 模型参数

        返回:
            训练好的模型
        """
        print(f"训练 {self.model_name} 模型...")
        print(f"  训练样本: {len(X_train)}")
        print(f"  特征数量: {X_train.shape[1]}")

        # 设置默认参数
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbosity': 0
            }

        # 创建模型
        model = xgb.XGBRegressor(**params)

        # 训练模型
        model.fit(X_train, y_train)

        # 保存特征重要性
        self.feature_importances_ = model.feature_importances_

        # 保存最佳参数
        self.best_params_ = params

        print(f"  {self.model_name} 模型训练完成")

        return model

    def predict(self, model, X):
        """
        使用训练好的模型进行预测
        """
        return model.predict(X)

    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5, n_iter=50, random_state=42):
        """
        调优XGBoost超参数
        """
        print(f"调优 {self.model_name} 超参数...")

        # 默认参数网格
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [1, 1.5, 2]
            }

        # 创建基础模型
        base_model = xgb.XGBRegressor(
            random_state=random_state,
            n_jobs=self.n_jobs,
            verbosity=0
        )

        # 使用随机搜索
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=self.n_jobs,
            random_state=random_state,
            verbose=1
        )

        # 执行搜索
        search.fit(X_train, y_train)

        print(f"  最佳参数: {search.best_params_}")
        print(f"  最佳R²分数: {search.best_score_:.3f}")

        # 保存最佳参数
        self.best_params_ = search.best_params_

        return search.best_estimator_

    def get_feature_importance(self, model, feature_names=None, importance_type='weight'):
        """
        获取特征重要性

        参数:
            model: 训练好的模型
            feature_names: 特征名称列表
            importance_type: 重要性类型 ('weight', 'gain', 'cover', 'total_gain', 'total_cover')

        返回:
            特征重要性DataFrame
        """
        if hasattr(model, 'get_booster'):
            # 获取重要性
            importance_dict = model.get_booster().get_score(importance_type=importance_type)

            # 转换为列表
            importances = []
            features = []

            for i, (feature, importance) in enumerate(importance_dict.items()):
                # 移除特征名的"f"前缀
                if feature.startswith('f'):
                    try:
                        idx = int(feature[1:])
                        if feature_names and idx < len(feature_names):
                            features.append(feature_names[idx])
                        else:
                            features.append(feature)
                    except:
                        features.append(feature)
                else:
                    features.append(feature)

                importances.append(importance)

            # 创建DataFrame
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            })

            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)

            return importance_df
        else:
            print(f"警告: 无法获取特征重要性")
            return None

    def evaluate_model(self, model, X_test, y_test):
        """
        评估模型性能
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        metrics = {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'n_samples': len(y_test)
        }

        print(f"{self.model_name} 评估结果:")
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  样本数: {len(y_test)}")

        return metrics

    def plot_learning_curve(self, model, X_train, y_train, X_test, y_test, save_path=None):
        """
        绘制学习曲线
        """
        import matplotlib.pyplot as plt

        # 获取模型迭代过程中的评估结果
        if hasattr(model, 'evals_result'):
            evals_result = model.evals_result()

            plt.figure(figsize=(10, 6))

            # 绘制训练集误差
            if 'validation_0' in evals_result:
                train_error = evals_result['validation_0']['rmse']
                plt.plot(range(1, len(train_error) + 1), train_error,
                         label='Training RMSE', linewidth=2)

            # 绘制验证集误差
            if 'validation_1' in evals_result:
                val_error = evals_result['validation_1']['rmse']
                plt.plot(range(1, len(val_error) + 1), val_error,
                         label='Validation RMSE', linewidth=2)

            plt.xlabel('Boosting Rounds')
            plt.ylabel('RMSE')
            plt.title('XGBoost Learning Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"学习曲线已保存: {save_path}")

            plt.show()
        else:
            print("无法获取学习曲线数据")


def test_xgb_model():
    """测试XGBoost模型"""
    if not XGB_INSTALLED:
        print("跳过XGBoost测试: 未安装xgboost")
        return None, None

    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.5

    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建模型实例
    xgb_model = XGBoostModel()

    # 训练模型
    model = xgb_model.train_model(X_train, y_train)

    # 评估模型
    metrics = xgb_model.evaluate_model(model, X_test, y_test)

    # 获取特征重要性
    feature_names = [f'X{i}' for i in range(n_features)]
    importance_df = xgb_model.get_feature_importance(model, feature_names)

    print("\n特征重要性:")
    print(importance_df)

    return model, metrics


if __name__ == "__main__":
    print("测试XGBoost模型...")
    model, metrics = test_xgb_model()
    if model:
        print(f"\n测试完成! R² = {metrics['R2']:.3f}")
