"""
随机森林回归模型模块
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class RandomForestModel:
    """随机森林回归模型封装类"""

    def __init__(self, n_jobs=-1, random_state=42):
        """
        初始化随机森林模型

        参数:
            n_jobs: 并行作业数，-1表示使用所有CPU核心
            random_state: 随机种子
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params_ = None
        self.feature_importances_ = None
        self.model_name = "Random Forest"

        print(f"初始化 {self.model_name} 模型")

    def train_model(self, X_train, y_train, params=None):
        """
        训练随机森林模型

        参数:
            X_train: 训练特征
            y_train: 训练目标
            params: 模型参数，为None时使用默认参数

        返回:
            训练好的模型
        """
        print(f"训练 {self.model_name} 模型...")
        print(f"  训练样本: {len(X_train)}")
        print(f"  特征数量: {X_train.shape[1]}")

        # 设置参数
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }

        # 创建模型
        model = RandomForestRegressor(**params)

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
        模型预测

        参数:
            model: 训练好的模型
            X: 特征数据

        返回:
            预测值
        """
        return model.predict(X)

    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5, n_iter=50, random_state=42):
        """
        超参数调优

        参数:
            X_train: 训练特征
            y_train: 训练目标
            param_grid: 参数网格
            cv: 交叉验证折数
            n_iter: 随机搜索迭代次数
            random_state: 随机种子

        返回:
            最优模型
        """
        print(f"调优 {self.model_name} 超参数...")

        # 默认参数网格
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

        # 创建基础模型
        base_model = RandomForestRegressor(
            random_state=random_state,
            n_jobs=self.n_jobs
        )

        # 随机搜索调优
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

        search.fit(X_train, y_train)

        print(f"  最佳参数: {search.best_params_}")
        print(f"  最佳R²分数: {search.best_score_:.3f}")

        self.best_params_ = search.best_params_

        return search.best_estimator_

    def get_feature_importance(self, model, feature_names=None):
        """
        获取特征重要性

        参数:
            model: 训练好的模型
            feature_names: 特征名称列表

        返回:
            特征重要性DataFrame
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            return importance_df
        else:
            print(f"警告: 模型不支持特征重要性计算")
            return None

    def evaluate_model(self, model, X_test, y_test):
        """
        模型评估

        参数:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标

        返回:
            评估指标字典
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        y_pred = model.predict(X_test)

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


def test_rf_model():
    """测试随机森林模型"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.5

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestModel()
    model = rf_model.train_model(X_train, y_train)
    metrics = rf_model.evaluate_model(model, X_test, y_test)

    feature_names = [f'X{i}' for i in range(n_features)]
    importance_df = rf_model.get_feature_importance(model, feature_names)

    print("\n特征重要性:")
    print(importance_df)

    return model, metrics


if __name__ == "__main__":
    print("测试随机森林模型...")
    model, metrics = test_rf_model()
    print(f"\n测试完成! R² = {metrics['R2']:.3f}")
