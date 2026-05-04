"""
配置文件 - 空间预测模型通用配置 (特征工程优化版，支持植被指数接入+核心优化策略)
"""

# ==================== 数据路径和文件 ====================
DATA_PATH = './data'  # 数据目录
TRAIN_FILE = 'train_data.xls'  # 训练数据文件

# ==================== 数据列配置 ====================
# 目标变量列
TARGET_COLS = ['BD0_10', 'BD10_20', 'BD20_30']

# 坐标列
COORD_COLS = ['POINT_X', 'POINT_Y']

# 基础固定协变量列
FIXED_COVARIATES = ['DEM', 'Tem', 'Pre', 'SM_szj', 'TWI', 'Year']

# 植被指数核心配置 - 支持EVI/NDVI二选一
VEG_INDEX_CONFIG = {
    'enable': True,
    'selected_index': None,
    'evi_col': 'EVI_szj_23',
    'ndvi_col': 'NDVI_szj',
    'is_core_feature': True
}

# 可选协变量
OPTIONAL_COVARIATES = ['NDVI_szj', 'EVI_szj_23']

# Year编码映射
YEAR_ENCODING = {
    2023: 0,
    2024: 1
}

# 土壤属性列（按深度分层）
SOIL_PROPERTIES = {
    '0_10': ['soc0_10', 'sand0_10', 'clay0_10'],
    '10_20': ['soc10_20', 'sand10_20', 'clay10_20'],
    '20_30': ['soc20_30', 'sand20_30', 'clay20_30']
}

# 土壤交互特征配置
INTERACTION_FEATURES = {
    'enable': True,
    'depth_list': ['0_10', '10_20', '30_30'],
    'soil_interaction': [('soc', 'sand'), ('sand', 'clay')],
    'geo_interaction': [('DEM', 'TWI'), ('DEM', 'slope')],
    'veg_interaction': [('veg_index', 'SM_szj'), ('veg_index', 'soc'), ('veg_index', 'sand')]
}

# 特征筛选策略-按土层差异化配置
FEATURE_FILTER_CONFIG = {
    '0_10': {'enable_filter': True, 'max_remove': 3, 'exclude_cols': []},
    '10_20': {'enable_filter': True, 'max_remove': 2, 'exclude_cols': []},
    '20_30': {'enable_filter': False, 'max_remove': 0, 'exclude_cols': []}
}

# 特征重要性阈值配置
FEATURE_IMPORTANCE_THRESHOLD = {
    'threshold': 0.02,
    'depth_adjust': True,
    'veg_importance_boost': 0.01,
    'force_keep_veg': True
}

# ==================== 训练集划分配置 ====================
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'random_state': 42,
    'shuffle': True,
    'stratify_by_year': True,
    'stratify_by_dem': True,
    'dem_bins': 4
}

# ==================== 模型参数配置 ====================
# 随机森林参数
RF_PARAMS = {
    '默认参数': {
        'n_estimators': 220,
        'max_depth': 8,
        'min_samples_split': 7,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'random_state': 42,
        'n_jobs': -1
    },
    '调优参数': {
        'n_estimators': [200, 220, 250],
        'max_depth': [7, 8, 9],
        'min_samples_split': [6,7,8],
        'min_samples_leaf': [4,5,6],
        'max_features': ['sqrt']
    },
    'deep_soil_params': {
        'n_estimators': 250,
        'max_depth': 10,
        'min_samples_split': 6,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }
}

# XGBoost参数
XGB_PARAMS = {
    '默认参数': {
        'n_estimators': 250,
        'max_depth': 3,
        'learning_rate': 0.06,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.4,
        'reg_lambda': 1.2,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    },
    '调优参数': {
        'n_estimators': [200,250,300],
        'max_depth': [2,3],
        'learning_rate': [0.05,0.06,0.07],
        'subsample': [0.7,0.75],
        'colsample_bytree': [0.7,0.75],
        'reg_alpha': [0.3,0.4,0.5],
        'reg_lambda': [1.0,1.2]
    },
    'early_stopping': {
        'enable': True,
        'early_stopping_rounds': 20,
        'eval_metric': 'rmse'
    }
}

# Cubist参数
CUBIST_PARAMS = {
    '默认参数': {
        'committees': 12,
        'neighbors': 3,
        'rules': 22,
        'extrapolation': 0,
        'seed': 42,
        'unbiased': True,
        'sample': 0.8,
        'fuzzy': True,
        'label': 'none'
    },
    '调优参数': {
        'committees': [8,10,12],
        'neighbors': [4,5,6],
        'rules': [16,17,18,20],
        'unbiased': [True],
        'sample': [0.75,0.8],
        'fuzzy': [True]
    },
    # 分土层最优参数
    '分土层最优参数': {
        'BD0_10': {'committees':8, 'neighbors':3, 'rules':3, 'seed':42, 'unbiased':True, 'sample':0.80, 'fuzzy':False},
        'BD10_20': {'committees':10, 'neighbors':5, 'rules':2, 'seed':42, 'unbiased':True, 'sample':0.85, 'fuzzy':False},
        'BD20_30': {'committees':12,  'neighbors':7, 'rules':1, 'seed':42, 'unbiased':True, 'sample':0.90, 'fuzzy':False}
    }
}

# ==================== 交叉验证配置 ====================
CV_CONFIG = {
    '传统CV': {
        'test_size': 0.2,
        'random_state': 42
    },
    '空间CV': {
        'n_splits': 4,
        'default_block_size': 8,
        'random_state': 42,
        'shuffle_blocks': False
    }
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    'excel_format': 'xlsx',
    'figure_format': ['png', 'svg'],
    'dpi': 300,
    'figure_size': (12, 9),
    'save_shap_figure': True,
    'save_veg_importance': True
}

# ==================== 预测年份设置 ====================
PREDICTION_YEARS = {
    'training': 'both',
    'full_area': [0, 1],
    'year_weight': {0:0.5, 1:0.5}
}

# ==================== 植被指数生效开关 ====================
SELECTED_VEG_INDEX = 'NDVI_szj'
