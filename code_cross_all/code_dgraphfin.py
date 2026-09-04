import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import gc
import psutil, os
import time
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import time

# ========= 核心交叉验证（内存精简版） =========
def cv_model(clf, train_x, train_y, test_x, clf_name, train_y_2, sd, cols_cat=None, folds=10, x_valid=None):
    infer_time_sum = 0
    seed = sd
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros((train_x.shape[0], 2))
    test = np.zeros((test_x.shape[0], 2))
    valid = np.zeros((x_valid.shape[0], 2)) if x_valid is not None else None
    
    cv_auc, cv_f1 = [], []
    inference_times = []  # 存储每折的推理时间
    gpu_memory_usage = []  # 存储每折的GPU内存使用情况
    feature_importance_df = pd.DataFrame()
    if cols_cat is None:
        cols_cat = []

    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y_2)):
        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f'******** {clf_name} fold {i+1} ********  [Mem] {mem_start:.2f} GiB')

        trn_x, trn_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
        val_x, val_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

        # ---- 1. LightGBM (优化版) ----
        if clf_name == "lgb":
            import lightgbm as lgb
            
            # 计算类别权重
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(trn_y)
            class_weights = compute_class_weight('balanced', classes=classes, y=trn_y)
            weight_dict = dict(zip(classes, class_weights))
            scale_pos_weight = weight_dict[1] / weight_dict[0]
            
            categorical_feature = [c for c in cols_cat if c in train_x.columns]
            trn_data = lgb.Dataset(trn_x, label=trn_y, categorical_feature=categorical_feature, free_raw_data=False)
            val_data = lgb.Dataset(val_x, label=val_y, categorical_feature=categorical_feature, free_raw_data=False)

            # 优化后的参数
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': ['binary_error'],
                
                # 学习率和树结构
                'learning_rate': 0.05,
                'num_leaves': 63,
                'max_depth': -1,
                'min_child_samples': 20,
                'min_child_weight': 0.001,
                
                # 正则化参数
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                
                # 类别不平衡处理
                'is_unbalance': True,
                
                # 性能优化
                'n_jobs': -1,
                'seed': 2022,
                'verbose': -1,
                'feature_pre_filter': False
            }
            if categorical_feature:
                params['categorical_feature'] = categorical_feature

            model = lgb.train(params, trn_data, 2000,
                              valid_sets=[trn_data, val_data],
                              valid_names=['train', 'valid'],
                              callbacks=[
                                  lgb.early_stopping(stopping_rounds=200, verbose=True),
                                  lgb.log_evaluation(500)
                              ])
            
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            tst_pred = model.predict(test_x, num_iteration=model.best_iteration)
            vld_pred = model.predict(x_valid, num_iteration=model.best_iteration) if x_valid is not None else None

            fold_importance_df = pd.DataFrame({"Feature": train_x.columns,
                                               "importance": model.feature_importance(importance_type='gain'),
                                               "fold": i + 1})
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            if model.params.get('device') == 'gpu':
                lgb.model_free(model)

            # ---- 结果累加 + 指标 ----
            train[va_idx, 0] = 1 - val_pred
            train[va_idx, 1] = val_pred
            test[:, 0] += (1 - tst_pred) / folds
            test[:, 1] += tst_pred / folds
            if vld_pred is not None:
                valid[:, 0] += (1 - vld_pred) / folds
                valid[:, 1] += vld_pred / folds
            
            val_pos_prob = val_pred
            val_prd_lbl = (val_pos_prob > 0.5).astype(int)
            true_binary = (val_y == 1).astype(int)
            cv_auc.append(roc_auc_score(true_binary, val_pos_prob))
            cv_f1.append(f1_score(true_binary, val_prd_lbl))
            print(f'  fold AUC: {cv_auc[-1]:.4f}  F1: {cv_f1[-1]:.4f}')

            # ===== 内存清理 =====
            del trn_data, val_data, model, val_pred, tst_pred, vld_pred, fold_importance_df

        # ---- 2. XGBoost (GPU版本) ----
        elif clf_name == "xgb":
            import xgboost as xgb
            
            # ===== 修复：添加数据预处理函数 =====
            def preprocess_xgboost_data(data):
                """预处理数据，处理inf、nan和过大值"""
                data_processed = data.copy()
                
                # 替换无穷大值为NaN
                data_processed = data_processed.replace([np.inf, -np.inf], np.nan)
                
                # 处理超出合理范围的值
                threshold = 1e6  # 根据你的数据特性调整这个阈值
                data_processed[data_processed > threshold] = np.nan
                data_processed[data_processed < -threshold] = np.nan
                
                return data_processed
            
            # 预处理所有数据
            trn_x_processed = preprocess_xgboost_data(trn_x)
            val_x_processed = preprocess_xgboost_data(val_x)
            test_x_processed = preprocess_xgboost_data(test_x)
            x_valid_processed = preprocess_xgboost_data(x_valid) if x_valid is not None else None
            
            # 设置missing参数为np.nan，让XGBoost自动处理缺失值
            trn_data = xgb.DMatrix(trn_x_processed, label=trn_y, 
                                 enable_categorical=False, missing=np.nan)
            val_data = xgb.DMatrix(val_x_processed, label=val_y, 
                                 enable_categorical=False, missing=np.nan)
            tst_data = xgb.DMatrix(test_x_processed, 
                                 enable_categorical=False, missing=np.nan)
            vld_data = xgb.DMatrix(x_valid_processed, 
                                 enable_categorical=False, missing=np.nan) if x_valid_processed is not None else None

            # 使用GPU加速的参数 - XGBoost 3.1.1版本
            params = {
                'booster': 'gbtree', 
                'objective': 'binary:logistic',  # 二分类问题
                'eval_metric': ['auc', 'logloss'],  # 使用多个评估指标
                'learning_rate': 0.04,  # XGBoost 3.x使用learning_rate替代eta
                'seed': 2020, 
                'verbosity': 1,
                # 树参数
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                # 缺失值处理
                'missing': np.nan,
                # GPU加速参数 - XGBoost 3.1.1版本
                'tree_method': 'hist',  # 使用hist算法
                'device': 'cuda:0',  # XGBoost 3.x使用device参数
                # 处理类别不平衡
                'scale_pos_weight': len(trn_y[trn_y==0]) / len(trn_y[trn_y==1]) if len(trn_y[trn_y==1]) > 0 else 1,
                # 性能优化
                'nthread': -1,  # 使用所有CPU线程
                'predictor': 'gpu_predictor'  # GPU预测
            }
            
            print(f"  XGBoost using GPU (device: cuda:0)")
            
            try:
                model = xgb.train(params, trn_data, 50000,
                                  evals=[(trn_data, 'train'), (val_data, 'eval')],
                                  verbose_eval=500, 
                                  early_stopping_rounds=500,
                                  maximize=True)  # AUC需要最大化
            except Exception as e:
                print(f"  GPU训练失败，尝试CPU训练: {e}")
                # 如果GPU失败，回退到CPU
                params['device'] = 'cpu'
                params['predictor'] = 'cpu_predictor'
                print(f"  Fallback to CPU training")
                model = xgb.train(params, trn_data, 50000,
                                  evals=[(trn_data, 'train'), (val_data, 'eval')],
                                  verbose_eval=500, 
                                  early_stopping_rounds=500,
                                  maximize=True)
            
            # 推理时间统计开始
            inference_start = time.time()
            val_pred = model.predict(val_data, iteration_range=(0, model.best_iteration))
            tst_pred = model.predict(tst_data, iteration_range=(0, model.best_iteration))
            vld_pred = model.predict(vld_data, iteration_range=(0, model.best_iteration)) if vld_data is not None else None
            inference_end = time.time()
            
            # 记录推理时间
            fold_inference_time = inference_end - inference_start
            inference_times.append(fold_inference_time)
            
            # GPU内存统计
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_mem_mb = float(result.stdout.strip().split('\n')[0])
                    gpu_memory_usage.append(gpu_mem_mb / 1024)  # 转换为GB
                    print(f"  GPU Memory Usage: {gpu_mem_mb/1024:.2f} GB")
            except:
                print("  GPU memory info not available")

            importance_dict = model.get_score(importance_type='gain')
            fold_importance_df = pd.DataFrame({
                "Feature": train_x.columns,
                "importance": [importance_dict.get(f, 0) for f in train_x.columns],
                "fold": i + 1
            })
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            # ---- 结果累加 + 指标 ----
            train[va_idx, 0] = 1 - val_pred
            train[va_idx, 1] = val_pred
            test[:, 0] += (1 - tst_pred) / folds
            test[:, 1] += tst_pred / folds
            if vld_pred is not None:
                valid[:, 0] += (1 - vld_pred) / folds
                valid[:, 1] += vld_pred / folds
                
            val_pos_prob = val_pred
            val_prd_lbl = (val_pos_prob > 0.5).astype(int)
            true_binary = (val_y == 1).astype(int)
            cv_auc.append(roc_auc_score(true_binary, val_pos_prob))
            cv_f1.append(f1_score(true_binary, val_prd_lbl))
            print(f'  fold AUC: {cv_auc[-1]:.4f}  F1: {cv_f1[-1]:.4f}')
            print(f'  Inference Time: {fold_inference_time:.4f} seconds')

            # ===== 内存清理 =====
            del trn_x_processed, val_x_processed, test_x_processed, x_valid_processed
            del trn_data, val_data, tst_data, vld_data, model, val_pred, tst_pred, vld_pred, fold_importance_df



        # ---- 3. CatBoost ----
        elif clf_name == "cab":
            from catboost import CatBoostClassifier, Pool
            cat_idx = [train_x.columns.get_loc(c) for c in cols_cat if c in train_x.columns]
            train_pool = Pool(trn_x, label=trn_y, cat_features=cat_idx)
            valid_pool = Pool(val_x, label=val_y, cat_features=cat_idx)
            test_pool = Pool(test_x, cat_features=cat_idx)
            vld_pool = Pool(x_valid, cat_features=cat_idx) if x_valid is not None else None

            model = CatBoostClassifier(
                iterations=2500,
                learning_rate=0.05,
                depth=5,
                l2_leaf_reg=3,
                border_count=254,
                bootstrap_type='Bernoulli',
                subsample=0.85,
                sampling_frequency='PerTree',
                grow_policy='SymmetricTree',
                random_strength=1.5,
                auto_class_weights='Balanced',
                loss_function='MultiClass',
                eval_metric='AUC',
                task_type='CPU',  # 使用GPU
                early_stopping_rounds=200,
                verbose=100,
                random_seed=2022
            )

            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            val_pred = model.predict_proba(valid_pool)
            inference_start = time.time()
            tst_pred = model.predict_proba(test_pool)
            inference_end = time.time()
            fold_inference_time = inference_end - inference_start
            print('infer time is {}'.format(round(fold_inference_time, 4)))
            infer_time_sum += fold_inference_time
            vld_pred = model.predict_proba(vld_pool) if vld_pool is not None else None
            
            
            # GPU内存统计 (CatBoost没有直接的GPU内存查询接口)
            if hasattr(model, 'get_metadata') and 'device' in model.get_metadata():
                print(f"  Using GPU for inference")
                # CatBoost没有直接查询GPU内存的API，这里输出GPU使用状态
                import subprocess
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        gpu_mem_mb = float(result.stdout.strip().split('\n')[0])
                        gpu_memory_usage.append(gpu_mem_mb / 1024)  # 转换为GB
                        print(f"  GPU Memory Usage: {gpu_mem_mb/1024:.2f} GB")
                except:
                    print("  GPU memory info not available")

            fold_importance_df = pd.DataFrame({
                "Feature": train_x.columns,
                "importance": model.get_feature_importance(),
                "fold": i + 1
            })
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            # ---- 结果累加 + 指标 ----
            train[va_idx] = val_pred
            test += tst_pred / folds
            if vld_pred is not None:
                valid += vld_pred / folds
                
            val_pos_prob = val_pred[:, 1]
            val_prd_lbl = (val_pos_prob > 0.5).astype(int)
            true_binary = (val_y == 1).astype(int)
            cv_auc.append(roc_auc_score(true_binary, val_pos_prob))
            cv_f1.append(f1_score(true_binary, val_prd_lbl))
            print(f'  fold AUC: {cv_auc[-1]:.4f}  F1: {cv_f1[-1]:.4f}')
            print(f'  Inference Time: {fold_inference_time:.4f} seconds')

            # ===== 内存清理 =====
            del train_pool, valid_pool, test_pool, vld_pool, model, val_pred, tst_pred, vld_pred, fold_importance_df

        # ===== 公共清理 =====
        del trn_x, trn_y, val_x, val_y
        gc.collect()
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f'[Mem] 释放后 {mem_end:.2f} GiB  ↓ {mem_start - mem_end:.2f} GiB\n')

    # 最终统计 - 包括推理时间和GPU使用
    auc_mean, auc_std = np.mean(cv_auc), np.std(cv_auc)
    f1_mean, f1_std = np.mean(cv_f1), np.std(cv_f1)
    
    # 推理时间统计
    if inference_times:
        inf_time_mean, inf_time_std = np.mean(inference_times), np.std(inference_times)
        inf_time_min, inf_time_max = np.min(inference_times), np.max(inference_times)
        print(f'\n===== Inference Time Statistics ({clf_name}) =====')
        print(f'Mean Inference Time: {inf_time_mean:.4f} ± {inf_time_std:.4f} seconds')
        print(f'Min Inference Time:  {inf_time_min:.4f} seconds')
        print(f'Max Inference Time:  {inf_time_max:.4f} seconds')
        print(f'Total Inference Time: {sum(inference_times):.4f} seconds')
    
    # GPU内存统计
    if gpu_memory_usage:
        gpu_mem_mean, gpu_mem_std = np.mean(gpu_memory_usage), np.std(gpu_memory_usage)
        gpu_mem_min, gpu_mem_max = np.min(gpu_memory_usage), np.max(gpu_memory_usage)
        print(f'\n===== GPU Memory Statistics ({clf_name}) =====')
        print(f'Mean GPU Memory: {gpu_mem_mean:.2f} ± {gpu_mem_std:.2f} GB')
        print(f'Min GPU Memory:  {gpu_mem_min:.2f} GB')
        print(f'Max GPU Memory:  {gpu_mem_max:.2f} GB')
    
    print('\n===== Model Performance =====')
    print('auc list is: ', cv_auc)
    print(f'{clf_name}_auc_mean: {auc_mean:.4f} ± {auc_std:.4f}')
    print(f'{clf_name}_f1_mean:  {f1_mean:.4f} ± {f1_std:.4f}')
    print('infer_time_sum is {}'.format(infer_time_sum))

    if not feature_importance_df.empty:
        importance_summary = (feature_importance_df.groupby("Feature")["importance"]
                              .mean().sort_values(ascending=False).reset_index())
        return train, valid, test, importance_summary, cv_auc, auc_mean, auc_std, cv_f1, f1_mean, f1_std
    return train, test, valid, None, cv_auc, auc_mean, auc_std, cv_f1, f1_mean, f1_std

# ========= 三模型封装 =========
def lgb_model(x_train, y_train, x_test, train_y_2, sd, cols_cat=None, folds=5, x_valid=None):
    return cv_model('lgb', x_train, y_train, x_test, "lgb", train_y_2, sd, cols_cat, folds=folds, x_valid=x_valid)

def xgb_model(x_train, y_train, x_test, train_y_2, sd, cols_cat=None, folds=5, x_valid=None):
    return cv_model('xgb', x_train, y_train, x_test, "xgb", train_y_2, sd, cols_cat, folds=folds, x_valid=x_valid)

def cab_model(x_train, y_train, x_test, train_y_2, sd, cols_cat=None, folds=5, x_valid=None):
    return cv_model('cab', x_train, y_train, x_test, "cab", train_y_2, sd, cols_cat, folds=folds, x_valid=x_valid)





# In[8]:


# path_data = '../data/dgraph.npz'
data_np = np.load('../data/dgraphfin_cross.npz')
print([f for f in data_np.keys()])

df_node_enhanced_np = data_np['df_node_enhanced']
cols_keep = data_np['cols_keep']
train_mask = data_np['train_mask']
valid_mask = data_np['valid_mask']
test_mask = data_np['test_mask']

df_node_enhanced = pd.DataFrame(df_node_enhanced_np, columns = cols_keep)


# In[9]:


# 准备训练测试数据
print("\n📋 准备训练测试数据...")
df_train = df_node_enhanced.iloc[train_mask, :].reset_index(drop=True)
df_valid = df_node_enhanced.iloc[valid_mask, :].reset_index(drop=True)
df_test = df_node_enhanced.iloc[test_mask, :].reset_index(drop=True)

del df_node_enhanced
[gc.collect() for _ in range(5)]

mem_gib = psutil.Process(os.getpid()).memory_info().rss / 1024**3
print(f"当前进程物理内存: {mem_gib:.2f} GiB")


# In[12]:

cols_info = ['label', 'node_id', 'timestamp']
cols_fea = [f for f in df_train.columns if ( (f not in cols_info)
            and ('recent_activity' not in f) 
            and ('risk_cluster' not in f) ) ]
print(len(cols_fea), cols_fea)

# In[13]:
import pandas as pd
import numpy as np

def create_cross_features(df):
    """
    基于特征重要性排序创建16个精炼交叉特征
    特征重要性：越靠前的特征越重要
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    df = df.copy()
    new_features = []
    
    # 只取前20个最重要的特征进行组合（基于提供的排序）
    top_features = [
        'fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11',
        'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6',
        'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean',
        'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0',
        'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8'
    ]
    
    # 1. 核心时间窗口分析：最后活跃时间与最早交易时间的间隔
    if 'last_active_out' in df.columns and 'undirected_1hop_timestamp_min' in df.columns:
        df['active_window_span'] = df['last_active_out'] - df['undirected_1hop_timestamp_min']
        new_features.append('active_window_span')
    
    # 2. 高风险交易密度：最大交易额与时间跨度的比值
    if 'undirected_1hop_f16_max' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['risk_transaction_density'] = df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6)
        new_features.append('risk_transaction_density')
    
    # 3. 近期突击性交易：最后活跃时间很近 + 交易额巨大
    if 'last_active_out' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['recent_burst_risk'] = (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max']
        new_features.append('recent_burst_risk')
    
    # 4. 交易时间异常：出账时间范围异常 + 交易间隔不稳定
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['timing_anomaly_score'] = df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std']
        new_features.append('timing_anomaly_score')
    
    # 5. 核心特征组合：最重要的3个特征x的交互
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
        df['core_feature_interaction'] = df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6)
        new_features.append('core_feature_interaction')
    
    # 6. 账户不活跃风险：不活跃天数与交易额的异常组合
    if 'inactive_days_out' in df.columns and 'undirected_1hop_f15_max' in df.columns:
        df['dormant_high_value_risk'] = df['inactive_days_out'] * df['undirected_1hop_f15_max']
        new_features.append('dormant_high_value_risk')
    
    # 7. 交易时间集中度：出账时间最小值和最大值的相对关系
    if 'directed_1hop_out_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_max' in df.columns:
        df['out_timestamp_concentration'] = 1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1)
        new_features.append('out_timestamp_concentration')
    
    # 8. 边类型异常：均值与最大值的差异
    if 'fea_out_edge_type_mean' in df.columns and 'fea_out_edge_type_max' in df.columns:
        df['edge_type_discrepancy'] = df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean']
        new_features.append('edge_type_discrepancy')
    
    # 9. 时间戳最小值异常：最早交易时间的异常模式
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_min' in df.columns:
        df['first_interaction_gap'] = df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min']
        new_features.append('first_interaction_gap')
    
    # 10. 特征x的加权组合：基于重要性排序的权重
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
        # 权重分配：重要性越高的特征权重越大
        df['weighted_x_features'] = (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                                    df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                                    df['fea_x_8'] * 0.05)
        new_features.append('weighted_x_features')
    
    # 11. 交易稳定性评分：时间范围与间隔稳定性的综合
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['transaction_stability'] = 1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1)
        new_features.append('transaction_stability')
    
    # 12. 高频风险交易：时间跨度小但交易额大
    if 'directed_1hop_out_timestamp_range' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['high_freq_high_value'] = df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1)
        new_features.append('high_freq_high_value')
    
    # 13. 账户活性异常：最后活跃与不活跃天数的矛盾
    if 'last_active_out' in df.columns and 'inactive_days_out' in df.columns:
        df['activity_inconsistency'] = df['last_active_out'] * df['inactive_days_out']
        new_features.append('activity_inconsistency')
    
    # 14. 时间模式异常：最早交易时间与出账时间范围的交互
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['early_wide_spread_risk'] = df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range'])
        new_features.append('early_wide_spread_risk')
    
    # 15. 多重风险叠加：结合多个最重要特征的异常模式
    if all(f in df.columns for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
        df['multi_risk_composite'] = (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                                     np.log1p(df['undirected_1hop_f16_max']))
        new_features.append('multi_risk_composite')
    
    # 16. 核心时间特征交互：最重要时间特征的组合
    if all(f in df.columns for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
        df['core_time_interaction'] = (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * \
                                     (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1))
        new_features.append('core_time_interaction')
    
    # 确保正好16个特征
    if len(new_features) > 16:
        # 保留前16个特征（按定义的顺序）
        features_to_keep = new_features[:16]
        features_to_remove = new_features[16:]
        
        for feat in features_to_remove:
            if feat in df.columns:
                df.drop(columns=[feat], inplace=True)
        
        new_features = features_to_keep
    
    print(f"基于重要性排序创建 {len(new_features)} 个精炼交叉特征")
    print("新特征列表（基于重要性前20的特征组合）:")
    for i, feat in enumerate(new_features, 1):
        print(f"{i:2d}. {feat}")
    
    return df, new_features

# 示例使用
# 假设 df 是包含原始特征的数据框
# df, new_features = create_cross_features(df)


# In[14]:


# 使用示例:
df_train, new_features = create_cross_features(df_train)
df_valid, _ = create_cross_features(df_valid)
df_test, _ = create_cross_features(df_test)


# In[15]:


cols_fea_cross = cols_fea + new_features[:]

print(len(cols_fea_cross), cols_fea_cross)


# In[23]:


x_train = df_train[cols_fea_cross].reset_index(drop=True)
x_valid = df_valid[cols_fea_cross].reset_index(drop=True)
x_test = df_test[cols_fea_cross].reset_index(drop=True)

y_train = df_train['label'].reset_index(drop=True)
y_valid = df_valid['label'].reset_index(drop=True)
y_test = df_test['label'].reset_index(drop=True)
train_y_2 = y_train.copy()


sd = 5
cab_train, cab_valid, cab_test, cab_top_feature, cab_auc_list, cab_auc_mean, cab_auc_std, cab_f1_list, cab_f1_mean, cab_f1_std = cab_model(x_train, y_train, x_test, train_y_2,sd,cols_cat=None, folds=5, x_valid=x_valid)

stack_train = cab_train[:, 1]
auc_train = roc_auc_score(y_train, stack_train)

stack_valid = cab_valid[:, 1]
auc_valid = roc_auc_score(y_valid, stack_valid)

stack_test = cab_test[:, 1]
auc_test = roc_auc_score(y_test, stack_test)

ap_train = average_precision_score(y_train, stack_train)
ap_valid = average_precision_score(y_valid, stack_valid)  # 修正：使用y_valid
ap_test = average_precision_score(y_test, stack_test)

print(f"Train - AUC: {auc_train:.4f}, AP: {ap_train:.4f}")
print(f"Valid - AUC: {auc_valid:.4f}, AP: {ap_valid:.4f}")
print(f"Test  - AUC: {auc_test:.4f}, AP: {ap_test:.4f}")


# In[ ]:





