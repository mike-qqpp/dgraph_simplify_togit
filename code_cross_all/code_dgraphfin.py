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

# == sync, corrected by elderman == @elder man
def cv_model(clf, train_x, train_y, test_x, clf_name, train_y_2, sd, cols_cat=None, folds=10, x_valid=None):
    infer_time_sum = 0
    seed = sd
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros((train_x.shape[0], 2))
    test = np.zeros((test_x.shape[0], 2))
    valid = np.zeros((x_valid.shape[0], 2)) if x_valid is not None else None
    
    cv_auc, cv_f1 = [], []
    inference_times = []  # Store every discount of reasoning time
    gpu_memory_usage = []  # Storage of each discount of GPS memory usage
    feature_importance_df = pd.DataFrame()
    if cols_cat is None:
        cols_cat = []

    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y_2)):
        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f'******** {clf_name} fold {i+1} ********  [Mem] {mem_start:.2f} GiB')

        trn_x, trn_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
        val_x, val_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

        # - 1. LightGBM (optimization) -
        if clf_name == "lgb":
            import lightgbm as lgb
            
            # Calculate category weights
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(trn_y)
            class_weights = compute_class_weight('balanced', classes=classes, y=trn_y)
            weight_dict = dict(zip(classes, class_weights))
            scale_pos_weight = weight_dict[1] / weight_dict[0]
            
            categorical_feature = [c for c in cols_cat if c in train_x.columns]
            trn_data = lgb.Dataset(trn_x, label=trn_y, categorical_feature=categorical_feature, free_raw_data=False)
            val_data = lgb.Dataset(val_x, label=val_y, categorical_feature=categorical_feature, free_raw_data=False)

            # Optimized parameters
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': ['binary_error'],
                
                # Learning rate and tree structure
                'learning_rate': 0.05,
                'num_leaves': 63,
                'max_depth': -1,
                'min_child_samples': 20,
                'min_child_weight': 0.001,
                
                # Regularize parameters
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                
                # Treatment of category imbalances
                'is_unbalance': True,
                
                # Performance optimization
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

            # - Result cumulative + indicator -
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

            # == sync, corrected by elderman == @elder man
            del trn_data, val_data, model, val_pred, tst_pred, vld_pred, fold_importance_df

        # - 2. XGBoost (GPU version) -
        elif clf_name == "xgb":
            import xgboost as xgb
            
            # == sync, corrected by elderman == @elder man
            def preprocess_xgboost_data(data):
                """Preprocessing data, processing inf, nan and excessive"""
                data_processed = data.copy()
                
                # Replace Infinite Great Value with NAN
                data_processed = data_processed.replace([np.inf, -np.inf], np.nan)
                
                # Processing values beyond reasonable range
                threshold = 1e6  # Adjust this threshold to your data features.
                data_processed[data_processed > threshold] = np.nan
                data_processed[data_processed < -threshold] = np.nan
                
                return data_processed
            
            # Preprocess all data
            trn_x_processed = preprocess_xgboost_data(trn_x)
            val_x_processed = preprocess_xgboost_data(val_x)
            test_x_processed = preprocess_xgboost_data(test_x)
            x_valid_processed = preprocess_xgboost_data(x_valid) if x_valid is not None else None
            
            # Sets the missing parameter to np.nan, allowing XGBoost to automatically process the missing value
            trn_data = xgb.DMatrix(trn_x_processed, label=trn_y, 
                                 enable_categorical=False, missing=np.nan)
            val_data = xgb.DMatrix(val_x_processed, label=val_y, 
                                 enable_categorical=False, missing=np.nan)
            tst_data = xgb.DMatrix(test_x_processed, 
                                 enable_categorical=False, missing=np.nan)
            vld_data = xgb.DMatrix(x_valid_processed, 
                                 enable_categorical=False, missing=np.nan) if x_valid_processed is not None else None

            # Using GPU Accelerating Parameters - XGBoost 3.1.1
            params = {
                'booster': 'gbtree', 
                'objective': 'binary:logistic',  # Classification issues
                'eval_metric': ['auc', 'logloss'],  # Use of multiple assessment indicators
                'learning_rate': 0.04,  # XGBoost 3.x Replace eta with learning rate
                'seed': 2020, 
                'verbosity': 1,
                # Tree Parameters
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                # Missing value processing
                'missing': np.nan,
                # GPU Accelerating Parameter - XGBoost 3.1.1
                'tree_method': 'hist',  # Use hist algorithm
                'device': 'cuda:0',  # XGBoost 3.x Use device parameters
                # Addressing category imbalances
                'scale_pos_weight': len(trn_y[trn_y==0]) / len(trn_y[trn_y==1]) if len(trn_y[trn_y==1]) > 0 else 1,
                # Performance optimization
                'nthread': -1,  # Use all CPU threads
                'predictor': 'gpu_predictor'  # GPU projections
            }
            
            print(f"  XGBoost using GPU (device: cuda:0)")
            
            try:
                model = xgb.train(params, trn_data, 50000,
                                  evals=[(trn_data, 'train'), (val_data, 'eval')],
                                  verbose_eval=500, 
                                  early_stopping_rounds=500,
                                  maximize=True)  # AUC needs to be maximized.
            except Exception as e:
                print(f"  GPUTraining failure，TryCPUtraining: {e}")
                # If GPU fails, back to CPU.
                params['device'] = 'cpu'
                params['predictor'] = 'cpu_predictor'
                print(f"  Fallback to CPU training")
                model = xgb.train(params, trn_data, 50000,
                                  evals=[(trn_data, 'train'), (val_data, 'eval')],
                                  verbose_eval=500, 
                                  early_stopping_rounds=500,
                                  maximize=True)
            
            # The reasoning time count begins.
            inference_start = time.time()
            val_pred = model.predict(val_data, iteration_range=(0, model.best_iteration))
            tst_pred = model.predict(tst_data, iteration_range=(0, model.best_iteration))
            vld_pred = model.predict(vld_data, iteration_range=(0, model.best_iteration)) if vld_data is not None else None
            inference_end = time.time()
            
            # Recording reasoning time
            fold_inference_time = inference_end - inference_start
            inference_times.append(fold_inference_time)
            
            # GPU Memory Statistics
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_mem_mb = float(result.stdout.strip().split('\n')[0])
                    gpu_memory_usage.append(gpu_mem_mb / 1024)  # Convert to GB
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

            # - Result cumulative + indicator -
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

            # == sync, corrected by elderman == @elder man
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
                task_type='CPU',  # Use GPU
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
            
            
            # GPU memory statistics (no direct GPU memory query interface for CatBoost)
            if hasattr(model, 'get_metadata') and 'device' in model.get_metadata():
                print(f"  Using GPU for inference")
                # CatBoost does not directly search for GPS memory API, output GPS usage status here
                import subprocess
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        gpu_mem_mb = float(result.stdout.strip().split('\n')[0])
                        gpu_memory_usage.append(gpu_mem_mb / 1024)  # Convert to GB
                        print(f"  GPU Memory Usage: {gpu_mem_mb/1024:.2f} GB")
                except:
                    print("  GPU memory info not available")

            fold_importance_df = pd.DataFrame({
                "Feature": train_x.columns,
                "importance": model.get_feature_importance(),
                "fold": i + 1
            })
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            # - Result cumulative + indicator -
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

            # == sync, corrected by elderman == @elder man
            del train_pool, valid_pool, test_pool, vld_pool, model, val_pred, tst_pred, vld_pred, fold_importance_df

        # == sync, corrected by elderman == @elder man
        del trn_x, trn_y, val_x, val_y
        gc.collect()
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f'[Mem] After release {mem_end:.2f} GiB  ↓ {mem_start - mem_end:.2f} GiB\n')

    # Final statistics - including reasoning time and GPU use
    auc_mean, auc_std = np.mean(cv_auc), np.std(cv_auc)
    f1_mean, f1_std = np.mean(cv_f1), np.std(cv_f1)
    
    # Logical Time Statistics
    if inference_times:
        inf_time_mean, inf_time_std = np.mean(inference_times), np.std(inference_times)
        inf_time_min, inf_time_max = np.min(inference_times), np.max(inference_times)
        print(f'\n===== Inference Time Statistics ({clf_name}) =====')
        print(f'Mean Inference Time: {inf_time_mean:.4f} ± {inf_time_std:.4f} seconds')
        print(f'Min Inference Time:  {inf_time_min:.4f} seconds')
        print(f'Max Inference Time:  {inf_time_max:.4f} seconds')
        print(f'Total Inference Time: {sum(inference_times):.4f} seconds')
    
    # GPU Memory Statistics
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

# == sync, corrected by elderman ==
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


# Prepare training test data
print("Prepare training test data...")
df_train = df_node_enhanced.iloc[train_mask, :].reset_index(drop=True)
df_valid = df_node_enhanced.iloc[valid_mask, :].reset_index(drop=True)
df_test = df_node_enhanced.iloc[test_mask, :].reset_index(drop=True)

del df_node_enhanced
[gc.collect() for _ in range(5)]

mem_gib = psutil.Process(os.getpid()).memory_info().rss / 1024**3
print(f"Current Process Physical Memory: {mem_gib:.2f} GiB")


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
    """Create 16 refined cross-cutting features based on the importance of character ranking
The importance of identity: the importance of the features ahead
Return: (data box with new features, list of new feature names)"""
    df = df.copy()
    new_features = []
    
    # Only the top 20 most important features are combined (based on the ranking provided)
    top_features = [
        'fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11',
        'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6',
        'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean',
        'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0',
        'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8'
    ]
    
    # Core time window analysis: interval between last active time and earliest transaction time
    if 'last_active_out' in df.columns and 'undirected_1hop_timestamp_min' in df.columns:
        df['active_window_span'] = df['last_active_out'] - df['undirected_1hop_timestamp_min']
        new_features.append('active_window_span')
    
    # 2. High-risk transaction density: ratio of maximum transaction value to time span
    if 'undirected_1hop_f16_max' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['risk_transaction_density'] = df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6)
        new_features.append('risk_transaction_density')
    
    # 3. Recent surprise trading: the last active time is close + the value of the transaction is huge
    if 'last_active_out' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['recent_burst_risk'] = (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max']
        new_features.append('recent_burst_risk')
    
    # 4. Transaction time anomalies: unusual time frames for the disbursement of accounts + unstable transaction intervals
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['timing_anomaly_score'] = df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std']
        new_features.append('timing_anomaly_score')
    
    # 5. Core identity mix: interaction of the three most important features x
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
        df['core_feature_interaction'] = df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6)
        new_features.append('core_feature_interaction')
    
    # 6. Account inactivity risk: abnormal combination of inactive days and transaction amounts
    if 'inactive_days_out' in df.columns and 'undirected_1hop_f15_max' in df.columns:
        df['dormant_high_value_risk'] = df['inactive_days_out'] * df['undirected_1hop_f15_max']
        new_features.append('dormant_high_value_risk')
    
    # 7. Transaction time concentration: the relationship between the minimum and maximum value at the time of payment
    if 'directed_1hop_out_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_max' in df.columns:
        df['out_timestamp_concentration'] = 1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1)
        new_features.append('out_timestamp_concentration')
    
    # 8. Edge type anomalies: difference between average and maximum
    if 'fea_out_edge_type_mean' in df.columns and 'fea_out_edge_type_max' in df.columns:
        df['edge_type_discrepancy'] = df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean']
        new_features.append('edge_type_discrepancy')
    
    # 9. Minimum time stamp abnormal: unusual pattern of the earliest transaction time
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_min' in df.columns:
        df['first_interaction_gap'] = df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min']
        new_features.append('first_interaction_gap')
    
    # 10. Weighted combination of feature x: weights based on ranking of importance
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
        # Distribution of weights: the more important the weight of features
        df['weighted_x_features'] = (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                                    df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                                    df['fea_x_8'] * 0.05)
        new_features.append('weighted_x_features')
    
    # 11. Transaction stability rating: a combination of time-frame and interval stability
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['transaction_stability'] = 1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1)
        new_features.append('transaction_stability')
    
    # 12. High-frequency risk trading: small time horizon but high transaction value
    if 'directed_1hop_out_timestamp_range' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['high_freq_high_value'] = df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1)
        new_features.append('high_freq_high_value')
    
    # 13. Account activity anomalies: conflict between last active and inactive days
    if 'last_active_out' in df.columns and 'inactive_days_out' in df.columns:
        df['activity_inconsistency'] = df['last_active_out'] * df['inactive_days_out']
        new_features.append('activity_inconsistency')
    
    # 14. Anomalous time patterns: interaction between the earliest transaction time and the time frame for the disbursement of accounts
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['early_wide_spread_risk'] = df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range'])
        new_features.append('early_wide_spread_risk')
    
    # 15. Multiple risk overlaps: abnormal patterns combining the most important features
    if all(f in df.columns for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
        df['multi_risk_composite'] = (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                                     np.log1p(df['undirected_1hop_f16_max']))
        new_features.append('multi_risk_composite')
    
    # 16. Core time features interactive: a combination of the most important time features
    if all(f in df.columns for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
        df['core_time_interaction'] = (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * \
                                     (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1))
        new_features.append('core_time_interaction')
    
    # Make sure it happens to be 16.
    if len(new_features) > 16:
        # Retain the first 16 features (in order of definition)
        features_to_keep = new_features[:16]
        features_to_remove = new_features[16:]
        
        for feat in features_to_remove:
            if feat in df.columns:
                df.drop(columns=[feat], inplace=True)
        
        new_features = features_to_keep
    
    print(f"Create Based on Importance Sorting {len(new_features)} A precise cross-cutting feature")
    print("New feature list (based on pre-material 20 feature combinations):")
    for i, feat in enumerate(new_features, 1):
        print(f"{i:2d}. {feat}")
    
    return df, new_features

# Example used
# Assume df is a data frame with original features
# df, new_features = create_cross_features(df)


# In[14]:


# Example:
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
ap_valid = average_precision_score(y_valid, stack_valid)  # Amendments: use y valid
ap_test = average_precision_score(y_test, stack_test)

print(f"Train - AUC: {auc_train:.4f}, AP: {ap_train:.4f}")
print(f"Valid - AUC: {auc_valid:.4f}, AP: {ap_valid:.4f}")
print(f"Test  - AUC: {auc_test:.4f}, AP: {ap_test:.4f}")


# In[ ]:

