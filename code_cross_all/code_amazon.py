#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import gc
import psutil, os
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

        # ---- 2. XGBoost ----
        elif clf_name == "xgb":
            import xgboost as xgb
            trn_data = xgb.DMatrix(trn_x, label=trn_y, enable_categorical=False)
            val_data = xgb.DMatrix(val_x, label=val_y, enable_categorical=False)
            tst_data = xgb.DMatrix(test_x, enable_categorical=False)
            vld_data = xgb.DMatrix(x_valid, enable_categorical=False) if x_valid is not None else None

            params = {
                'booster': 'gbtree', 'objective': 'multi:softprob', 'num_class': 2,
                'eval_metric': 'mlogloss', 'eta': 0.04, 'seed': 2020, 'verbosity': 0
            }
            model = xgb.train(params, trn_data, 50000,
                              evals=[(trn_data, 'train'), (val_data, 'eval')],
                              verbose_eval=500, early_stopping_rounds=500)
            val_pred = model.predict(val_data, iteration_range=(0, model.best_iteration))
            tst_pred = model.predict(tst_data, iteration_range=(0, model.best_iteration))
            vld_pred = model.predict(vld_data, iteration_range=(0, model.best_iteration)) if vld_data is not None else None

            importance_dict = model.get_score(importance_type='gain')
            fold_importance_df = pd.DataFrame({
                "Feature": train_x.columns,
                "importance": [importance_dict.get(f, 0) for f in train_x.columns],
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

            # == sync, corrected by elderman == @elder man
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
                learning_rate=0.06,
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
                task_type='CPU',
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

            # == sync, corrected by elderman == @elder man
            del train_pool, valid_pool, test_pool, vld_pool, model, val_pred, tst_pred, vld_pred, fold_importance_df

        # == sync, corrected by elderman == @elder man
        del trn_x, trn_y, val_x, val_y
        gc.collect()
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f'[Mem] After release {mem_end:.2f} GiB  ↓ {mem_start - mem_end:.2f} GiB\n')

    # Final statistics
    auc_mean, auc_std = np.mean(cv_auc), np.std(cv_auc)
    f1_mean, f1_std = np.mean(cv_f1), np.std(cv_f1)
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


# In[ ]:





# In[8]:


# path_data = '../data/dgraph.npz'
data_np = np.load('../data/amazon_cross.npz')
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

def create_cross_features(df):
    """Create cross-group features (strictly using only the input features provided)
Return: (data box with new features, list of new feature names)"""
    df = df.copy()
    new_features = []
    
    # Check for features in the data box
    existing_features = df.columns.tolist()
    
    # Define time-related cross-cutting features
    if 'fea_x_19' in existing_features and 'fea_x_12' in existing_features:
        df['time_related_feature'] = df['fea_x_19'] * df['fea_x_12']
        new_features.append('time_related_feature')
    
    # Define cross-cutting features associated with chart structure
    if 'fea_out_neighbor_fea_x_0_max' in existing_features and 'fea_out_neighbor_fea_x_7_max' in existing_features:
        df['graph_structure_feature'] = df['fea_out_neighbor_fea_x_0_max'] + df['fea_out_neighbor_fea_x_7_max']
        new_features.append('graph_structure_feature')
    
    # Cross-cutting features associated with the definition type distribution
    if 'td_type_ratio_7d_1' in existing_features and 'td_type2_30' in existing_features:
        df['type_distribution_feature'] = df['td_type_ratio_7d_1'] * df['td_type2_30']
        new_features.append('type_distribution_feature')
    
    # Define cross-cutting features of border type associated with time interval
    if 'fea_out_edge_type_mean' in existing_features and 'directed_1hop_in_type_0_count' in existing_features:
        df['edge_type_time_interval_feature'] = df['fea_out_edge_type_mean'] * df['directed_1hop_in_type_0_count']
        new_features.append('edge_type_time_interval_feature')
    
    # Define cross-cutting features between risk dissemination and chart structure
    if 'mixed_propagation_type_4_count' in existing_features and 'directed_1hop_out_type_4_count' in existing_features:
        df['risk_propagation_feature'] = df['mixed_propagation_type_4_count'] + df['directed_1hop_out_type_4_count']
        new_features.append('risk_propagation_feature')
    
    # Defines multi-jump neighbour behaviour contrasting features
    if 'undirected_1hop_type_0_count' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        df['hop_behavior_contrast'] = df['undirected_1hop_type_0_count'] - df['undirected_2hop_type_0_count']
        new_features.append('hop_behavior_contrast')
    
    # Define business-specific anomaly detection features
    if 'transaction_chain_f0_mean' in existing_features and 'directed_1hop_out_type_0_count' in existing_features:
        df['business_specific_anomaly'] = df['transaction_chain_f0_mean'] * df['directed_1hop_out_type_0_count']
        new_features.append('business_specific_anomaly')
    
    # Definition based on cross-cutting features of the box
    if 'fea_x_2' in existing_features:
        df['fea_x_2_bin'] = pd.cut(df['fea_x_2'], bins=[-2, -1, 0, 1, 3, 10, 112], labels=False)
        new_features.append('fea_x_2_bin')
    
    if 'fea_x_6' in existing_features:
        df['fea_x_6_bin'] = pd.cut(df['fea_x_6'], bins=[-1, 0, 100, 200, 400, 822], labels=False)
        new_features.append('fea_x_6_bin')
    
    if 'fea_x_2_bin' in df.columns and 'fea_x_6_bin' in df.columns:
        df['bin_interaction'] = df['fea_x_2_bin'] * df['fea_x_6_bin']
        new_features.append('bin_interaction')
    
    # Define statistical cluster features
    if 'undirected_1hop_f5_std' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        df['statistical_combination'] = df['undirected_1hop_f5_std'] + df['undirected_2hop_type_0_count']
        new_features.append('statistical_combination')
    
    # Define cross-cutting features of original identity stability
    if 'fea_x_2' in existing_features and 'fea_x_6' in existing_features:
        df['feature_stability'] = df['fea_x_2'] * df['fea_x_6']
        new_features.append('feature_stability')
    
    print(f"Created successfully {len(new_features)} Cross-cutting features")
    print("New feature list:", new_features)
    
    return df, new_features

# Example used
# Assume df is a data frame with original features
# df, new_features = create_cross_features(df)


# In[14]:

start_dt = time.time()
# Example:
df_train, new_features = create_cross_features(df_train)
df_valid, _ = create_cross_features(df_valid)
df_test, _ = create_cross_features(df_test)

end_dt = time.time()

dt_cross_fea = end_dt - start_dt
print('Cross feature time is{'.format(dt_cross_fea))

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


