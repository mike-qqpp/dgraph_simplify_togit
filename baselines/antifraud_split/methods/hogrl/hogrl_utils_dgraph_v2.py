import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import torch
import copy as cp
import os
from sklearn.metrics import confusion_matrix

filelist = {
    'amz_upu': 'amz_upu_adjlists.pickle',
    'amz_usu': 'amz_usu_adjlists.pickle',
    'amz_uvu': 'amz_uvu_adjlists.pickle',
    'yelp_rsr': 'yelp_rsr_adjlists.pickle',
    'yelp_rtr': 'yelp_rtr_adjlists.pickle',
    'yelp_rur': 'yelp_rur_adjlists.pickle',
    'dgraphfin': 'dgraphfin_homo_adjlists.pickle',
    'dgraph_exp130_nona': 'dgraph_exp130_nona_homo_adjlists.pickle',
    'dgraph_exp130_v2_nona': 'dgraph_exp130_v2_nona_homo_adjlists.pickle',
    'yelpchi': 'yelpchi_homo_adjlists.pickle',
    'amazon': 'amazon_homo_adjlists.pickle',
    'tfinance': 'tfinance_homo_adjlists.pickle'
}

file_matrix_prefix = {
    'amz_upu': 'amazon_upu_matrix_',
    'amz_usu': 'amazon_usu_matrix_',
    'amz_uvu': 'amazon_uvu_matrix_',
    'yelp_rsr': 'yelpnet_rsr_matrix_decompision_',
    'yelp_rtr': 'yelpnet_rtr_matrix_decompision_',
    'yelp_rur': 'yelpnet_rur_matrix_decompision_',
    'dgraphfin': 'dgraphfin_adj_power_matrix',
    'dgraph_exp130_nona': 'dgraph_exp130_nona_adj_power_matrix',
    'dgraph_exp130_v2_nona': 'dgraph_exp130_v2_nona_adj_power_matrix',
    'yelpchi': 'yelpchi_adj_power_matrix',
    'amazon': 'amazon_adj_power_matrix',
    'tfinance': 'tfinance_adg_power_matrix'
}


def calculate_g_mean(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    

    g_mean = np.sqrt(sensitivity * specificity)
    return g_mean


def dict_to_edge_index(edge_dict):
    source_nodes = []
    target_nodes = []

    for src, targets in edge_dict.items():
        for target in targets:
            source_nodes.append(src)
            target_nodes.append(target)

    edge_index = [source_nodes, target_nodes]
    return torch.LongTensor(edge_index)



def numpy_array_to_edge_index(np_array):
    """将稠密邻接矩阵转换为边索引 - 修复版"""
    
    assert np_array.ndim == 2, "Input must be a 2D matrix."
    
    # 对于非常大的矩阵，使用稀疏格式避免内存问题
    if np_array.shape[0] > 10000 and np_array.shape[1] > 10000:
        # 使用稀疏矩阵操作
        if sp.issparse(np_array):
            rows, cols = np_array.nonzero()
        else:
            # 如果已经是稠密矩阵且很大，转换为稀疏矩阵
            sparse_matrix = sp.csr_matrix(np_array)
            rows, cols = sparse_matrix.nonzero()
    else:
        # 对于小矩阵，直接使用nonzero
        rows, cols = np.nonzero(np_array)
    
    # 创建边索引
    edge_index = np.vstack((rows, cols))
    # 转换为PyTorch张量
    edge_index_tensor = torch.from_numpy(edge_index).long()
    return edge_index_tensor

def sparse_matrix_to_edge_index(sparse_matrix):
    """直接将稀疏矩阵转换为边索引，避免转换为稠密矩阵"""
    rows, cols = sparse_matrix.nonzero()
    edge_index = np.vstack((rows, cols))
    return torch.from_numpy(edge_index).long()

def load_data(data, k=2, prefix=''):
    """
    Load graph, feature, and label given dataset name
    """
    pickle_file = {}
    matrix_prefix = {}
    for key in filelist: # update the file paths
        pickle_file[key] = os.path.join(prefix, filelist[key])
        matrix_prefix[key] = os.path.join(prefix, file_matrix_prefix[key])
    
    if data == 'yelp':
        data_file = loadmat(os.path.join(prefix, 'YelpChi.mat'))
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        
        with open(pickle_file['yelp_rur'], 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        relation1 = dict_to_edge_index(relation1)
        relation1_tree = []
        for i in range(1, k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['yelp_rur'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation1_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['yelp_rtr'], 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        relation2 = dict_to_edge_index(relation2)
        relation2_tree = []
        for i in range(1, k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['yelp_rtr'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation2_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['yelp_rsr'], 'rb') as file:
            relation3 = pickle.load(file)
        file.close()
        relation3 = dict_to_edge_index(relation3)
        relation3_tree = []
        for i in range(1, k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['yelp_rsr'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation3_tree.append(numpy_array_to_edge_index(tree))
        return [[relation1,relation1_tree],[relation2,relation2_tree],[relation3,relation3_tree]],feat_data,labels
    
    elif data == 'amazon_init':
        data_file = loadmat(os.path.join(prefix, 'Amazon.mat'))
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        
        with open(pickle_file['amz_upu'], 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        relation1 = dict_to_edge_index(relation1)
        relation1_tree = []
        for i in range(1, k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['amz_upu'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation1_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['amz_usu'], 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        relation2 = dict_to_edge_index(relation2)
        relation2_tree = []
        for i in range(1, k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['amz_usu'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation2_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['amz_uvu'], 'rb') as file:
            relation3 = pickle.load(file)
        file.close()
        relation3_tree = []
        for i in range(1, k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['amz_uvu'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation3_tree.append(numpy_array_to_edge_index(tree))
        relation3 = dict_to_edge_index(relation3)
        
        return [[relation1,relation1_tree],[relation2,relation2_tree],[relation3,relation3_tree]],feat_data,labels
    
    elif data == 'CCFD':
        assert False,'CCFD dataset is secret, please contact the author for the dataset.'
        
        data_file= loadmat(os.path.join(prefix, 'CCFD.mat'))
        labels = data_file['labels'].flatten()
        feat_data = data_file['features']
        with open('../data/net_source_CCFD.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        relation1 = dict_to_edge_index(relation1)
        relation1_tree = []
        for i in range(1, k+1):
            file_name = f'../data/CCFD_r1_matrix_{k}.pkl'
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation1_tree.append(numpy_array_to_edge_index(tree))	
        return [[relation1,relation1_tree]],feat_data,labels
    elif data is not None:
        # 加载DGraphFin格式数据集（包括dgraph_exp130_nona）
        print(f"加载{data}数据集...")
        
        # 加载特征和标签
        data_file = np.load(os.path.join('../data', f'{data}.npz'))
        feat_data = data_file['x']
        labels = data_file['y']
        
        print(f"  节点数: {feat_data.shape[0]}, 特征维度: {feat_data.shape[1]}")
        
        # 加载原始边索引（从原始.npz）
        try:
            orig_file = np.load(os.path.join(prefix, f'{data}.npz'))
            edge_index_original = orig_file['edge_index']
            print(f"  原始边数: {edge_index_original.shape[1]}")
        except:
            print(f"  警告: 无法加载原始边索引")
            edge_index_original = None
        
        # 加载邻接列表并转换为边索引
        if os.path.exists(pickle_file[data]):
            print(data)
            with open(pickle_file[data], 'rb') as file:
                relation1 = pickle.load(file)
            file.close()
            relation1 = dict_to_edge_index(relation1)
            print(f"  邻接列表边数: {relation1.shape[1]}")
        else:
            print(f"  警告: 邻接列表文件不存在，使用原始边索引")
            if edge_index_original is not None:
                # 转换为PyTorch张量 [2, n_edges] 格式
                if edge_index_original.shape[0] != 2:
                    edge_index_original = edge_index_original.T
                relation1 = torch.from_numpy(edge_index_original).long()
            else:
                # 创建自环
                num_nodes = feat_data.shape[0]
                relation1 = torch.tensor([range(num_nodes), range(num_nodes)], dtype=torch.long)
        
        # 加载矩阵幂（1-k阶）- 使用稀疏矩阵避免内存爆炸
        relation1_tree = []
        for i in range(1, k+1):
            try:
                # 尝试加载.npz文件（稀疏矩阵格式）
                file_name = '{}{}.npz'.format(matrix_prefix[data], i)
                if os.path.exists(file_name):
                    # 从npz文件直接加载稀疏矩阵并转换为边索引
                    sparse_matrix = sp.load_npz(file_name)
                    edge_index_tree = sparse_matrix_to_edge_index(sparse_matrix)
                    relation1_tree.append(edge_index_tree)
                    print(f"  第{i}层矩阵幂: {edge_index_tree.shape[1]} 条边")
                else:
                    # 尝试加载.pkl文件
                    file_name = '{}{}.pkl'.format(matrix_prefix[data], i)
                    if os.path.exists(file_name):
                        with open(file_name, 'rb') as file:
                            tree_array = pickle.load(file)
                        
                        # 如果是稀疏矩阵，直接转换
                        if sp.issparse(tree_array):
                            edge_index_tree = sparse_matrix_to_edge_index(tree_array)
                        else:
                            # 如果是稠密矩阵，使用安全的转换方法
                            edge_index_tree = numpy_array_to_edge_index(tree_array)
                        
                        relation1_tree.append(edge_index_tree)
                        print(f"  第{i}层矩阵幂: {edge_index_tree.shape[1]} 条边")
                    else:
                        print(f"  警告: 第{i}层矩阵幂文件不存在，使用自环")
                        num_nodes = feat_data.shape[0]
                        edge_index_tree = torch.tensor([range(num_nodes), range(num_nodes)], dtype=torch.long)
                        relation1_tree.append(edge_index_tree)
            except Exception as e:
                print(f"  警告: 加载第{i}层矩阵幂失败: {e}")
                num_nodes = feat_data.shape[0]
                edge_index_tree = torch.tensor([range(num_nodes), range(num_nodes)], dtype=torch.long)
                relation1_tree.append(edge_index_tree)
        
        # DGraphFin格式使用单一同质邻接图
        return [[relation1, relation1_tree]], feat_data, labels
    
    else:
        raise ValueError(f"不支持的数据集: {data}。支持的数据集: 'yelp', 'amazon', 'dgraphfin', 'dgraph_exp130_nona'")


def Visualization(labels, embedding, prefix):
    train_pos, train_neg = pos_neg_split(list(range(len(labels))), labels)
    sampled_idx_train = undersample(train_pos, train_neg, scale=1)
    tsne = TSNE(n_components=2, random_state=43)
    sampled_idx_train = np.array(sampled_idx_train)
    sampled_idx_train = np.random.choice(sampled_idx_train, size=5000, replace=True)
    ps = embedding[sampled_idx_train]
    ls = labels[sampled_idx_train]

    X_reduced = tsne.fit_transform(ps)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_reduced)
    
    plt.figure(figsize=(8, 8))

    plt.scatter(X_scaled[ls == 0, 0], X_scaled[ls == 0, 1], c='#14517C', label='Label 0', s=3)
    plt.scatter(X_scaled[ls == 1, 0], X_scaled[ls == 1, 1], c='#FA7F6F', label='Label 1', s=3)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.xticks([])
    plt.yticks([])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    filepath = os.path.join(prefix, 'HOGRL.png')
    plt.savefig(filepath)
    plt.show()
    
def normalize(mx):
    """
        Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def pos_neg_split(nodes, labels):
    """
    Find positive and negative nodes given a list of nodes and their labels
    :param nodes: a list of nodes
    :param labels: a list of node labels
    :returns: the spited positive and negative nodes
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
    """
    Under-sample the negative nodes
    :param pos_nodes: a list of positive nodes
    :param neg_nodes: a list negative nodes
    :param scale: the under-sampling scale
    :return: a list of under-sampled batch nodes
    """

    aux_nodes = cp.deepcopy(neg_nodes)
    aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes)*scale))
    batch_nodes = pos_nodes + aux_nodes

    return batch_nodes

def calculate_g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        sensitivities.append(sensitivity)
    g_mean = np.prod(sensitivities) ** (1 / len(sensitivities))
    return g_mean
