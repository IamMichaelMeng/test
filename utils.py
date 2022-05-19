import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import pdb
def encode_onehot(labels):
    classes = set(labels) # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in # identity创建方矩阵
            enumerate(classes)} # 字典 key为label的值，value为矩阵的每一行
    labels_onehot = np.array(list(map(classes_dict.get, labels)), # get函数得到字典key对应的value
            dtype=np.int32)
    return labels_onehot

def load_data(path="./data/cora/", dataset="cora"):
    matrix = tools()
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
            dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # 储存为csr型稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}


    '''
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
            dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), # flatten:降维，返回一维数组
            dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0])) # eye创建单位矩阵，第一个参数为行数，第二个为列数

    '''


    features = normalize(features)
    # 原数据集:140,200:500,500:1500
    idx_train = range(900)
    idx_val = range(900, 1200)
    idx_test = range(1200, 1500)

    features = torch.FloatTensor(np.array(features.todense())) # tensor为pytorch常用的数据结构
    labels = torch.LongTensor(np.where(labels)[1])
    #adj = sparse_mx_to_torch_sparse_tensor(adj) # 邻接矩阵转为tensor处理
    adj = torch.LongTensor(matrix)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    rowsum = np.array(mx.sum(1)) # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten() # 求倒数
    r_inv[np.isinf(r_inv)] = 0. # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv) # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double() # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx): # 把一个sparse matrix转为torch稀疏张量
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def tools():
    # 导入数据：分隔符为空格
    raw_data = pd.read_csv('./data/cora/cora.content',sep = '\t',header = None)
    num = raw_data.shape[0] # 样本点数2708
    
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b,a)
    map = dict(c)
    
    # 将词向量提取为特征,第二列到倒数第二列
    features =raw_data.iloc[:,1:-1]
    # 检查特征：共1433个特征，2708个样本点

    labels = pd.get_dummies(raw_data[1434])
    # 导入论文引用数据
    raw_data_cites = pd.read_csv('data/cora/cora.cites',sep = '\t',header = None)
    # 创建一个规模和邻接矩阵一样大小的矩阵
    matrix = np.zeros((num,num))
    # 创建邻接矩阵
    for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
        x = map[i]
        y = map[j]  #替换论文编号为[0,2707]
        matrix[x][y] = matrix[y][x] = 1 #有引用关系的样本点之间取1,同时创建出对称矩阵

    matrix = normalize(matrix + sp.eye(matrix.shape[0])) # eye创建单位矩阵，第一个参数为行数，第二个为列数

    return matrix

