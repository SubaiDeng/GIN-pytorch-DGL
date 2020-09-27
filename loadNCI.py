import pickle
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

import dgl


class NCIDataset(Dataset):
    def __init__(self, graph_list, label_list):
        self.graph_list = graph_list
        self.label_list = label_list
        self.list_len = len(graph_list)

    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index]

    def __len__(self):
        return self.list_len


def load_nci_data(num_nci=1, cuda=False):
    path = './data/NCI_balanced/'
    filename = 'nci' + str(num_nci) + '.pkl'
    train_test_ratio = 0.7

    with open(path + filename, 'rb') as nci_pkl_file:
        nci_data = pickle.load(nci_pkl_file)
    N = len(nci_data)
    rand_indices = np.random.permutation(N)
    idx_test = rand_indices[int(train_test_ratio * N):]
    idx_train = rand_indices[:int(train_test_ratio * N)]

    #  start to sparse the list of graph
    element_set = set([])
    for i in list(map(lambda x: set(x['node_label']), nci_data)):
        element_set = element_set.union(i)
    element_list = list(element_set)
    node_label_map, node_label_id, node_one_hot = GetOneHotMap(element_list)

    dgl_list = list([])
    label_list = list([])
    for idx in rand_indices:
        graph = nci_data[idx]
        g = dgl.DGLGraph()
        node_label = list(map(lambda x: node_label_map[x], graph['node_label']))
        g.add_nodes(graph['number_node'])  # add node
        g.ndata['feature'] = torch.tensor(node_label)  # add node feature
        if cuda:
            g = g.to('cuda:0')
            # g.ndata['feature'] = g.ndata['feature'].cuda()
        edge_start, edge_end = map(list, zip(*(graph['edge'])))
        edge_start = [i - 1 for i in edge_start]
        edge_end = [i - 1 for i in edge_end]
        edge_w = torch.tensor([*(graph['edge_weight']), *(graph['edge_weight'])])
        if cuda:
            edge_w = edge_w.cuda()
        g.add_edges([*edge_start, *edge_end], [*edge_end, *edge_start])  # add undirected edges
        g.edata['w'] = edge_w
        dgl_list.append(g)
        label_list.append(int((graph['graph_label']+1)/2))
    test_graph_list = [dgl_list[i] for i in idx_test]
    test_label_list = [label_list[i] for i in idx_test]
    train_graph_list = [dgl_list[i] for i in idx_train]
    train_label_list = [label_list[i] for i in idx_train]

    # if cuda:
    #     test_label_list = test_label_list.cuda()
    #     train_label_list = train_label_list.cuda()

    train_set = NCIDataset(train_graph_list, train_label_list)
    test_set = NCIDataset(test_graph_list, test_label_list)
    # train_set = list(zip(test_graph_list, test_label_list))
    # test_set = list(zip(train_graph_list, train_label_list))
    print('SUCCESS: Load NCI finished.\n')
    return train_set, test_set


def GetOneHotMap(element_list):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(element_list))
    print(integer_encoded)
    #  binary encoder
    onehot_encoded = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoded.fit_transform(integer_encoded)
    mapp = dict()
    for label, encoder in zip(element_list, onehot_encoded.tolist()):
        mapp[label] = np.array(encoder, dtype=np.float32)
    return mapp, integer_encoded.reshape(-1, ), onehot_encoded


def collate(samples):
    graph_list, label_list = map(list, zip(*samples))
    batched_graph = dgl.batch(graph_list)
    return batched_graph, torch.tensor(label_list)


if __name__ == '__main__':
    test_set, test_set = load_nci_data()
    print(test_set, test_set)
