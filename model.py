import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class MLP_layer(nn.Module):
    def __init__(self,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 is_cuda):
        super(MLP_layer, self).__init__()
        self.num_layers = num_mlp_layers
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        if self.num_layers == 1:
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(self.num_layers-2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(self.num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


    def forward(self, h):
        if self.num_layers == 1:
            return self.linears[0](h)
        else:
            for layer in range(self.num_layers-1):
                t1 = self.linears[layer](h)
                t1 = self.batch_norms[layer](t1)
                h = F.relu(t1)
        return self.linears[self.num_layers-1](h)


class GINClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 feat_drop,
                 learn_eps,
                 graph_pooling_type,
                 neighbor_pooling_type,
                 final_drop,
                 is_cuda=False):
        super(GINClassifier, self).__init__()
        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList()
        self.linear_predictions = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.eps_list = nn.Parameter(torch.zeros(self.num_layers-1))
        self.graph_pooling_type = graph_pooling_type
        self.final_drop = final_drop
        self.feat_drop = feat_drop
        self.neighbor_pooling_type = neighbor_pooling_type
        self.id_layers = 0
        self.learn_eps = learn_eps

        for layer in range(num_layers-1):
            if layer == 0:  # The first layer
                self.mlp_layers.append(MLP_layer(num_mlp_layers, input_dim, hidden_dim, hidden_dim, is_cuda))
                self.linear_predictions.append(nn.Linear(input_dim, output_dim))
            else:
                self.mlp_layers.append(MLP_layer(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, is_cuda))
                self.linear_predictions.append(nn.Linear(hidden_dim, output_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linear_predictions.append(nn.Linear(hidden_dim, output_dim))

        # if is_cuda:
        #     self.mlp_layers.cuda()
        #     self.linear_predictions.cuda()
        #     self.batch_norms.cuda()
        #     self.eps_list.cuda()

    # dgl node pooling function
    def self_eps_aggregate(self, h_self, h_neigh):
        if self.learn_eps:
            h = (1 + self.eps_list[self.id_layers]) * h_self + h_neigh
        else:
            h = h_self + h_neigh
        return h

    def message_func(self, edges):
        # print("h:{}".format(edges.src['h']))
        # print("w:{}".format(edges.data['w']))
        # h = torch.mul(edges.data['w'].float().reshape(1,-1), edges.src['h'].float(),)
        h = edges.data['w'].float() * edges.src['h'].float().t()
        # a_1 = edges.data['w'].float()
        # a_2 = edges.src['h'].float()
        h = h.t()
        return {'msg_h': h}

    def reduce_mean_func(self, nodes):
        h = torch.mean(nodes.mailbox['msg_h'], dim=1)
        h = self.self_eps_aggregate(nodes.data['h'], h)
        return {'h': h}

    def reduce_sum_func(self, nodes):
        h = torch.sum(nodes.mailbox['msg_h'], dim=1)
        h = self.self_eps_aggregate(nodes.data['h'], h)
        return {'h': h}

    def node_pooling(self, g):
        if self.neighbor_pooling_type == 'sum':
            g.update_all(self.message_func, self.reduce_sum_func)
        elif self.neighbor_pooling_type == 'mean':
            g.update_all(self.message_func, self.reduce_mean_func)
        return g.ndata.pop('h')

    def graph_pooling(self, g):
        h = 0
        if self.graph_pooling_type == 'max':
            hg = dgl.max_nodes(g, 'h')
        elif self.graph_pooling_type == 'mean':
            hg = dgl.mean_nodes(g, 'h')
        elif self.graph_pooling_type == 'sum':
            hg = dgl.sum_nodes(g, 'h')
        return hg

    def forward(self, g):
        score_over_layer = 0
        h = g.ndata['feature']

        # 0 layer
        g.ndata['h'] = h
        h_graph = self.graph_pooling(g)
        h_graph = self.linear_predictions[0](h_graph)
        score_over_layer += F.dropout(h_graph, self.final_drop)

        for layer in range(self.num_layers-1):
            self.id_layers = layer
            g.ndata['h'] = h
            # step 1 aggregate
            h = self.node_pooling(g)
            # step 2 MLP
            h = self.mlp_layers[layer](h)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            # step 3 Graph pooling
            g.ndata['h'] = h
            h_graph = self.graph_pooling(g)
            h_graph = self.linear_predictions[layer+1](h_graph)
            score_over_layer += F.dropout(h_graph, self.final_drop)

        return score_over_layer
