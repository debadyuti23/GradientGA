import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import Set2Set
from dgl.nn.pytorch.conv import NNConv
from torch.utils import data

class GraphDataset(data.Dataset):
    def __init__(self, graphs, scores,n_device):
        self.graphs = graphs
        self.scores = torch.tensor(scores)
        self.device=n_device
        

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index].to(self.device)
        score = torch.tensor(self.scores[index],device=self.device)
        return graph, score

    
    @staticmethod
    def collate_fn(batch):
        g,s = zip(*batch)
        g = dgl.batch(g)
        s = torch.stack(list(s), dim=0)
        return g,s



class GraphEncoder(nn.Module):
    def __init__(self,
            n_atom_feat, n_node_hidden,
            n_bond_feat, n_edge_hidden, n_layers):
        super().__init__()
        self.embedding = Embedding(n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden)
        self.mpnn = MPNN(n_node_hidden, n_edge_hidden, n_layers)

    def forward(self, g, x_node, x_edge):
        '''
        @params:
            g      : batch of dgl.DGLGraph
            x_node : node features, torch.FloatTensor of shape (tot_n_nodes, n_atom_feat)
            x_edge : edge features, torch.FloatTensor of shape (tot_n_edges, n_atom_feat)
        @return:
            h_node : node hidden states
        '''
        h_node, h_edge = self.embedding(g, x_node, x_edge)
        h_node = self.mpnn(g, h_node, h_edge)
        return h_node


class MPNN(nn.Module):
    def __init__(self, n_node_hidden, n_edge_hidden, n_layers):
        super().__init__()
        self.n_layers = n_layers
        edge_network = nn.Sequential(
            nn.Linear(n_edge_hidden, n_edge_hidden),
            nn.BatchNorm1d(n_edge_hidden),
            nn.LeakyReLU(0.8),
            nn.Linear(n_edge_hidden, n_node_hidden * n_node_hidden),
            nn.BatchNorm1d(n_node_hidden * n_node_hidden),
        )
        self.conv = NNConv(
            n_node_hidden, n_node_hidden,
            edge_network, aggregator_type='mean', bias=False)
        self.gru = nn.GRU(n_node_hidden, n_node_hidden)

    def forward(self, g, h_node, h_edge):
        h_gru = h_node.unsqueeze(0)
        for _ in range(self.n_layers):
            m = self.conv(g, h_node, h_edge)
            h_node, h_gru = self.gru(m.unsqueeze(0), h_gru)
            h_node = h_node.squeeze(0)
        return h_node


class Embedding(nn.Module):
    def __init__(self, n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden):
        super().__init__()
        self.batch_norm_node = nn.BatchNorm1d(n_atom_feat)
        self.node_emb = nn.Linear(n_atom_feat, n_node_hidden)
        self.batch_norm_edge = nn.BatchNorm1d(n_bond_feat)
        self.edge_emb = nn.Linear(n_bond_feat, n_edge_hidden)

    def forward(self, g, x_node, x_edge):
        x_node = self.batch_norm_node(x_node)
        h_node = self.node_emb(x_node)
        x_edge = self.batch_norm_edge(x_edge)
        h_edge = self.edge_emb(x_edge)
        return h_node, h_edge


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2):
        super().__init__()
        self.out = nn.Linear(in_channels, out_channels)
        self.linears = nn.ModuleList([
            nn.Linear(in_channels, in_channels
            ) for i in range(n_layers)])

    def forward(self, x):
        for lin in self.linears:
            x = F.relu(lin(x))
        x = self.out(x)
        return x
    
n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden, n_layers, n_out_features = 17, 16, 5, 8, 2,1


class Discriminator(nn.Module):
    def __init__(self,n_device):
        super().__init__()
        self.device = n_device
        self.encoder = GraphEncoder(n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden, n_layers)
        self.set2set = Set2Set(n_node_hidden, n_iters=6, n_layers=4)
        self.classifier1 = nn.Linear(2*n_node_hidden,n_node_hidden) #value and gradient will come from this layer's output
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.classifier2 = nn.Linear(n_node_hidden, n_out_features)
        self.batch_norm = nn.BatchNorm1d(2*n_node_hidden)
        #self.classifier = nn.Linear(2*n_node_hidden, n_out_features)
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)


    def forward(self, g):
        with torch.no_grad():
            g = g.to(self.device)
            x_node = g.ndata['n_feat'].to(self.device)
            x_edge = g.edata['e_feat'].to(self.device)

        h = self.encoder(g, x_node, x_edge)
        #print("HIDDEN SHAPE AFTER GCN--> ",h.shape)
        #print("after gcn: ",h)
        h = self.set2set(g, h)
        #print("HIDDEN SHAPE AFTER SET2SET--> ",h.shape)
        #print("after set2set: ",h)
        #h = self.classifier1(h)
        #h = self.relu(h)
        #h = self.classifier2(h)
        #h = self.relu(h)
        h = self.batch_norm(h)
        h= self.classifier1(h)
        #print("HIDDEN SHAPE AFTER NORM--> ",h.shape)
        h = self.relu(h)
        h = self.classifier2(h)
        h = self.sigmoid(h)
        return h

    def get_embedding(self):
        outputs = {}
        def forward_hook(self, output):
            outputs['embedding'] = output.detach()
        self.classifier1.register_forward_hook(forward_hook)
        return outputs['embedding']

    def get_gradients(self):
        gradients = {}
        def backward_hook(self, grad_output):
            gradients['grad_embedding'] = grad_output[0].detach()
        self.classifier1.register_backward_hook(backward_hook)
        return gradients['grad_embedding']