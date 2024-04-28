import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import Set2Set
from dgl.nn.pytorch.conv import NNConv
from torch.utils import data

class GraphDataset(data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    @staticmethod
    def collate_fn(batch):
        g = dgl.batch(batch)
        return g

class GraphEncoder(nn.Module):
    def __init__(self,
            n_atom_feat, n_node_hidden,
            n_bond_feat, n_edge_hidden, n_layers
        ):
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
            nn.Linear(n_edge_hidden, n_edge_hidden), nn.ReLU(),
            nn.Linear(n_edge_hidden, n_node_hidden * n_node_hidden)
        )
        self.conv = NNConv(
            n_node_hidden, n_node_hidden,
            edge_network, aggregator_type='mean', bias=False)
        self.gru = nn.GRU(n_node_hidden, n_node_hidden)

    def forward(self, g, h_node, h_edge):
        h_gru = h_node.unsqueeze(0)
        for _ in range(self.n_layers):
            m = F.relu(self.conv(g, h_node, h_edge))
            h_node, h_gru = self.gru(m.unsqueeze(0), h_gru)
            h_node = h_node.squeeze(0)
        return h_node


class Embedding(nn.Module):
    def __init__(self, n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden):
        super().__init__()
        self.node_emb = nn.Linear(n_atom_feat, n_node_hidden)
        self.edge_emb = nn.Linear(n_bond_feat, n_edge_hidden)

    def forward(self, g, x_node, x_edge):
        h_node = self.node_emb(x_node)
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
    
n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden, n_layers, n_out_features = 3, 4, 1, 2, 2,1
device_="cuda" if torch.cuda.is_available() else "cpu"

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device_
        self.encoder = GraphEncoder(n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden, n_layers)
        self.set2set = Set2Set(n_node_hidden, n_iters=6, n_layers=2)
        self.classifier1 = nn.Linear(n_node_hidden*2,n_node_hidden) #value and gradient will come from this layer's output
        self.relu = nn.ReLU()
        self.classifier2 = nn.Linear(n_node_hidden, n_out_features)


    def forward(self, g):
        with torch.no_grad():
            g = g.to(self.device)
            x_node = g.ndata['feat'].to(self.device)
            x_edge = g.edata['feat'].to(self.device)

        h = self.encoder(g, x_node, x_edge)
        h = self.set2set(g, h)
        h = self.classifier1(h)
        h = self.relu(h)
        h = self.classifier2(h)
        return h

    '''
    def loss(self, batch, metrics=['loss']):

        @params:
            batch: batch from the dataset
                g: batched dgl.DGLGraph
                targs: prediction targets
        @returns:
            g.batch_size
            metric_values: cared metric values for
                           training and recording

        g, targs = batch
        targs = targs.to(self.device)
        pred = self(g) # (batch_size, 2)
        loss = F.mse_loss(pred,targs)


        with torch.no_grad():
            pred = logits.argmax(dim=1)
            true = pred == targs
            acc = true.float().sum() / g.batch_size
            tp = (true * targs).float().sum()
            rec = tp / (targs.long().sum() + 1e-6)
            prec = tp / (pred.long().sum() + 1e-6)
            f1 = 2 * rec * prec / (rec + prec + 1e-6)
            local_vars = locals()

        local_vars['loss'] = loss
        metric_values = [local_vars[metric] for metric in metrics]

        return g.batch_size, loss
  '''
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

#print("graph embeddings:", outputs['embedding'])
#print("Gradient of the output of the second layer:", gradients['grad_embedding'])