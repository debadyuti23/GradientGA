from __future__ import print_function
import random
import os
from typing import List
from PIL import Image
import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')
from tdc import Oracle
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import Draw
import main.dlp_graph_ga.crossover as co, main.dlp_graph_ga.mutate as mu
from main.optimizer import BaseOptimizer
#from main.mars.common.chem import mol_to_dgl
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from models import GraphDataset,Discriminator
import dgl
import seaborn as sns
from itertools import islice
import torch
import matplotlib.pyplot as plt
from main.optimizer import BaseOptimizer
from models import MPNN
#from main.mars.common.chem import mol_to_dgl
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from models import GraphDataset,Discriminator
import dgl
import seaborn as sns
from itertools import islice
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from concurrent.futures import ThreadPoolExecutor

MINIMUM = 1e-10

ATOM_TYPES = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
HYBRID_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    None
]

def mol_to_dgl(mol):
    """
    Converts RDKit molecule to DGL graph with optimized node and edge feature processing.
    """
    g = dgl.graph([])

    num_atoms = mol.GetNumAtoms()

    # Prepare atom features
    atom_feats_dict = defaultdict(list)
    atom_feats_list = []

    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        charge = atom.GetFormalCharge()
        symbol = atom.GetSymbol()
        atom_type = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        num_h = atom.GetTotalNumHs()

        
        h_u = [int(symbol == x) for x in ATOM_TYPES]
        h_u += [atom_type, int(charge), int(aromatic)]
        h_u += [int(hybridization == x) for x in HYBRID_TYPES]
        h_u.append(num_h)

        atom_feats_list.append(h_u)
        atom_feats_dict['node_type'].append(atom_type)
        atom_feats_dict['node_charge'].append(charge)

    
    atom_feats_dict['n_feat'] = torch.FloatTensor(atom_feats_list)
    atom_feats_dict['node_type'] = torch.LongTensor(atom_feats_dict['node_type'])
    atom_feats_dict['node_charge'] = torch.LongTensor(atom_feats_dict['node_charge'])

    
    g.add_nodes(num_atoms, data=atom_feats_dict)

    
    edges = []
    bond_feats_list = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        
        edges.append((u, v))
        edges.append((v, u))

        
        bond_type = bond.GetBondType()

        
        bond_feats_list.append([float(bond_type == x) for x in BOND_TYPES])
        bond_feats_list.append([float(bond_type == x) for x in BOND_TYPES])

    
    if edges:
        src, dst = zip(*edges)
        g.add_edges(src, dst)

        
        bond_feats = torch.FloatTensor(bond_feats_list)
        g.edata['e_feat'] = bond_feats

    return g




def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs 
    population_scores = [abs(s) + MINIMUM for s in population_scores] #ENSURE NON-NEGATIVE/NON-ZERO SCORES
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def reproduce2(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    children_space = co.get_co_space(parent_a, parent_b)
    '''
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    '''
    #print("CHILDREN SPACE SIZE: ", len(children_space))
    #print("CHILDREN SPACE: ", children_space)
    return children_space

def reproduce_dlp(model, parent_a, parent_b, child, mutation_rate, device, oracle, config):
  outputs = {}
  def forward_hook(module, input, output):
    outputs['embedding'] = output.detach()
  def backward_hook(module, input, output):
    outputs['gradient'] = output[0].detach()

  model.classifier1.register_forward_hook(forward_hook)
  model.classifier1.register_backward_hook(backward_hook)

  #parent_a = random.choice(mating_pool)
  #parent_b = random.choice(mating_pool)
  #children_space = co.get_co_space(parent_a, parent_b)
  '''
  if new_child is not None:
    new_child = mu.mutate(new_child, mutation_rate)
  '''
  #print("CHILDREN SPACE SIZE: ", len(children_space))

  ##get parent embs, grads##
  parent_graphs = dgl.batch([mol_to_dgl(parent_a).to(device), mol_to_dgl(parent_b).to(device)])
  parent_preds = model(parent_graphs)
  parent_embs = outputs['embedding']
  #print(parent_graphs.batch_size)
  #print(parent_embs.shape)
  parent_preds.backward(torch.ones(parent_embs.shape[0],1).to(device))
  parent_grads = outputs['gradient']
  parent_scores = [float(oracle.evaluator(Chem.MolToSmiles(parent_a))), float(oracle.evaluator(Chem.MolToSmiles(parent_b)))]
  torch_parent_scores = torch.tensor(parent_scores).to(device)
  parent_grads = parent_grads/torch_parent_scores[:,None]

  ##get sample space embs##
  child_graph = mol_to_dgl(child)
  model(child_graph)
  child_emb = outputs['embedding']

  ##calculate DLP probas##
  alpha = 1.0
  #mean
  #probas =  get_dlp_proba_from_mating_pool(parent_embs,parent_grads,offspring_embs, alpha)
  #max version
  max_index = parent_scores.index(max(parent_scores))

  max_emb = parent_embs[max_index]
  max_grad = parent_grads[max_index]
  #print("MAX EMBEDDING SHAPE: ", max_emb.shape)
  #print("MAX GRAD SHAPE: ", max_grad.shape)
  #print(avg_parent_emb,avg_parent_grad,max_emb,max_grad,sep='\n')
  #return get_dlp_proba_one(max_emb, max_grad, child_emb, alpha)


  probas =  get_dlp_proba_max_from_mating_pool(parent_embs,parent_grads,max_index, children_embs, alpha)
  #children_smiles = [Chem.MolToSmiles(mol) for mol in children_space]
  #children_scores = [float(oracle.evaluator(smi)) for smi in children_smiles]

  ##sample children from sample space##
  n_samples = min(config["offspring_size"], int(0.6*len(children_space)))
  offspring_mol = random.choices(children_space,probas,k=n_samples)
  offspring_mol = [mu.mutate(i,config["mutation_rate"]) for i in offspring_mol]
  #print("Sampled {} molecules from sample space".format(n_samples))



def get_dlp_proba(parent_emb,parent_grad,children_emb,alpha=1.0):
  #parent_emb: 1xvector_length
  #parent grad: 1xvector_length
  #children_emb: children_sizexvector_length
  #print("PARENT GRAD INFO: ",parent_grad)
  children_size = children_emb.shape[0]
  #print("sample space--> ",children_size)
  diffs = children_emb-(parent_emb+0.5*alpha*parent_grad).repeat(children_size,1)
  #print("diff shape--> ",diffs.shape)
  nlog_probas = torch.norm(diffs,dim=1)**2
  #print("sq norm shape--> ",nlog_probas)
  probas = torch.exp(-0.5*nlog_probas/alpha)
  #top100_probas, top100_indices = torch.topk(probas, 100)
  #probas = top100_probas
  probas = probas/torch.sum(probas)
  #print("proba shape--> ",probas)
  #print("final probability: ",probas)
  #print("Maximum proba: {} Minimum proba: {}".format(torch.max(probas).item(), torch.min(probas).item()))
  #return probas, top100_indices
  return probas

def get_dlp_proba_one(parent_emb,parent_grad,child_emb,alpha=1.0):
  #parent_emb: 1xvector_length
  #parent grad: 1xvector_length
  #children_emb: children_sizexvector_length
  #print("PARENT GRAD INFO: ",parent_grad)
  #children_size = child_emb.shape[0]
  #print("sample space--> ",children_size)
  diff = child_emb-(parent_emb+0.5*alpha*parent_grad)
  #print("diff shape--> ",diffs.shape)
  logit = torch.norm(diff,dim=1)**2
  #print("sq norm shape--> ",nlog_probas)
  proba = torch.exp(-0.5*logit/alpha)
  #top100_probas, top100_indices = torch.topk(probas, 100)
  #probas = top100_probas
  #probas = probas/torch.sum(probas)
  #print("proba shape--> ",probas)
  #print("final probability: ",probas)
  #print("Maximum proba: {} Minimum proba: {}".format(torch.max(probas).item(), torch.min(probas).item()))
  #return probas, top100_indices
  return proba

def get_dlp_proba_avg_from_mating_pool(parents_emb,parents_grad,children_emb,alpha=1.0):
  avg_parent_emb = torch.mean(parents_emb,0).unsqueeze(0)
  avg_parent_grad = torch.mean(parents_grad,0).unsqueeze(0)
  #print("AVG PARENT EMB SHAPE---> ",avg_parent_emb.shape)
  #print("AVG PARENT GRAD SHAPE--> ", avg_parent_grad.shape)
  return get_dlp_proba(avg_parent_emb, avg_parent_grad, children_emb, alpha)

def get_dlp_proba_max_from_mating_pool(parents_emb,parents_grad,max_index, children_emb,alpha=1.0):
  #avg_parent_emb = torch.mean(parents_emb,0).unsqueeze(0)
  #avg_parent_grad = torch.mean(parents_grad,0).unsqueeze(0)
  max_emb = parents_emb[max_index]
  max_grad = parents_grad[max_index]
  #print("MAX EMBEDDING SHAPE: ", max_emb.shape)
  #print("MAX GRAD SHAPE: ", max_grad.shape)
  #print(avg_parent_emb,avg_parent_grad,max_emb,max_grad,sep='\n')
  return get_dlp_proba(max_emb, max_grad, children_emb, alpha)

def get_dlp_proba_max_from_mating_pool(parents_emb,parents_grad,max_index, children_emb,alpha=1.0):
  #avg_parent_emb = torch.mean(parents_emb,0).unsqueeze(0)
  #avg_parent_grad = torch.mean(parents_grad,0).unsqueeze(0)
  max_emb = parents_emb[max_index]
  max_grad = parents_grad[max_index]
  #print("MAX EMBEDDING SHAPE: ", max_emb.shape)
  #print("MAX GRAD SHAPE: ", max_grad.shape)
  #print(avg_parent_emb,avg_parent_grad,max_emb,max_grad,sep='\n')
  return get_dlp_proba(max_emb, max_grad, children_emb, alpha)

def print_results(t,a,b,x):
    directory = f"C:/Users/chris/molecule_benchmark/mol_opt/results/oracle/{t}/{a}"
    file_path = os.path.join(directory, f'{b}.csv')
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Open the file in write mode
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the rows to the CSV
        writer.writerow(x)
    print(f"File saved at: {file_path}")
def print_results_oracle(t,a,b,x):
    directory = f"C:/Users/chris/molecule_benchmark/mol_opt/results/oracle/{t}/{a}"
    file_path = os.path.join(directory, f'{b}.csv')
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Open the file in write mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the rows to the CSV
        writer.writerow(x)
    print(f"File appended at: {file_path}")

class GA_DLP_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "dlp_graph_ga"
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
    def scores_from_dicts(self, dicts):
        '''
        @params:
            dicts (list): list of score dictionaries
        @return:
            scores (list): sum of property scores of each molecule after clipping
        '''
        scores = []
        # score_norm = sum(self.score_wght.values())
        for score_dict in dicts:
            score = 0.
            for k, v in score_dict.items():
                score += v 
            # score /= score_norm
            score = max(score, 0.)
            scores.append(score)
        return scores
    
    def train(self, population_mol, model, criterion, optim, print_logs = False):
        graphs = [mol_to_dgl(s) for s in population_mol] #change popluation to dgl graph objects

        population_smiles = [Chem.MolToSmiles(mol) for mol in population_mol]
        population_scores = [float(self.oracle.evaluator(smi)) for smi in population_smiles]
        log_scores = torch.tensor(population_scores)

        dataset = GraphDataset(graphs,log_scores,self.device)
          #loader = DataLoader(dataset, 
          #    batch_size=self.batch_size, 
          #    collate_fn=GraphDataset.collate_fn)

        loader = DataLoader(dataset, 
              batch_size=32 if len(population_mol)%32 != 1 else 31, #make sure that the batch does not cause a error with batch norm
              collate_fn=GraphDataset.collate_fn)
          
        losses = []

        for i in range(200):
          for graphs, targs in loader:
                  #print("GRAPHS---> ",graphs)
                  #print("TARGS---> ",targs)
                  #print("TARGS SHAPE ---> ",targs.shape)
                  optim.zero_grad()
                  preds = model(graphs).squeeze() #to bring exact same dim as targets
                  #print("PREDS----> ",preds)
                  #print("PREDS SHAPE ---> ",preds.shape)
                  loss = criterion(preds,targs)
                  loss.backward()
                  optim.step()
          losses.append(loss.item())
          if(print_logs): print('Epoch {} Loss {}'.format(i,loss.item()))
          #Early stopping
          if(loss.item()<1e-8): return losses

        return losses


    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, config["population_size"])

        # select initial population
        # population_smiles = heapq.nlargest(config["population_size"], starting_population, key=oracle)
        population_smiles = starting_population
        self.oracle.assign_population(population_smiles)
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        prev_mol = population_mol
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
        prev_mol_scores = population_scores
        max_score = max(population_scores)

        print('----------------------Training model with sample population-----------------')
        
        #initialization of models
        criterion = nn.MSELoss()
        model = Discriminator(self.device)
        #model = GCN(self.device)
        optim = Adam(model.parameters())

        losses = self.train(population_mol, model, criterion, optim, print_logs=False)

        print('----------------------Training completed-----------------')
        
        #print("Losses = ", losses)
        '''
        plt.figure(figsize=(10, 5))
        plt.title('Losses Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.plot(losses, marker='o', linestyle='-', color='blue')
        plt.grid(True)
        plt.savefig('plot_inital_train.png')

        plt.show()
        '''

        #register forward and backward hooks for graph embedding and grad
        outputs = {}
        def forward_hook(module, input, output):
            outputs['embedding'] = output.detach()
        def backward_hook(module, input, output):
            outputs['gradient'] = output[0].detach()

        model.classifier1.register_forward_hook(forward_hook)
        model.classifier1.register_backward_hook(backward_hook)


        oracle_call = 0
        patience = 0


        itr = 0
        new_molecules = []
        sample_space_map = dict()
        mols_to_train = []
        train_mol_scores = []
        num_trained = 0
        training_increment = 50
        max_score = 0
        while True:            
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            # get sample space. embeddings
            a = 0
            while a == 0:
                mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
                children_space = None
                while children_space is None:
                    parent_a = random.choice(mating_pool)
                    parent_b = random.choice(mating_pool)
                    smile_a = Chem.MolToSmiles(parent_a)
                    smile_b = Chem.MolToSmiles(parent_b)
                    parent_smiles = tuple(sorted((smile_a, smile_b)))
                    if not parent_smiles is sample_space_map.keys():
                        children_space = co.get_co_space(parent_a, parent_b)
                        sample_space_map[parent_smiles] = children_space
                    else:
                        #print("sample space found in dict")
                        children_space = sample_space_map[parent_smiles]
                children_graphs = [mol_to_dgl(s).to(self.device) for s in children_space]
                a = len(children_graphs)
            children_graphs = dgl.batch(children_graphs)
            parent_graphs = dgl.batch([mol_to_dgl(parent_a).to(self.device), mol_to_dgl(parent_b).to(self.device)])
            parent_preds = model(parent_graphs)
            parent_embs = outputs['embedding']
            parent_preds.backward(torch.ones(parent_embs.shape[0],1).to(self.device))
            parent_grads = outputs['gradient']
            parent_scores = [float(self.oracle.evaluator(smile_a)), float(self.oracle.evaluator(smile_b))]
            torch_parent_scores = torch.tensor(parent_scores).to(self.device)
            parent_grads = parent_grads/torch_parent_scores[:,None]
            model(children_graphs)
            children_embs = outputs['embedding']

            

            ##calculate DLP probas##
            alpha = 1.0
            max_index = parent_scores.index(max(parent_scores))
            max_emb = parent_embs[max_index]
            max_grad = parent_grads[max_index]
            probas = get_dlp_proba(max_emb, max_grad, children_embs, alpha)
            ##sample children from sample space##
            n_samples = min(config["offspring_size"], int(0.6*len(children_space)))
            offspring_mol = random.choices(children_space,probas,k=n_samples)
            offspring_mol = [mu.mutate(i,config["mutation_rate"]) for i in offspring_mol]
            offspring_mol = [mol for mol in offspring_mol if not (mol is None)] 
            offspring_scores = [self.oracle(Chem.MolToSmiles(mol)) for mol in offspring_mol]
            # add new_population
            population_mol.extend(offspring_mol)
            population_mol = self.sanitize(population_mol)
            new_molecules.extend(offspring_mol)
            new_molecules = self.sanitize(new_molecules)
            itr+=1

            # stats
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            #population_scores = [float(self.oracle.evaluator(Chem.MolToSmiles(mol))) for mol in  population_mol]
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            #mols_to_train = []
            #max_score = 0
            for idx,score in enumerate(offspring_scores):
              if score >= (max_score - 0.001):
                mols_to_train.append(offspring_mol[idx])
                train_mol_scores.append(score)

            if len(self.oracle) >= training_increment:
              print("---Training for {} samples---".format(len(mols_to_train)))
              self.train(mols_to_train, model, criterion, optim)
              training_increment += 100
              if max(train_mol_scores) > max_score:
                max_score = max(train_mol_scores)
              #mols_to_train = []
              #train_mol_scores = []
              num_trained +=1
            if oracle_call == 0:
                x = self.log_intermediate(finish=False)
                t = self.args.method
                a = self.args.oracles[0]
                b = self.seed
                oracle_call += 500
                print_results(t,a,b,x)
            elif len(self.oracle) > oracle_call:
                x = self.log_intermediate(finish=False)
                t = self.args.method
                a = self.args.oracles[0]
                b = self.seed
                oracle_call += 500
                print_results_oracle(t,a,b,x)
            if self.finish:
                x = self.log_intermediate(finish=True)
                t = self.args.method
                a = self.args.oracles[0]
                b = self.seed
                print_results_oracle(t,a,b,x)
                print('Final, abort ...... ')
                break
            #if self.finish:
            #    print(num_trained)
            #    print("----Final Statistics----")
            #    num_mol = '10000'
            #    oracletype = str(self.args.oracles[0])
            #    population_scores = [self.oracle(Chem.MolToSmiles(mol)) for mol in population_mol]
            #    # Get top 5 indices of population molecule image and also score
            #    print(self.seed)
            #    top_indices = np.argsort(population_scores)[-5:]
            #    filepath = f"C:/Users/chris/molecule_benchmark/mol_opt/main/dlp_graph_ga/results/images/Experiment/{oracletype}/Seed{self.seed}/"
            #    os.makedirs(filepath, exist_ok=True) 
            #    with open(filepath+f'{num_mol}.txt', 'w') as f:
            #        for idx,i in enumerate(top_indices):
            #            f.write(f'{num_mol}{idx + 1} {oracletype}:' + str(population_scores[i]) + '\n')
            #            img = Draw.MolToImage(population_mol[i])
            #           file_name = f'{num_mol}_{oracletype}_{idx + 1}.png'
            #            img.save(os.path.join(filepath, file_name))
            #        
            #    results = self.log_intermediate(finish=True)
            #    with open(filepath + 'DLP_GA.txt', 'w') as f:
            #       f.write(str(results))
            #    with open(filepath + 'SMILES_GA.txt', 'w') as f:
            #       f.write('')    
            #    with open(filepath + 'GRAPH_GA.txt', 'w') as f:  
            #       f.write('')

            #    break
               