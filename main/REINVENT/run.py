import os, pickle, torch, random, argparse
import yaml
import numpy as np 
from tqdm import tqdm 
torch.manual_seed(1)
np.random.seed(2)
random.seed(1)
from tdc import Oracle
import sys
# sys.path.append('../..')
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
print("path:", '/'.join(path_here.rstrip('/').split('/')[:-2]))
print(sys.path)
from main.optimizer import BaseOptimizer
import time
from train_agent import train_agent


# parser.add_argument('--scoring-function', action='store', dest='scoring_function',
#                     choices=['activity_model', 'tanimoto', 'no_sulphur'],
#                     default='tanimoto',
#                     help='What type of scoring function to use.')
# parser.add_argument('--scoring-function-kwargs', action='store', dest='scoring_function_kwargs',
#                     nargs="*",
#                     help='Additional arguments for the scoring function. Should be supplied with a '\
#                     'list of "keyword_name argument". For pharmacophoric and tanimoto '\
#                     'the keyword is "query_structure" and requires a SMILES. ' \
#                     'For activity_model it is "clf_path " '\
#                     'pointing to a sklearn classifier. '\
#                     'For example: "--scoring-function-kwargs query_structure COc1ccccc1".')
# parser.add_argument('--learning-rate', action='store', dest='learning_rate',
#                     type=float, default=0.0005)
# parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
#                     default=3000)
# parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
#                     default=64)
# parser.add_argument('--sigma', action='store', dest='sigma', type=int,
#                     default=20)
# parser.add_argument('--experience', action='store', dest='experience_replay', type=int,
#                     default=0, help='Number of experience sequences to sample each step. '\
#                     '0 means no experience replay.')
# parser.add_argument('--num-processes', action='store', dest='num_processes',
#                     type=int, default=0,
#                     help='Number of processes used to run the scoring function. "0" means ' \
#                     'that the scoring function will be run in the main process.')
# parser.add_argument('--prior', action='store', dest='restore_prior_from',
#                     default='data/Prior.ckpt',)
# parser.add_argument('--agent', action='store', dest='restore_agent_from',
#                     default='data/Prior.ckpt',)
# parser.add_argument('--save-dir', action='store', dest='save_dir',)



class REINVENToptimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 

        # train_agent(**arg_dict)
        mol_buffer = train_agent(restore_prior_from=restore_prior_from,
                restore_agent_from=restore_agent_from,
                scoring_function=self.oracle,  ### 'tanimoto'
                scoring_function_kwargs=dict(),
                save_dir=None, 
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'], 
                n_steps=config['n_steps'],
                num_processes=0, 
                sigma=config['sigma'],
                experience_replay=0)






