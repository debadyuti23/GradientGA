import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
rdBase.DisableLog('rdApp.error')


def cut(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring

    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None

    return None

def get_all_cuts(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
        return None

    fragment_list = []

    #print("len cuts: ", len(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]'))))
    for bis in mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')):
      bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
      fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

      try:
          fragment = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
          fragment_list.append(fragment)
      except ValueError:
          print("Invalid fragment found, not added to fragment space")
        

    return fragment_list

def cut_ring(mol):

    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')))
            bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')))
            bis = ((bis[0], bis[1]), (bis[1], bis[2]),)

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            print("Invalid fragment found, not added to fragment space")

    return None
  
def get_all_cut_rings(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None

    fragment_list = []

    if random.random() < 0.5:
      bis_list = mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]'))
      bis_list = [((bis[0], bis[1]), (bis[2], bis[3]),) for bis in bis_list]
    
    else:
      bis_list = mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]'))
      bis_list = [((bis[0], bis[1]), (bis[1], bis[2]),) for bis in bis_list]
    

    for bis in bis_list:
      bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

      fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

      try:
        fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        if len(fragments) == 2:
          fragment_list.append(fragments)

      except ValueError:
        return None

    return fragment_list



def ring_OK(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


# TODO: set from main? calculate for dataset?
average_size = 39.15
size_stdev = 3.50


def mol_ok(mol):
    try:
        Chem.SanitizeMol(mol)
        target_size = size_stdev * np.random.randn() + average_size  # parameters set in GA_mol
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except ValueError:
        return False

def get_ring_co_space(parent_A, parent_B):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    fragments_A_list = get_all_cut_rings(parent_A)
    fragments_B_list = get_all_cut_rings(parent_B)
    if (fragments_A_list) is None or (fragments_B_list) is None:
        return None

    #print("FRAG A: ",fragments_A_list)
    #print("LEN FRAG A: ", len(fragments_A_list))
    #print("FRAG B: ",fragments_B_list)
    #print("LEN FRAG B: ", len(fragments_B_list))

    new_mols2 = []
    
    for fragments_A in fragments_A_list:
      for fragments_B in fragments_B_list:

          if fragments_A is None or fragments_B is None:
              return None

          new_mol_trial = []
          for rs in rxn_smarts1:
              rxn1 = AllChem.ReactionFromSmarts(rs)
              for fa in fragments_A:
                  for fb in fragments_B:
                      new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

          new_mols = []
          for rs in rxn_smarts2:
              rxn2 = AllChem.ReactionFromSmarts(rs)
              for m in new_mol_trial:
                  m = m[0]
                  if mol_ok(m):
                      new_mols += list(rxn2.RunReactants((m,)))

          
          for m in new_mols:
              m = m[0]
              if mol_ok(m) and ring_OK(m):
                  new_mols2.append(m)

    return new_mols2

def crossover_ring(parent_A, parent_B):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    
    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)

        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    return None


def get_non_ring_co_space(parent_A, parent_B):

  fragments_A = get_all_cuts(parent_A)
  fragments_B = get_all_cuts(parent_B)
  
  new_mols = []

  #print("FRAG A: ",fragments_A)
  #print("LEN FRAG A: ", len(fragments_A))
  #print("FRAG B: ",fragments_B)
  #print("LEN FRAG B: ", len(fragments_B))
  for fragment_A in fragments_A:
    for fragment_B in fragments_B:
        if fragment_A is None or fragment_B is None:
            continue
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        new_mol_trial = []
        for fa in fragment_A:
            for fb in fragment_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol) and mol is not None:
                new_mols.append(mol)

  if len(new_mols) > 0:
      return new_mols

  return new_mols    


def crossover_non_ring(parent_A, parent_B):

    for i in range(10):
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        if fragments_A is None or fragments_B is None:
            return None
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol) and mol is not None:
                new_mols.append(mol)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return new_mols


def crossover(parent_A, parent_B):
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass

    for i in range(10):
        if random.random() <= 0.5:
            # print 'non-ring crossover'
            new_mol = crossover_non_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
        else:
            # print 'ring crossover'
            new_mol = crossover_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol

    return None

def get_co_space(parent_A, parent_B):
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass
    
    a= random.random()
    if a <= 0.5:
      return get_non_ring_co_space(parent_A, parent_B)
    
    else:
      return get_ring_co_space(parent_A, parent_B)