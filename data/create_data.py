
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
import pickle
import numpy as np
import torch
import os
from collections import Counter
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
from CarbonDataset import Carbons13C
from HydrogenDataset import Hydrogens1H

from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader

CONFORMER_DIR = "conformations"

def get_cut_off_graph(edge_index, edge_attr, p, cut_off = 6.0):
    row, col = edge_index
    dist = torch.sqrt(torch.sum((p[row]- p[col])**2, dim = 1))
    mask = dist <= cut_off
    edge_index = edge_index[:,mask]
    edge_attr = edge_attr[mask]
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce  = "max")
    return edge_index, edge_attr

def generate_conformations(
    mol,
    nbr_confs=10,
    oversample=2,
    seed=1,
    num_threads=10,
    optimize_mmff=True,
):
    params = Chem.rdDistGeom.ETKDGv3()
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = 0.001
    params.numThreads = num_threads
    params.enforceChirality = True
    params.randomSeed = seed

    m_ = Chem.Mol(mol)
    m_.RemoveAllConformers()
    Chem.SanitizeMol(m_)
    m_ = Chem.AddHs(m_, addCoords=True)

    num_confs = max(1, nbr_confs * oversample)
    conf_ids = list(Chem.rdDistGeom.EmbedMultipleConfs(m_, numConfs=num_confs, params=params))

    if not conf_ids:
        m_ = Chem.RemoveHs(m_)
        return [], np.array([]), m_

    energies = []
    if optimize_mmff and AllChem.MMFFHasAllMoleculeParams(m_):
        results = AllChem.MMFFOptimizeMoleculeConfs(
            m_,
            numThreads=num_threads,
            maxIters=200,
            mmffVariant="MMFF94",
        )
        energies = [energy for _, energy in results]
    else:
        ps = AllChem.MMFFGetMoleculeProperties(m_, mmffVariant='MMFF94')
        for conf in m_.GetConformers():
            if ps is None:
                ff = AllChem.UFFGetMoleculeForceField(m_, confId=conf.GetId())
            else:
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m_, ps, confId=conf.GetId())
            if isinstance(ff, type(None)):
                energies.append(float("inf"))
            else:
                energies.append(ff.CalcEnergy())

    m_ = Chem.RemoveHs(m_)
    energies = np.array(energies)
    ind = energies.argsort()[:nbr_confs]
    energies = energies[ind]
    conformations = []
    for i, c in enumerate(m_.GetConformers()):

        if i not in ind:
            continue
        xyz = c.GetPositions()
        conformations.append(xyz)

    return conformations, energies, m_

def conformer_path(name):
    return os.path.join(CONFORMER_DIR, name + ".pickle")

def load_conformer_count(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return len(pickle.load(handle))

def has_usable_conformer_cache(existing_count, requested_count):
    if existing_count is None:
        return False
    if existing_count == requested_count:
        return True
    min_partial_count = max(1, int(requested_count * 0.8))
    return min_partial_count <= existing_count < requested_count

def create_conformer(
    mols,
    nbr_confs=10,
    force=False,
    oversample=2,
    seed=0,
    num_threads=10,
    optimize_mmff=True,
):
    counts = Counter()
    for mol in mols:
        if mol is None:
            continue
        name = mol.GetProp("_Name")

        name = name.replace("->", "")
        name = name.replace("(","")
        name = name.replace(")","")

        path = conformer_path(name)
        existing_count = load_conformer_count(path)
        if not force and has_usable_conformer_cache(existing_count, nbr_confs):
            counts[existing_count] += 1
            continue
        conformations, _,_ = generate_conformations(
            mol,
            nbr_confs=nbr_confs,
            oversample=oversample,
            seed=seed,
            num_threads=num_threads,
            optimize_mmff=optimize_mmff,
        )
        with open(path, 'wb') as handle:
            pickle.dump(conformations, handle)
        counts[len(conformations)] += 1
    return counts

def train_test_split(mols, fold):
    N_t = len(mols)
    indices = range(N_t)
    indices_kfold = np.array_split(indices, 10)[fold]
    train_mask = np.ones(N_t, dtype=bool)
    train_mask[indices_kfold] = False
    test_mask = [not(a) for a in train_mask]
    train_datas = [data for data, m in zip(mols, train_mask) if m]
    test_datas = [data for data, m in zip(mols, test_mask) if m]   
    return  train_datas, test_datas

def add_conformations_to_train_dataset(dataset):
    new_dataset = []
    for d_ in dataset:
        name = d_.name
        name = name.replace("->","")
        name = name.replace(")","")
        name = name.replace("(","")
        conf_path = conformer_path(name)
        #if not os.path.exists(file_path):
        #    continue

        with open(conf_path, 'rb') as handle:
            distencies = pickle.load(handle)
            for dis_ in distencies:
                new_data = d_.clone()
                new_data.pos = torch.tensor(dis_)
                edge_index, edge_attr = get_cut_off_graph(new_data.edge_index, new_data.edge_attr, new_data.pos ,cut_off=6.)
                new_data.edge_index = edge_index
                new_data.edge_attr = edge_attr
                new_dataset.append(new_data)
    new_dataset_ = InMemoryDataset()
    data, slices, _ = collate(
                new_dataset[0].__class__,
                data_list=new_dataset,
                increment=False,
                add_batch=False,
            )
    new_dataset_.data = data

    new_dataset_.slices = slices
    return new_dataset_

def add_conformations_to_test_dataset(dataset):
    test_datas = []
    for data in dataset:
        datas_ = []
        name = data.name[0] if isinstance(data.name, (list, tuple)) else data.name
        name = name.replace("->","")
        name = name.replace("(","")
        name = name.replace(")","")
        n_atoms = data.x.size(0)

        conf_path = conformer_path(name)
        k = 0
        if os.path.exists(conf_path):
                with open(conf_path, 'rb') as handle:
                        distencies = pickle.load(handle)
                for j, dis_ in enumerate(distencies):
                        d = data.clone()
                        
                        pos = torch.from_numpy(dis_).float()
                        d.pos = pos
                        edge_index, edge_attr = get_cut_off_graph(d.edge_index, d.edge_attr, d.pos ,cut_off=6.)
                        d.edge_index = edge_index
                        d.edge_attr = edge_attr
                        datas_.append(d)
                        k = k + 1

        else:
                continue
        if k == 0:
                continue
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)
        test_datas.append( {"name":data.name,"n_atoms": n_atoms, "n_nmr":torch.sum(mask).item(), "n_mols": k,
                                        "loader" : DataLoader(datas_, batch_size = len(datas_), shuffle=False )})
    return test_datas

def main(
    path_to_mono,
    path_to_di,
    path_to_tri,
    fold,
    mono_confs,
    di_confs,
    tri_confs,
    force_conformers,
    conformer_oversample,
    conformer_seed,
    conformer_threads,
    conformer_dir,
    optimize_mmff,
):
    global CONFORMER_DIR
    CONFORMER_DIR = conformer_dir
    np.random.seed(0)
    torch.manual_seed(0)

    mols_mono = []
    with Chem.SDMolSupplier(path_to_mono) as suppl:
        for mol in suppl:
            mols_mono.append(mol)
    if not os.path.exists("roots/carbon/mono"):  
        os.makedirs("roots/carbon/mono")
    if not os.path.exists("roots/hydrogen/mono"):  
        os.makedirs("roots/hydrogen/mono")              
    mols_mono_carbon_d = list(Carbons13C("roots/carbon/mono", mols_mono))
    mols_mono_hydrogen_d = list(Hydrogens1H("roots/hydrogen/mono", mols_mono))

    mols_di = []
    with Chem.SDMolSupplier(path_to_di) as suppl:
        for mol in suppl:
            mols_di.append(mol)
    if not os.path.exists("roots/carbon/di"):  
        os.makedirs("roots/carbon/di") 
    if not os.path.exists("roots/hydrogen/di"):  
        os.makedirs("roots/hydrogen/di")             
    mols_di_carbon_d = list(Carbons13C("roots/carbon/di", mols_di))
    mols_di_hydrogen_d = list(Hydrogens1H("roots/hydrogen/di", mols_di))

    mols_tri = []
    with Chem.SDMolSupplier(path_to_tri) as suppl:
        for mol in suppl:
            mols_tri.append(mol)
    if not os.path.exists("roots/carbon/tri"):  
        os.makedirs("roots/carbon/tri")   
    if not os.path.exists("roots/hydrogen/tri"):  
        os.makedirs("roots/hydrogen/tri")             
    mols_tri_carbon_d = list(Carbons13C("roots/carbon/tri",mols_tri ))
    mols_tri_hydrogen_d = list(Hydrogens1H("roots/hydrogen/tri",mols_tri ))

    if not os.path.exists(CONFORMER_DIR):
        os.makedirs(CONFORMER_DIR)
    print(
        "Conformer config: "
        f"mono={mono_confs}, di={di_confs}, tri={tri_confs}, "
        f"oversample={conformer_oversample}, seed={conformer_seed}, "
        f"threads={conformer_threads}, optimize_mmff={optimize_mmff}, "
        f"force={force_conformers}, dir={CONFORMER_DIR}",
        flush=True,
    )
    mono_counts = create_conformer(
        mols_mono,
        mono_confs,
        force_conformers,
        conformer_oversample,
        conformer_seed,
        conformer_threads,
        optimize_mmff,
    )
    di_counts = create_conformer(
        mols_di,
        di_confs,
        force_conformers,
        conformer_oversample,
        conformer_seed,
        conformer_threads,
        optimize_mmff,
    )
    tri_counts = create_conformer(
        mols_tri,
        tri_confs,
        force_conformers,
        conformer_oversample,
        conformer_seed,
        conformer_threads,
        optimize_mmff,
    )
    print(f"Conformer counts mono: {dict(sorted(mono_counts.items()))}", flush=True)
    print(f"Conformer counts di: {dict(sorted(di_counts.items()))}", flush=True)
    print(f"Conformer counts tri: {dict(sorted(tri_counts.items()))}", flush=True)

    train_data_carbon_mo, test_data_carbon_mo = train_test_split(mols_mono_carbon_d, fold)
    train_data_carbon_di, test_data_carbon_di = train_test_split(mols_di_carbon_d, fold)
    train_data_carbon_tri, test_data_carbon_tri = train_test_split(mols_tri_carbon_d, fold)

    train_data_hydrogen_mo, test_data_hydrogen_mo = train_test_split(mols_mono_hydrogen_d, fold)
    train_data_hydrogen_di, test_data_hydrogen_di = train_test_split(mols_di_hydrogen_d, fold)
    train_data_hydrogen_tri, test_data_hydrogen_tri = train_test_split(mols_tri_hydrogen_d, fold)


    train_data_carbon = train_data_carbon_mo + train_data_carbon_di + train_data_carbon_tri
    train_data_hydrogen = train_data_hydrogen_mo + train_data_hydrogen_di + train_data_hydrogen_tri

    train_data_carbon = add_conformations_to_train_dataset(train_data_carbon)
    train_data_hydrogen = add_conformations_to_train_dataset(train_data_hydrogen)
    
    if not os.path.exists("datasets"):  
        os.makedirs("datasets") 
    
    with open("datasets/train_data_13C.pickle", 'wb') as handle:
        pickle.dump(train_data_carbon, handle)
    with open("datasets/train_data_1H.pickle", 'wb') as handle:
        pickle.dump(train_data_hydrogen, handle)

    test_data_carbon_mo = add_conformations_to_test_dataset(test_data_carbon_mo)
    test_data_carbon_di = add_conformations_to_test_dataset(test_data_carbon_di)
    test_data_carbon_tri = add_conformations_to_test_dataset(test_data_carbon_tri)

    test_data_hydrogen_mo = add_conformations_to_test_dataset(test_data_hydrogen_mo)
    test_data_hydrogen_di = add_conformations_to_test_dataset(test_data_hydrogen_di)
    test_data_hydrogen_tri = add_conformations_to_test_dataset(test_data_hydrogen_tri)

    with open("datasets/test_data_13C_mo.pickle", 'wb') as handle:
        pickle.dump(test_data_carbon_mo, handle)
    with open("datasets/test_data_13C_di.pickle", 'wb') as handle:
        pickle.dump(test_data_carbon_di, handle)
    with open("datasets/test_data_13C_tri.pickle", 'wb') as handle:
        pickle.dump(test_data_carbon_tri, handle)
    with open("datasets/test_data_1H_mo.pickle", 'wb') as handle:
        pickle.dump(test_data_hydrogen_mo, handle)
    with open("datasets/test_data_1H_di.pickle", 'wb') as handle:
        pickle.dump(test_data_hydrogen_di, handle)
    with open("datasets/test_data_1H_tri.pickle", 'wb') as handle:
        pickle.dump(test_data_hydrogen_tri, handle)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_path_mo', type=str, help='Path to train dataset mono')
    parser.add_argument('--data_path_di', type=str, help='Path to train dataset di')
    parser.add_argument('--data_path_tri', type=str, help='Path to train dataset tri')
    parser.add_argument('--fold', type=int, default=1, help='Fold in k-fold')
    parser.add_argument('--mono_confs', type=int, default=10, help='Number of conformers per monosaccharide')
    parser.add_argument('--di_confs', type=int, default=1, help='Number of conformers per disaccharide')
    parser.add_argument('--tri_confs', type=int, default=1, help='Number of conformers per trisaccharide')
    parser.add_argument('--force_conformers', action='store_true', help='Regenerate conformer pickle files even if the requested count is already cached')
    parser.add_argument('--conformer_oversample', type=int, default=2, help='Embed this multiple of requested conformers before energy ranking')
    parser.add_argument('--conformer_seed', type=int, default=1, help='RDKit ETKDG random seed')
    parser.add_argument('--conformer_threads', type=int, default=10, help='RDKit conformer generation/optimization threads')
    parser.add_argument('--conformer_dir', type=str, default='conformations', help='Directory used for conformer pickle cache')
    parser.add_argument('--no_mmff_optimize', action='store_true', help='Skip MMFF geometry optimization before energy ranking')

    args = parser.parse_args()
    main(
        args.data_path_mo,
        args.data_path_di,
        args.data_path_tri,
        args.fold,
        args.mono_confs,
        args.di_confs,
        args.tri_confs,
        args.force_conformers,
        args.conformer_oversample,
        args.conformer_seed,
        args.conformer_threads,
        args.conformer_dir,
        not args.no_mmff_optimize,
    )
