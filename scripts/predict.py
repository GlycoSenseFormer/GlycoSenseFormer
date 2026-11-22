"""Predict atom-level reference sites of the archived external cohort."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/equiformer_v3'))
sys.path.insert(0, str(ROOT / 'src/equiformer_v3/src'))

def records(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/external100', help='Directory of prepared evaluation graph files')
    parser.add_argument('--case-manifest', type=Path, default=ROOT/'data/cases/fig5_representative_validation_molecules_selected.csv', help='Site-selection manifest for case comparisons')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT/'checkpoints', help='Directory containing the optional manuscript checkpoints')
    parser.add_argument('--group', choices=['table2','table4','cases'], default='table2')
    parser.add_argument('--model', default='main', help='main, all, or an architecture from configs/legacy_runs.csv')
    parser.add_argument('--task', choices=['13C','1H','both'], default='both')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--example', action='store_true', help='Predict the first archived external monosaccharide')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/predictions.csv')
    args=parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f'Prepared evaluation input not found: {args.input_dir}. Supply --input-dir.')
    if args.group == 'cases' and not args.case_manifest.is_file():
        parser.error('Case comparisons require --case-manifest.')
    import torch
    from module.custom_data import load_pickle
    from module.custom_models import build_nmr_model
    from module.custom_tasks.nmr_shift import build_model_kwargs
    from scripts.export_fig5_cross_attention_predictions import predict_molecule, target_mask
    torch.set_num_threads(2)
    device=torch.device(args.device)
    main_name='eqv3_glycanaa_cross_attention_identity_residual_fusion'
    model_name=main_name if args.model=='main' else args.model
    runs=[r for r in records(ROOT/'configs/legacy_runs.csv') if r['group']==args.group and (model_name=='all' or r['model']==model_name) and (args.seed is None or int(r['seed'])==args.seed) and args.task in (r['task'],'both')]
    if not runs:parser.error('No matching runs in configs/legacy_runs.csv')
    for run in runs:
        relative = Path(run['checkpoint']).relative_to('checkpoints')
        checkpoint = args.checkpoint_dir / relative
        if not checkpoint.is_file():
            parser.error(f'Missing {checkpoint}. Supply --checkpoint-dir with the archived checkpoints directory.')
        run['weight_path'] = checkpoint
    out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    partial=out.with_suffix(out.suffix+'.partial')
    norm=json.loads((ROOT/'configs/normalization.json').read_text())
    subs = {(r['split'], int(r['sample_idx'])): r['panel'] for r in records(args.case_manifest)} if args.case_manifest.is_file() else {}
    fields=['model','task','seed','split','sample_idx','molecule','panel','atom_index','reference_ppm','prediction_ppm']
    with partial.open('w',newline='') as handle:
        write=csv.DictWriter(handle,fieldnames=fields);write.writeheader()
        for task in sorted({r['task'] for r in runs}):
            data={s:load_pickle(args.input_dir/f'test_data_{task}_{suffix}.pickle') for s,suffix in [('mono','mo'),('di','di'),('tri','tri')]}
            mean,std=[torch.tensor(norm[task][k],device=device) for k in ('mean','std')]
            for run in [r for r in runs if r['task']==task]:
                config=json.loads((ROOT/run['config']).read_text())
                config['equiformer_v3_root']=str(ROOT/'src/equiformer_v3')
                kwargs=build_model_kwargs(argparse.Namespace(**config))
                net=build_nmr_model(**kwargs).to(device)
                net.load_state_dict(torch.load(run['weight_path'],map_location='cpu',weights_only=False))
                net.eval()
                errors=[]
                for split,items in data.items():
                    for i,item in enumerate(items):
                        if args.example and (split,i)!=('mono',0):continue
                        if args.group=='cases' and (split,i) not in subs:continue
                        with torch.no_grad():
                            y,p=predict_molecule(net,item,mean,std,device,10)
                        ids=torch.nonzero(target_mask(item['loader'].dataset[0])).flatten().tolist()
                        name=item['name'];name=name[0] if isinstance(name,(tuple,list)) else name
                        for atom,ref,pred in zip(ids,y,p,strict=True):
                            write.writerow(dict(zip(fields,(run['model'],task,run['seed'],split,i,name,subs.get((split,i),''),atom,float(ref),float(pred)))))
                            errors.append(abs(float(pred)-float(ref)))
                print(task,run['model'],run['seed'],len(errors),'sites; MAE',sum(errors)/len(errors),flush=True)
                del net
    partial.replace(out)

if __name__=='__main__':main()
