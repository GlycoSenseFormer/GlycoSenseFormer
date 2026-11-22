"""Train one manuscript configuration in an isolated output directory."""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--graphs-dir', type=Path, default=ROOT/'data/paper375/graphs', help='Directory of prepared historical split graphs')
    parser.add_argument('--task',choices=['1H','13C'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--print-command',action='store_true')
    args=parser.parse_args()
    from scripts.train_nmr import build_parser
    with (ROOT/'configs/legacy_runs.csv').open() as f: runs=list(csv.DictReader(f))
    conf=args.config.resolve()
    if not any((ROOT/r['config']).resolve()==conf and r['task']==args.task for r in runs):
        parser.error('Select a config for the requested nucleus from configs/legacy_runs.csv')
    cfg=json.loads(conf.read_text())
    train=args.graphs_dir.resolve()
    inputs={'train_path':train/f'train_data_{args.task}.pickle'}
    inputs.update({f'test_path_{s}':train/f'test_data_{args.task}_{s}.pickle' for s in ['mo','di','tri']})
    if not args.print_command and not all(p.is_file() for p in inputs.values()):
        parser.error(f'Training graphs missing in {train}. See README.md for graph preparation and --graphs-dir.')
    fields={a.dest for a in build_parser()._actions}
    effective={k:v for k,v in cfg.items() if k in fields and v is not None and not k.startswith('glycangt_')}
    effective.update(inputs,equiformer_v3_root=ROOT/'src/equiformer_v3')
    cmd=[sys.executable,str(ROOT/'scripts/train_nmr.py')]
    for key,value in effective.items():
        flag='--'+key
        if isinstance(value,bool):
            if value:cmd.append(flag)
        else:cmd.extend([flag,str(value)])
    build_parser().parse_args(cmd[2:])
    if args.print_command:
        import shlex
        print(shlex.join(cmd));return
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):parser.error('Choose an empty output directory')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src/equiformer_v3',ROOT/'src/equiformer_v3/src'])))
    subprocess.run(cmd,cwd=out,env=env,check=True)
if __name__=='__main__':main()
