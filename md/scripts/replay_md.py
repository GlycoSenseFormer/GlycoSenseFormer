"""Replay one archived GROMACS production input in a new run directory."""
import argparse,csv,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, default=ROOT, help='Directory with the MD inputs and system manifest')
    p.add_argument('--source',choices=['charmm','glycam','gromos'],required=True)
    p.add_argument('--molecule',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gmx',default='gmx')
    p.add_argument('--print-command',action='store_true')
    a=p.parse_args()
    manifest = a.artifact_root / 'md/systems.csv'
    if not manifest.is_file():p.error(f'System manifest not found: {manifest}. Supply --artifact-root.')
    with manifest.open() as f:
        records=[r for r in csv.DictReader(f) if (r['source'],r['molecule_id'])==(a.source,a.molecule)]
    if len(records)!=1:p.error('No unique system with this molecule ID and source in md/systems.csv')
    tpr=a.artifact_root.resolve()/records[0]['system_dir']/'md_5ns_water_200cand.tpr'
    if not tpr.is_file():p.error(f'Production input unavailable: {tpr}')
    cmd=[a.gmx,'mdrun','-s',str(tpr),'-deffnm','production']
    if a.print_command:
        import shlex
        print(shlex.join(cmd));return
    a.out.mkdir(parents=True,exist_ok=True)
    if any(a.out.iterdir()):p.error('Choose an empty output directory')
    subprocess.run(cmd,cwd=a.out,check=True)
if __name__=='__main__':main()
