"""Recompute manuscript statistics from atom-level predictions."""
import argparse,csv,math,statistics as st
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from module.model_names import GLYCOSENSEFORMER, canonical_model_name

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def metrics(pairs):
    y,p=zip(*pairs);ae=sorted(abs(a-b) for a,b in pairs)
    def quant(f):
        v=f*(len(ae)-1);lo=math.floor(v);hi=math.ceil(v)
        return ae[lo]+(v-lo)*(ae[hi]-ae[lo])
    mse=st.mean((a-b)**2 for a,b in pairs)
    return dict(mae=st.mean(ae),rmse=math.sqrt(mse),r2=1-mse/st.pvariance(y),medae=quant(.5),p90_ae=quant(.9),p95_ae=quant(.95))
def check(values,mean,sd):
    obs=(st.mean(values),st.stdev(values));exp=(float(mean),float(sd))
    if any(round(a,4)!=round(b,4) for a,b in zip(obs,exp)):
        raise ValueError(f'Observed {obs}, expected {exp}')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--predictions',type=Path,help='Fresh output of predict.py for all three main-model seeds and both nuclei')
    ap.add_argument('--artifact-root',type=Path,default=ROOT,help='Directory with archived predictions and the run manifest')
    args=ap.parse_args();root=args.artifact_root/'data/reference'
    if not root.is_dir():ap.error(f'Reference data not found: {root}. Supply --artifact-root.')
    groups=defaultdict(list)
    if args.predictions:
        for r in read(args.predictions):
            if canonical_model_name(r['model']) != GLYCOSENSEFORMER:continue
            for s in [r['split'],'overall']:
                groups['main_table2',r['task'],s,r['seed']].append((float(r['reference_ppm']),float(r['prediction_ppm'])))
    else:
        for task in ['13C','1H']:
            for r in read(root/(task+'_atom_predictions.csv')):
                for s in [r['split'],'overall']:
                    groups[r['group'],task,s,r['seed']].append((float(r['reference_ppm']),float(r['predicted_ppm'])))
    stats={k:metrics(v) for k,v in groups.items()}
    n=0
    for r in read(root/'tables234_manuscript.csv'):
        if args.predictions and r['group']!='main_table2':continue
        values=[a[r['metric']] for k,a in stats.items() if k[:3]==(r['group'],r['task'],r['split'])]
        if len(values)!=3:raise ValueError((r,len(values)))
        check(values,r['display_mean'],r['display_sd']);n+=1
    print('Tables 2–4:',n,'mean/SD pairs agree to four decimals')
    groups=defaultdict(list)
    for p in root.glob('table5_seed*.csv'):
        seed=p.stem.split('seed')[-1]
        for r in read(p):groups[r['train_source'],r['task'],seed].append((float(r['true_ppm']),float(r['pred_ppm'])))
    stats={k:metrics(v) for k,v in groups.items()}
    n=0
    for r in read(args.artifact_root/'configs/table5_seeds.csv'):
        for k,c in [('mae','micro_mae_ppm'),('rmse','rmse_ppm'),('r2','r2')]:
            values=[stats[r['source'],r['task'],s][k] for s in r['seeds'].split('/')]
            check(values,r[c+'_mean'],r[c+'_sample_sd']);n+=1
    print('Table 5:',n,'mean/SD pairs agree to four decimals')
if __name__=='__main__':main()
