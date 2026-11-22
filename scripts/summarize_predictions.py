"""Aggregate fresh atom predictions by nucleus, model and chain length."""
import argparse,csv
import statistics as st
from collections import defaultdict
from pathlib import Path
from verify_results import metrics,read

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=defaultdict(list)
    for r in read(args.predictions):
        for s in ['overall',r['split']]:
            raw[r['model'],r['task'],s,r['seed']].append((float(r['reference_ppm']),float(r['prediction_ppm'])))
    groups=defaultdict(list)
    for (m,t,s,seed),sites in raw.items():
        for k,v in metrics(sites).items():groups[m,t,s,k].append((seed,len(sites),v))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','nucleus','split','metric','seeds','sites_per_run','mean','sample_sd'])
        for key,items in sorted(groups.items()):
            ns={n for _,n,_ in items}
            if len(ns)!=1:raise ValueError(f'Inconsistent site counts for {key}')
            v=[a for _,_,a in items];w.writerow([*key,'/'.join(s for s,_,_ in sorted(items)),ns.pop(),st.mean(v),st.stdev(v) if len(v)>1 else ''])
if __name__=='__main__':main()

