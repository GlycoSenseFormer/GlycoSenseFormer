"""Evaluate exactly the 24 four-source checkpoints used in manuscript Table 5."""
import argparse,csv
from collections import defaultdict
from pathlib import Path
from verify_results import metrics,check,read
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-root',type=Path,default=ROOT,help='Directory with the evaluation data and run manifest')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/table5_from_weights')
    ap.add_argument('--checkpoint-dir', type=Path, default=ROOT/'checkpoints', help='Directory containing the optional manuscript checkpoints')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--list-runs',action='store_true')
    args=ap.parse_args()
    manifest = args.artifact_root/'configs/table5_seeds.csv'
    if not manifest.is_file():ap.error(f'Run manifest not found: {manifest}. Supply --artifact-root.')
    basis=read(manifest)
    schedule=defaultdict(list)
    for row in basis:
        for seed in row['seeds'].split('/'):schedule[seed,row['task']].append(row['source'])
    if args.list_runs:
        for (s,t),src in sorted(schedule.items()):print(f'Seed {s} {t}:',', '.join(src))
        return
    weight_root = args.checkpoint_dir.resolve() / 'forcefield329'
    if not weight_root.is_dir():
        ap.error(f'Missing {weight_root}. Supply --checkpoint-dir with the archived checkpoints directory.')
    import evaluate_glycosenseformer_forcefields_clean_split as evaluator
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    estimates={}
    for (seed,task),sources in sorted(schedule.items()):
        folder=out/('seed'+seed)/task
        cli=['--run-dir',str(weight_root/('seed'+seed)),'--dataset-dir',str(args.artifact_root.resolve()/'data/external100'),'--out-dir',str(folder),'--sources',*sources,'--tasks',task,'--device',args.device,'--equiformer-v3-root',str(ROOT/'src/equiformer_v3')]
        evaluator.run(evaluator.build_parser().parse_args(cli))
        per=defaultdict(list)
        for r in read(folder/'predictions.csv'):per[r['train_source']].append((float(r['true_ppm']),float(r['pred_ppm'])))
        for source,values in per.items():estimates[seed,task,source]=metrics(values)
    from statistics import mean,stdev
    with (out/'summary.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['nucleus','source','seeds','metric','mean','sample_sd'])
        for r in basis:
            for key,col in [('mae','micro_mae_ppm'),('rmse','rmse_ppm'),('r2','r2')]:
                vals=[estimates[s,r['task'],r['source']][key] for s in r['seeds'].split('/')]
                writer.writerow([r['task'],r['source'],r['seeds'],key,mean(vals),stdev(vals)])
                check(vals,r[col+'_mean'],r[col+'_sample_sd'])
    print('Table 5: all 24 mean/SD pairs reproduced from fresh predictions.')
if __name__=='__main__':main()
