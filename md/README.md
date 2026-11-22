# Molecular dynamics and conformer selection

Use GROMACS with prepared simulation inputs and the corresponding force-field files. The workflow scripts are in md/scripts. Supply the directory that contains md/systems.csv and the corresponding run inputs to replay_md.py using --artifact-root.

For farthest-point sampling, provide a manifest of candidate conformers to scripts/build_glycosenseformer_md_fps.py.

```bash
python md/scripts/replay_md.py --help
python scripts/build_glycosenseformer_md_fps.py --help
```
