# GlycoSenseFormer

**A Hierarchical SE(3)-Equivariant Model for Carbohydrate NMR Chemical Shift Prediction**

![GlycoSenseFormer overview](assets/TOC.png)

GlycoSenseFormer is a conformer-aware deep-learning framework for predicting carbohydrate $^{13}\mathrm{C}$ and $^{1}\mathrm{H}$ NMR chemical shifts. It combines an SE(3)-equivariant atom-geometry encoder with hierarchical atom--residue message passing and residual cross-attention to integrate local three-dimensional geometry with monosaccharide-level context.

## Highlights

- Multi-conformer learning for carbohydrate NMR chemical-shift prediction
- SE(3)-equivariant encoding of atom-level three-dimensional geometry
- Hierarchical atom--residue message passing for glycan context
- Residual cross-attention for geometric and residue-level feature fusion
- External evaluation for both $^{13}\mathrm{C}$ and $^{1}\mathrm{H}$ prediction tasks
- Analysis of conformer ensembles generated with multiple force fields

## Repository status

> **Code release in preparation.**

The source code is currently being organized and documented. Training and evaluation scripts, model configurations, preprocessing utilities, and reproducibility instructions will be added in a forthcoming update.

## Planned contents

- Data preprocessing and conformer-generation workflows
- GlycoSenseFormer model implementation
- Training and independent-evaluation scripts
- Experiment configurations and random-seed settings
- Baseline and ablation-study workflows
- Force-field conformer analysis and figure-generation scripts
- Reproduction instructions for the manuscript results

## Citation

The manuscript is currently under preparation. Citation information will be added when it becomes available.

## License

License information will be provided with the code release.
