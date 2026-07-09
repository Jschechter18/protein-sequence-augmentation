
# Protein Sequence Classification with ESM-2

This projects benchmark pretrained ESM-2 protein language models for protein sequence classification

Current task:
- Solubility (binary classification)
- Localization (coming soon)

Classifier Head: 
- 1D CNN
- LSTM
- GRU

Fine-tuning strategies:
- Frozen ESM-2
- Partial backbone unfreezing
- Full backbone 

---------------
##### Pipeline
Stage 0
──────────────
Frozen ESM-2
↓

CNN / LSTM / GRU

Stage 1
──────────────
Last transformer layer unfrozen
↓

CNN

Stage Full
──────────────
Entire ESM-2 backbone trainable
↓

CNN

---------------
##### Directory Organization

Code/results/esm2/
├── solubility/
│   ├── cnn/
│   │   ├── stage0_frozen/
|   |   ├── stage1_unfreeze_last1/ 
│   │   └── stage_full_unfreeze/
│   ├── gru/
|   |   └── stage0_frozen/
│   └── lstm/
|   |   └── stage0_frozen/
|   |
└── README.md

---------------
##### Current Results

| Experiment              | Best Val Accuracy | Best Val F1 | Best Epoch | 
| :---------------- | :------: | ----: |
| Frozen + CNN       |   76.97%   | 75.62% | 8
| Frozen + GRU           |   76.68%   | 75.94% | 4
| Frozen + LSTM    |  75.99%   | 75.21% | 3
| Last Layer + CNN |  77.36%  | 76.06% | 6
| Full Unfrozen + CNN |  77.31%  | 76.29% | 8


