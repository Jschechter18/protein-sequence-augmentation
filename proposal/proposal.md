# Capstone Proposal

## Data Augmentation (Mutations) for Protein Classification

### Proposed by: Joshua Schechter

#### Email: joshua.d.schechter@gmail.com

#### Advisor: Amir Jafari

#### The George Washington University, Washington DC

#### Data Science Program

## 1 Objective:

This project will study when mutation-based data augmentation helps or hurts protein sequence classification. The central research question is: at what mutation rate does protein sequence augmentation transition from beneficial to harmful, and does that threshold depend on the amount of available training data?

The hypothesis is that mild mutation-based augmentation will improve performance most in low-data settings, but increasingly aggressive mutation rates will eventually damage biologically meaningful sequence features and reduce classification performance. The project will focus on protein classification tasks using pretrained protein language model embeddings and controlled augmentation experiments.

![Figure 1: Example figure](2026_Summer_7.png)
_Figure 1: Caption_

## 2 Dataset:

The project will use PEER protein datasets, prioritizing tasks that provide protein sequence records with idx, sequence, label, and split fields. The planned datasets are:

1. Protein localization: multiclass classification task.
2. Protein solubility: binary classification task.

PEER will be used as a benchmark source, meaning the project will use the provided data and train/validation/test splits rather than creating custom splits from scratch. This supports reproducibility and allows the study to focus on the effect of augmentation rather than dataset construction.

## 3 Rationale:

Protein sequences can be represented as strings of amino acids, which makes mutation-style augmentation a natural idea. However, unlike many image augmentations, sequence mutations can change biologically important motifs, localization signals, or functional regions. This creates a practical question: mutation augmentation may help regularize a model when data is limited, but too much mutation may make the augmented examples biologically misleading.

This project is designed to isolate that tradeoff. By holding the model backbone fixed and systematically varying data fraction, mutation type, and mutation rate, the study can identify where augmentation stops helping and starts hurting. The result would be useful for researchers who want to know when protein sequence augmentation is appropriate, especially in low-data biological machine learning settings.

## 4 Approach:

Model architecture:

- Use a pretrained ESM-2 protein language model as the sequence encoder.
- Freeze the ESM-2 backbone to isolate the effect of augmentation.
- Add and train a lightweight classification head on top of the ESM-2 representation.

Experimental variables:

- Dataset: PEER localization and PEER solubility.
- Training data fraction: 10%, 25%, 50%, and 100% of the training set.
- Augmentation type: BLOSUM/conservative substitution and random substitution.
- Mutation rate: 0%, 5%, 10%, 15%, 20%, and 30%.
- Random seeds: 2 or 3 seeds for reproducibility.

Baseline:

- The no-augmentation condition, represented by 0% mutation rate, will serve as the baseline for each dataset, training fraction, and random seed.

Training procedure:

1. Load the PEER dataset and preserve the official train/validation/test splits.
2. Subsample the training split according to the selected training data fraction.
3. Generate augmented training examples according to the selected augmentation type and mutation rate.
4. Embed sequences with the frozen ESM-2 model.
5. Train the classification head.
6. Evaluate on the unchanged validation/test split.
7. Repeat across seeds and aggregate results.

Analysis plan:

- Report F1, precision, recall, accuracy, and Cohen's kappa.
- Plot mutation rate vs. performance for each augmentation type.
- Plot mutation rate vs. performance for each training data fraction.
- Identify the mutation-rate threshold where performance becomes worse than the no-augmentation baseline for the same configuration.
- Compare thresholds across training fractions to test whether smaller datasets benefit from higher mutation rates.
- Compare augmentation types to determine whether conservative substitutions are safer or more useful than random substitutions.
- Report mean +/- standard deviation across seeds.
- Run statistical significance tests for observed improvements over baseline where appropriate.

## 5 Timeline:

Week 1: Finalize research question, confirm PEER dataset access, and verify that localization and solubility datasets can be exported with idx, sequence, label, and split fields.
Week 2: Build dataset loading and preprocessing scripts. Confirm train/validation/test split integrity and produce small sanity-check CSVs.
Week 3: Implement baseline ESM-2 frozen-encoder pipeline with a trainable classification head. Run first baseline experiments on one dataset.
Week 4: Implement augmentation methods: random substitution and BLOSUM/conservative substitution. Validate mutation rates and ensure augmented sequences remain valid amino acid sequences.
Week 5: Run pilot experiments across a reduced grid to check feasibility, runtime, memory usage, and metric logging.
Week 6: Run full experiments for the localization dataset across data fractions, mutation rates, augmentation types, and seeds.
Week 7: Run full experiments for the solubility dataset across data fractions, mutation rates, augmentation types, and seeds.
Week 8: Aggregate results, compute mean +/- standard deviation across seeds, and generate initial plots.
Week 9: Identify augmentation thresholds where performance drops below baseline. Compare thresholds across training data fractions.
Week 10: Conduct statistical significance tests for meaningful improvements or degradations relative to baseline.
Week 11: Write results and analysis sections. Interpret whether low-data settings benefit more from augmentation.
Week 12: Refine experiments if needed, rerun failed configurations, and verify reproducibility.
Week 13: Draft final report/paper and prepare figures, tables, and appendix details.
Week 14: Finalize report, code repository, reproducibility instructions, and presentation materials.

TOTAL: 14 weeks (one semester)

KEY MILESTONES:

- Week 2: Dataset extraction and preprocessing complete.
- Week 4: Baseline model and augmentation methods implemented.
- Week 5: Pilot experiment completed and grid finalized.
- Week 7: Full experiment runs completed for both datasets.
- Week 9: Main figures and threshold analysis completed.
- Week 12: Reproducibility checks and reruns completed.
- Week 14: Final paper/report, code, and presentation completed.

DELIVERABLES BY WEEK 14:

- Clean GitHub repository with dataset processing, augmentation, training, evaluation, and plotting scripts.
- Reproducible experiment configuration files.
- Final results tables with metrics across datasets, seeds, data fractions, augmentation types, and mutation rates.
- Plots showing mutation rate vs. performance and threshold comparisons.
- Final written report/paper describing motivation, methods, results, limitations, and future work.
- Final presentation summarizing the study.

## 6 Expected Number Students:

RECOMMENDED: 1-2 students

ROLE DISTRIBUTION FOR 1 STUDENT:
Student 1:

- Responsibilities: dataset extraction, augmentation implementation, ESM-2 modeling pipeline, experiment execution, statistical analysis, plots, final report, and presentation.

ROLE DISTRIBUTION FOR 2 STUDENTS:
Student 1:

- Responsibilities: dataset extraction, preprocessing, augmentation implementation, experiment configuration, and reproducibility infrastructure.

Student 2:

- Responsibilities: ESM-2 modeling pipeline, training runs, metric evaluation, statistical analysis, plots, and final paper/presentation support.

## 7 Possible Issues:

TECHNICAL CHALLENGES AND SOLUTIONS:

1. Dataset access or formatting issues:

- Challenge: PEER datasets may require repository-specific loaders or wrappers.
- Mitigation: Build a simple export script that saves idx, sequence, label, and split to CSV before modeling.

2. Runtime and GPU constraints:

- Challenge: ESM-2 embeddings may be expensive to compute repeatedly across many configurations.
- Mitigation: Cache embeddings when possible, start with a smaller ESM-2 variant, and use pilot runs to estimate full-grid runtime.

3. Experiment grid size:

- Challenge: The full grid can grow quickly across datasets, seeds, fractions, augmentation types, and mutation rates.
- Mitigation: Begin with 2 seeds and a reduced pilot grid, then expand only after the pipeline is stable.

4. Biological validity of mutations:

- Challenge: Random substitutions may produce unrealistic sequences or destroy important motifs.
- Mitigation: Compare random substitutions against BLOSUM/conservative substitutions and explicitly analyze when performance drops below baseline.

5. Class imbalance:

- Challenge: Localization or solubility labels may be imbalanced.
- Mitigation: Use macro-F1 as a primary metric, report precision/recall, and consider class-weighted loss if imbalance is severe.

6. Statistical power:

- Challenge: With only 2-3 seeds, statistical tests may be limited.
- Mitigation: Report confidence carefully, include mean +/- standard deviation, and treat significance testing as supporting evidence rather than the only basis for conclusions.

RISK MITIGATION TIMELINE:

- Weeks 1-2: Confirm dataset access and export format.
- Weeks 3-4: Validate baseline model and augmentation correctness.
- Weeks 5-6: Run pilot grid and reduce scope if needed.
- Weeks 7-8: Monitor full experiment runtime and cache reusable outputs.
- Weeks 9-10: Check metric consistency and rerun failed configurations.
- Weeks 11-12: Validate statistical analysis and figure generation.
- Weeks 13-14: Final reproducibility pass and documentation cleanup.

## Contact

- Author: Amir Jafari
- Email: [ajafari@gwu.edu](mailto:ajafari@gwu.edu)
- GitHub: [https://github.com/Jschechter18/protein-sequence-augmentation](https://github.com/https://github.com/Jschechter18/protein-sequence-augmentation)
