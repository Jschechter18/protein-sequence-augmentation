# Capstone Proposal

## Data Augmentation (Mutations) for Protein Classification

### Proposed by: Joshua Schechter, Ashley Gyapomah

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

- The no-augmentation condition, represented by 0% mutation rate (or no added augmentation depending on what we can find from the dataset existing mutations), will serve as the baseline for each dataset, training fraction, and random seed.

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

Week 1: Structure github repo, finalize research question, confirm PEER dataset access, and verify that localization and solubility datasets can be exported with idx, sequence, label, and split fields. Build dataset loading and preprocessing scripts. Confirm train/validation/test split integrity and produce small sanity-check CSVs.
Week 2: Implement baseline ESM-2 frozen-encoder pipeline with a trainable classification head. Run first baseline experiments on one dataset.
Week 3: Implement augmentation methods: random substitution and BLOSUM/conservative substitution. Validate mutation rates and ensure augmented sequences remain valid amino acid sequences. Structure the entire pipeline for experimentation to allow for easy experimentation runs.
Week 4: Run pilot experiments across a reduced grid to check feasibility, runtime, memory usage, and metric logging.
Week 5: Run full experiments for the localization dataset across data fractions, mutation rates, augmentation types, and seeds.
Week 6: Run full experiments for the solubility dataset across data fractions, mutation rates, augmentation types, and seeds.
Week 7: Aggregate results, compute mean +/- standard deviation across seeds, and generate initial plots.
Week 8: Identify augmentation thresholds where performance drops below baseline. Compare thresholds across training data fractions.
Week 9: Conduct statistical significance tests for meaningful improvements or degradations relative to baseline.
Week 10: Write results and analysis sections. Interpret whether low-data settings benefit more from augmentation.
Week 11: Refine experiments if needed, rerun failed configurations, and verify reproducibility.
Week 12: Draft final report/paper and prepare figures, tables, and appendix details.
Week 13: Finalize report, code repository, reproducibility instructions, and presentation materials.

TOTAL: 13 weeks (estimated the last week of summer term)

KEY MILESTONES:

- Week 1: (week of May 18th) Dataset extraction and preprocessing complete.
- Week 3: Baseline model and augmentation methods implemented.
- Week 4: Pilot experiment completed and grid finalized.
- Week 6: Full experiment runs completed for both datasets.
- Week 9: Main figures and threshold analysis completed.
- Week 11: Reproducibility checks and reruns completed.
- Week 13: Final paper/report, code, and presentation completed.

DELIVERABLES BY WEEK 13:

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

- Responsibilities: dataset extraction, augmentation implementation, ESM-2 modeling pipeline, experiment execution, statistical analysis, plots, final report, and presentation

ROLE DISTRIBUTION FOR 2 STUDENTS:
Student 1:

- Responsibilities: datasets extraction, preprocessing, solubility training set runs - and the downstream tasks for this dataset (statistical analysis, plots, results), paper and presentation support

Student 2:

- Responsibilities: ESM-2 model pipeline, augmentation pipeline, experiment configuration, localization training set runs - and the downstream tasks for this dataset (statistical analysis, plots, results), paper and presentation support

## 7 Possible Issues:

TECHNICAL CHALLENGES AND SOLUTIONS:

1. Dataset access or formatting issues:

- Challenge: PEER datasets do require repository-specific loaders or wrappers.
- Mitigation: Build a simple export script that saves idx, sequence, label, and split to CSV before modeling.

2. Runtime and GPU constraints:

- Challenge: ESM-2 embeddings may be expensive to compute repeatedly across many configurations. This proposal was written under the assumption we would have ample access to GPU's.

3. Experiment grid size:

- Challenge: The full grid can grow quickly across datasets, seeds, fractions, augmentation types, and mutation rates.
- Mitigation: Decide how many fraction sizes, augmentation types, and mutation rates to test.

4. Biological validity of mutations:

- Challenge: Random substitutions may produce unrealistic sequences or destroy important motifs.
- Mitigation: Compare random substitutions against BLOSUM/conservative substitutions and explicitly analyze when performance drops below baseline. There should be ways to force BLOSUM62 mutations.

5. Class imbalance:

- Challenge: Localization or solubility labels may be imbalanced.
- Mitigation: Use macro-F1 as a primary metric, report precision/recall, and consider class-weighted loss if imbalance is severe.
- Note: There could be ways to add this to the experimental setup. The mutations we are experimenting could help with class imbalance.

6. Statistical power:

- Challenge: With only 2-3 seeds, statistical tests may be limited.
- Mitigation: Report confidence carefully, include mean +/- standard deviation, and treat significance testing as supporting evidence rather than the only basis for conclusions.

## Contact

- Author: Josh Schechter, Ashley Gyapomah
- Email: [j.schechter@gwu.edu]
- GitHub: [https://github.com/Jschechter18/protein-sequence-augmentation](https://github.com/https://github.com/Jschechter18/protein-sequence-augmentation)
