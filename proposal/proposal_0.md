
# Capstone Proposal
## Protein Sequence Augmentations in Latent Space
### Proposed by: Joshua Schechter and Ashley Gyapomah
#### Email: j.schechter@gwu.edu, ashleygya03@gwu.edu
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  

            This project will test whether adding controlled noise to the latent vector of a trained protein sequence autoencoder, then decoding that perturbed vector back into a protein sequence, can generate useful protein sequence augmentations. The main goal is to determine whether latent-space augmentation can improve protein sequence classification performance compared with no augmentation and direct mutation-based augmentation baselines.

            The project is motivated by the label-preservation problem in protein sequence augmentation. In many protein classification tasks, an augmented sequence inherits the original label, but even small amino acid changes can potentially alter biological meaning. This project will evaluate whether autoencoder-based latent-space perturbation can produce augmented sequences that are more useful, more protein-like, and potentially more label-preserving than direct input-level mutations.

            

![Figure 1: Example figure](2026_Summer_0.png)
*Figure 1: Caption*

## 2 Dataset:  

            The project will use the PEER benchmark as the primary source of protein sequence classification datasets. The initial focus will be on at least one protein classification task, such as protein solubility or protein localization, depending on data availability, task suitability, and implementation feasibility.

            The dataset should include protein sequences, labels, and predefined train, validation, and test splits if available. The clean test set will be kept fixed across all experimental conditions so that each augmentation strategy is evaluated fairly under the same downstream classification setup.

            

## 3 Rationale:  

            Protein sequence classification often occurs in low-data settings, where labeled biological data can be expensive, time-consuming, or difficult to obtain. Data augmentation is useful in these settings because it can increase the amount of training data and potentially improve downstream model generalization.

            However, direct input-level augmentation of amino acid sequences can be risky. A random amino acid substitution, insertion, deletion, or even a conservative mutation may change biologically meaningful properties of the protein. For example, if a protein is originally labeled as soluble, a single amino acid change could theoretically alter its folding, stability, localization, or solubility. Since the true biological label of the mutated sequence is usually unknown, the augmented sequence may be incorrectly assigned the original label. This violates the label-preservation assumption and can hurt model performance.

            Latent-space augmentation may be worth testing because a trained protein sequence autoencoder or variational autoencoder learns a compressed numerical representation of protein sequences. If this learned latent space captures useful biological structure, then small controlled perturbations in latent space may produce decoded sequences that remain closer to the original protein's label-relevant properties than direct input-level mutations. This does not guarantee label preservation, but it creates a testable hypothesis: latent-space noise augmentation may produce more useful training examples than direct mutation baselines.

            

## 4 Approach:  

            The project will compare latent-space noise augmentation against direct mutation-based augmentation baselines for protein sequence classification.

            Experimental plan:

            1. Train a protein sequence autoencoder or variational autoencoder as well as possible.
               - The model will learn to encode protein sequences into latent vectors and decode latent vectors back into protein-like sequences.
               - Autoencoder quality will be important because poor reconstruction or poor decoding may limit the usefulness of latent-space augmentation.

            2. Use the encoder to map original protein sequences into latent vectors.
               - Each original protein sequence will be passed through the encoder to obtain a latent representation z.

            3. Add small controlled noise to the latent vector z.
               - Noise will be applied at a fixed rate or magnitude for simplicity.
               - The goal is to slightly perturb the learned representation without moving too far away from the original sequence's latent neighborhood.

            4. Decode the perturbed latent vector back into a new protein sequence.
               - The decoder will generate a new protein-like sequence from the noisy latent vector.
               - This decoded output will be treated as the latent-space augmented protein sequence.

            5. Train classifiers under four augmentation settings.
               - The initial classifier will use ESM-2 or an ESM-2-based classification pipeline.
               - Four versions of the classifier will be trained:
                 i. No augmentation
                 ii. Random substitution
                 iii. BLOSUM/conservative mutation
                 iv. Latent-space noise augmentation

            6. Use a fixed data rate and fixed augmentation/mutation rate for simplicity.
               - To keep the project manageable, the first experiment will use one fixed training-data fraction and one fixed augmentation intensity.
               - Additional sweeps over data fraction, mutation rate, or noise magnitude can be added if time allows.

            7. Evaluate each classifier on the same clean test set.
               - The test set will not contain augmented sequences.
               - This allows the project to evaluate whether each augmentation method improves generalization to real, unmodified protein sequences.

            8. Compare results using useful metrics.
               - F1 score will be the primary classification metric.
               - VEP-style scores may be used as an indicator of whether generated or mutated sequences appear biologically plausible, but these scores will not be treated as ground truth.
               - Additional metrics may include accuracy, precision, recall, sequence identity to the original sequence, edit distance, BLOSUM substitution score, reconstruction quality, invalid sequence rate, and decoded sequence quality.

            9. Analyze whether latent-space augmentation provides a measurable benefit.
               - The ideal result would be that latent-space noise augmentation achieves the best clean test-set F1 score and favorable VEP-style or sequence-quality indicators.
               - A negative result would still be informative if latent-space augmentation does not outperform direct mutation baselines or produces low-quality decoded sequences.

            

## 5 Timeline:  

            Week 1:    Finalize project scope, select PEER classification task, review dataset format, define evaluation protocol, and confirm train/validation/test split strategy.
            Week 2:    Build dataset loading, preprocessing, tokenization, and baseline data-cleaning pipeline for the selected PEER task.
            Week 3:    Implement no-augmentation classifier baseline using ESM-2 or an ESM-2-based sequence classification pipeline.
            Week 4:    Implement random substitution and BLOSUM/conservative mutation augmentation baselines at a fixed augmentation rate.
            Week 5:    Train and validate the baseline classifiers under no augmentation, random substitution, and BLOSUM/conservative mutation settings.
            Week 6:    Design and implement the protein sequence autoencoder or variational autoencoder architecture, including encoder, latent vector, decoder, and sequence reconstruction workflow.
            Week 7:    Train the autoencoder/VAE, monitor reconstruction quality, tune basic hyperparameters, and inspect decoded sequence validity.
            Week 8:    Implement latent-space noise augmentation by encoding sequences, perturbing latent vector z, decoding the perturbed vector, and saving generated augmented sequences.
            Week 9:    Train the ESM-2 classifier using latent-space noise augmentation under the same fixed data rate and augmentation rate as the baselines.
            Week 10:   Evaluate all classifier variants on the same clean validation and test sets using F1 score and any additional classification metrics.
            Week 11:   Compute sequence-quality and biological plausibility indicators, including VEP-style scores as indicators only, sequence identity, edit distance, BLOSUM score, invalid sequence rate, and reconstruction-related metrics.
            Week 12:   Analyze results across the four augmentation settings and determine whether latent-space augmentation improves clean test performance compared with direct mutation baselines.
            Week 13:   Prepare figures, tables, written interpretation, limitations, and discussion of label-preservation concerns, VEP-score limitations, and autoencoder-quality limitations.
            Week 14:   Finalize report, presentation, code cleanup, reproducibility documentation, and recommendations for future work.

            TOTAL: 14 weeks (one semester)

            KEY MILESTONES:
            - Week 2:  Dataset and preprocessing pipeline completed
            - Week 4:  Mutation-based augmentation baselines implemented
            - Week 5:  Initial classifier baselines trained
            - Week 7:  Autoencoder/VAE trained and evaluated for reconstruction quality
            - Week 9:  Latent-space augmentation classifier trained
            - Week 12: Full metric comparison completed
            - Week 14: Final report, presentation, and code deliverables completed

            DELIVERABLES BY WEEK 14:
            - Clean dataset preparation pipeline
            - No-augmentation classifier baseline
            - Random substitution augmentation baseline
            - BLOSUM/conservative mutation augmentation baseline
            - Protein sequence autoencoder or VAE
            - Latent-space noise augmentation pipeline
            - Classifier trained with latent-space augmented data
            - Clean test-set evaluation across all augmentation settings
            - Analysis of F1 score, VEP-style indicator scores, sequence-quality metrics, and limitations
            - Final written research summary and presentation

            


## 6 Expected Number Students:  

            RECOMMENDED: 2 students

            ROLE DISTRIBUTION FOR 2 STUDENTS:

            Student 1:
            - Responsibilities: Lead data setup and autoencoder training
            - Prepare and clean the PEER benchmark dataset
            - Build sequence preprocessing and tokenization workflows
            - Implement and train the protein sequence autoencoder or VAE
            - Evaluate reconstruction quality and decoded sequence validity
            - Generate latent-space augmented sequences

            Student 2:
            - Responsibilities: Lead classification pipeline and augmentation pipeline
            - Implement the ESM-2 classifier or ESM-2-based classification workflow
            - Implement no-augmentation, random substitution, and BLOSUM/conservative mutation baselines
            - Train classifiers under the four augmentation settings
            - Evaluate classification metrics on the clean validation and test sets
            - Support comparison of augmentation methods

            Shared responsibilities:
            - Experimental design
            - Data cleaning
            - Augmentation strategy decisions
            - Metric selection
            - Analysis and interpretation
            - Literature review
            - Writing final report and preparing final presentation

            

## 7 Possible Issues:  

            TECHNICAL CHALLENGES AND SOLUTIONS:

            1. Latent-space augmentation may not outperform baselines.
            - This is a possible negative result. The project should frame this as an empirical question rather than assuming latent-space augmentation will work.
            - If latent-space augmentation performs worse than random substitution or BLOSUM/conservative mutation, that finding can still help clarify the limits of the method.

            2. Decoded sequences may not preserve labels.
            - The central challenge is that generated sequences inherit the original label, but their true biological label is usually unknown.
            - Clean test-set performance can suggest whether the augmented data was useful, but it does not prove that every decoded sequence is label-preserving.

            3. VEP scores are not ground truth.
            - VEP-style scores may provide useful indicators of whether mutations or generated sequences seem plausible or harmful.
            - These scores should not be treated as definitive biological labels.

            4. Autoencoder training quality may limit results.
            - If the autoencoder or VAE does not reconstruct protein sequences well, or if the decoder produces invalid or low-quality sequences, latent-space augmentation may fail for model-quality reasons rather than because the core idea is invalid.
            - Reconstruction quality, decoded sequence validity, and sequence similarity should be monitored carefully.

            5. Data leakage concerns may arise between autoencoder training data and classifier training/test data.
            - The project should investigate whether sequences used to train the autoencoder can overlap with sequences used to train or test the classifier.
            - To avoid leakage concerns, the safest approach is to prevent test-set sequences from being used in any way that could influence classifier training or augmentation generation.

            6. Decoded protein sequences may be invalid or low quality.
            - The decoder may produce sequences with invalid amino acid tokens, unrealistic patterns, excessive similarity to the original sequence, or excessive drift away from the original sequence.
            - The project should track invalid sequence rate, sequence identity, edit distance, and other quality indicators.

            7. The selected PEER task may affect results.
            - Some protein classification labels may be more sensitive to sequence changes than others.
            - Results from one task, such as solubility or localization, may not generalize to all protein classification tasks.

            RISK MITIGATION TIMELINE:
            - Weeks 1-2: Confirm dataset splits, define leakage-prevention strategy, and verify that the selected PEER task is feasible.
            - Weeks 3-4: Establish strong no-augmentation and mutation-based baselines before relying on latent-space augmentation.
            - Weeks 5-6: Begin autoencoder/VAE implementation early enough to identify training or reconstruction issues.
            - Weeks 7-8: Inspect decoded sequences and filter invalid outputs before classifier training.
            - Weeks 9-10: Evaluate all methods on the same clean validation and test sets.
            - Weeks 11-12: Use VEP-style scores and sequence-quality metrics as supporting indicators only.
            - Weeks 13-14: Clearly document limitations, negative results, and future improvements.

            


## Contact
- Author: Amir Jafari
- Email: [ajafari@gwu.edu](mailto:ajafari@gwu.edu)
- GitHub: [https://github.com/amir-jafari](https://github.com/https://github.com/amir-jafari)
