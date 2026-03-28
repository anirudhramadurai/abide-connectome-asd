# connectome-gnn

**Graph-based classification of Autism Spectrum Disorder from resting-state fMRI functional connectivity data**

---

## Overview

Using publicly available resting-state fMRI data from 303 participants across three ABIDE sites (NYU, USM, UCLA), I built a five-script Python pipeline to classify ASD from whole-brain functional connectivity patterns. Each participant's brain is represented as a graph of 200 regions connected by functional correlations. The pipeline applies ComBat site harmonization to remove scanner-driven variance, extracts 44 biologically grounded graph-level features, and trains a gradient-boosted classifier with 5-fold cross-validation.

Without site harmonization, classification performance is at chance (AUC = 0.51), which is the expected result: scanner differences between sites dominate the connectivity signal and mask the biological ASD effect. After ComBat harmonization, AUC improves substantially, and the node importance analysis reveals that sensorimotor (SMN), frontoparietal (FPN), and limbic network features drive the classification, consistent with published findings on sensorimotor integration, executive function, and social cognition differences in ASD [8, 9].

A note on `models/gcn_numpy.py`: this file contains an educational from-scratch implementation of the graph convolution mathematics (Kipf & Welling, 2017). It is not the production classifier. It exists to make the GCN forward pass inspectable without requiring PyTorch. The actual classifier is gradient boosting on graph-derived features. A PyTorch Geometric GCN with autograd is listed as the primary future direction.

---

## What is this project?

This project applies a reproducible Python pipeline to publicly available neuroimaging data to investigate whether ASD can be detected from the pattern of functional connections between brain regions, when those connections are modeled as a graph.

The full pipeline runs in approximately 15 minutes on a standard laptop, from automatic data download to six figures.

---

## The Biology (plain language first)

### What is ASD and why study brain connectivity?

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition affecting approximately 1-2% of the global population [1], characterized by differences in social communication and restricted or repetitive behavior patterns. It is a spectrum, meaning the condition presents very differently across individuals, ranging from people who are largely independent to those who require substantial daily support.

Unlike conditions such as Alzheimer's disease, where specific brain regions are visibly lost, ASD does not produce obvious structural damage detectable by standard brain imaging. This has led researchers to focus instead on how brain regions **communicate with each other**: the hypothesis is that ASD arises not from failure of any single region, but from disrupted coordination across the brain's large-scale networks [1].

### What is resting-state fMRI and functional connectivity?

**Functional MRI (fMRI)** measures brain activity indirectly. When neurons in a region fire, they demand more oxygen-rich blood. The MRI scanner can detect the resulting change in the ratio of oxygenated to deoxygenated hemoglobin; this is called the **BOLD signal** (Blood Oxygen Level Dependent). The more active a region, the stronger its BOLD signal.

In **resting-state fMRI**, there is no task. Participants simply lie still in the scanner with their eyes open or closed. Even at rest, the brain is highly active: different regions spontaneously fluctuate in activity, and importantly, certain regions tend to fluctuate *together*, activating and deactivating in synchrony.

**Functional connectivity (FC)** is the statistical correlation between the BOLD time series of two brain regions across a scan. If region A and region B consistently go up and down together over the course of several minutes, their functional connectivity is high (correlation close to +1). If they consistently do the opposite, they are anti-correlated (close to -1). A near-zero correlation means the two regions operate independently.

By computing this correlation for every pair of 200 brain regions, we produce a 200x200 **functional connectome**, a matrix capturing the brain's entire large-scale coordination structure for that individual subject.

### What does the ASD connectivity literature say?

The most replicated finding in ASD neuroimaging is **long-range underconnectivity**: regions far apart in the brain, particularly within the Default Mode Network and between association cortices, show weaker-than-typical synchronization in ASD [3, 4]. This is thought to reflect disrupted information integration across distributed brain networks.

There is also evidence of **local overconnectivity** in sensorimotor regions, meaning elevated synchrony between nearby regions, which may relate to sensory hypersensitivity and motor coordination differences commonly observed in ASD [8].

These connectivity differences are real but subtle at the group level, and highly variable across individuals. This is a central reason why ASD classification from fMRI is difficult, and why site harmonization is a prerequisite for recovering a meaningful signal in multi-site datasets.

### Why model the connectome as a graph?

A 200x200 correlation matrix contains 19,900 unique pairwise connections. Using all of them as features for a 303-subject dataset would give a model roughly 66 times more features than samples, a situation called high-dimensional underpowering, where a classifier will overfit to noise and fail completely on new data.

A **graph** offers a more structured alternative. In graph theory, a graph is simply a collection of **nodes** (entities) connected by **edges** (relationships). Here, the 200 brain regions are nodes, and a connection is drawn between two regions if their correlation exceeds a threshold. Each node then has a compact set of local properties (how many connections it has, how tightly clustered those connections are) that summarize its role in the network without using all 19,900 raw edge values. This reduces the feature space from 19,900 to 44 biologically interpretable values per subject.

---

## Research Questions

**Primary:** Can ASD be classified from resting-state fMRI functional connectivity patterns using graph-based features, and does ComBat site harmonization recover biological signal that is masked by scanner-site variance in the unharmonized baseline?

**Secondary:** Which brain networks and connectivity properties drive classification performance, and are the network differences consistent with the published ASD connectivity literature?

---

## Data

Resting-state fMRI data were obtained from the **ABIDE Preprocessed Connectomes Project (ABIDE-PCP)** [1].

ABIDE (Autism Brain Imaging Data Exchange) is a publicly available, no-registration-required repository of resting-state fMRI data from ASD and neurotypical control participants collected across multiple sites worldwide. This analysis uses the preprocessed CC200 parcellation outputs from three sites.

| Field | Value |
|---|---|
| Source | ABIDE Preprocessed Connectomes Project (ABIDE-PCP) |
| Access | Fully public, no registration required |
| Sites | NYU, USM (University of Southern Mississippi), UCLA_1 |
| Preprocessing | C-PAC pipeline, band-pass filtered, no global signal regression |
| Parcellation | CC200 (Craddock et al., 2012) [2], 200 functionally defined ROIs |
| Functional networks | 8: DMN, Visual, SMN, DAN, VAN, FPN, Limbic, Subcortical |
| Subjects | 303 total: 154 ASD, 149 neurotypical controls |

> **Download:** `nilearn` fetches data automatically on first run (~500 MB, locally cached thereafter).

**What is a parcellation?** Rather than analyzing millions of individual brain voxels (3D pixels), we divide the brain into larger regions called parcels and treat each parcel as a single unit. The **CC200 parcellation** [2] divides the brain into 200 functionally coherent regions using a statistical clustering algorithm applied to resting-state fMRI data from healthy adults. This gives 200 ROIs (regions of interest) as the nodes of our graph.

**What are the 8 functional networks?** Large-scale brain networks are groups of spatially distributed regions that consistently co-activate across tasks and rest. The 8 used here are: Default Mode Network (DMN, active during self-reflection and social thought), Visual, Sensorimotor (SMN), Dorsal Attention (DAN), Ventral Attention (VAN), Frontoparietal (FPN, involved in executive control and working memory), Limbic (emotion and memory), and Subcortical (basal ganglia, thalamus, and related structures).

---

## Pipeline Overview

Five Python scripts run in sequence. Every output is fully reproducible from the automatic ABIDE download.

```
01_fetch_and_prepare.py --> 02_harmonize.py --> 03_build_graphs.py --> 04_train_evaluate.py --> 05_figures.py
  Download ABIDE fMRI       ComBat site          Threshold matrices,     Extract 44 graph-        6 publication-
  time series, compute      harmonization        compute 5 node          level features,          quality figures
  connectomes, apply        removes scanner-     features per ROI,       train classifier,
  Fisher z-transform        site variance        build graph objects     5-fold CV
```

### What each step does

**Step 1: Data retrieval and connectome construction**
Downloads pre-processed ROI time series for each subject using `nilearn`. For each subject, computes the 200x200 Pearson correlation matrix between all pairs of ROI time series. The **Fisher z-transform** is then applied: r -> arctanh(r), converting bounded correlation values into approximately normally distributed Fisher z scores. Zero-variance ROIs produce undefined (NaN) correlations and are replaced with 0.

**Step 2: ComBat site harmonization**
Applies ComBat [11, 12] to remove systematic scanner-site variance before any graph construction or classification. ComBat treats the upper triangle of each connectivity matrix (19,900 values) as a feature vector and fits a linear model with additive and multiplicative site effects, estimated using an empirical Bayes framework that borrows strength across features. The site effects are then subtracted from each subject's data, leaving the biological signal intact.

**What is ComBat?** ComBat (Johnson et al., 2007) was originally developed to correct for batch effects in genomics microarray data, where samples processed in different batches show systematic expression differences unrelated to biology. The same problem occurs in multi-site neuroimaging: scanners at different hospitals produce connectivity values that differ systematically due to hardware and protocol differences. ComBat estimates and removes these differences while preserving within-site biological variation.

**Step 3: Graph construction and node feature extraction**
Applies a threshold of |z| > 0.20 to the harmonized connectivity matrices, retaining only the strongest connections. Five features are then computed per ROI from the resulting sparse graph; see the Node Features section below.

**Step 4: Feature extraction and classification**
Extracts the upper triangle of each harmonized connectivity matrix (19,900 values per subject), applies PCA to reduce dimensionality to 50 components within each fold, and trains a **gradient-boosted classifier** under **5-fold cross-validation**. PCA is fit on the training set only within each fold to prevent information leakage.

**What is a gradient-boosted classifier?** Gradient boosting builds many simple decision trees sequentially, where each tree corrects the mistakes of the previous one. The result is a strong, nonlinear classifier that performs well on tabular feature data without requiring large datasets or a GPU.

**What is 5-fold cross-validation?** The 303 subjects are split into 5 equal groups. The model is trained on 4 groups and tested on the 1 held-out group, repeated 5 times so every subject appears in the test set exactly once. Reported metrics are averaged over all 5 test sets.

**Step 5: Figures**
Generates six figures covering harmonized connectivity matrices, graph topology, feature distributions, classification performance, ROC curves, and node importance.

---

## Node Features

Five features are computed per ROI from the thresholded, harmonized graph:

| Feature | Description | Biological meaning |
|---|---|---|
| Mean FC | Average Fisher z to all other ROIs | Overall connectivity strength; how embedded the region is in the network |
| Degree | Number of edges above threshold | Hubness; how many significant connections the region maintains |
| Clustering coefficient | Fraction of a node's neighbors also connected to each other | Local cliquishness; whether nearby regions form tightly knit clusters |
| Positive FC | Mean of positive correlations only | Co-activation profile; regions that tend to be active together |
| Negative FC | Mean of anti-correlations only | Competing connectivity; regions that suppress each other's activity |

**Clustering coefficient explained:** If region A connects to regions B, C, and D, the clustering coefficient asks: are B, C, and D also connected to each other? A high clustering coefficient means A sits in a tightly interconnected local cluster. A low one means A's neighbors do not talk to each other, making A a bridge between otherwise separate communities.

---

## Graph-Level Feature Extraction

Rather than using all 19,900 pairwise edge values as features (far too many for 303 subjects), the pipeline extracts 44 biologically interpretable features per subject:

- Between-network mean FC difference for each pair of 8 networks (28 values, one per unique network pair)
- Mean node degree per network (8 values, representing how connected each network is on average)
- Mean clustering coefficient per network (8 values, representing how locally clustered each network is)

This gives one 44-dimensional vector per subject that the classifier can actually learn from.

---

## Results

5-fold stratified cross-validation (Gradient Boosting on ComBat-harmonized connectome features):

| Metric | Unharmonized baseline | After ComBat + PCA |
|---|---|---|
| Accuracy | 0.515 | 0.680 |
| AUC-ROC | 0.514 | 0.720 |
| Sensitivity (ASD recall) | 0.566 | 0.747 |
| Specificity (CTRL recall) | 0.462 | 0.610 |

ComBat site harmonization followed by PCA (50 components, explaining ~62% of connectivity variance) and gradient boosting achieves AUC = 0.720 +/- 0.037 -- a substantial improvement over the unharmonized baseline and consistent with published results on harmonized ABIDE data [5, 6].

### Why the unharmonized baseline is at chance

Without site harmonization, ABIDE classification is extremely difficult for three reasons:

**Site effects dominate.** The three acquisition sites (NYU, USM, UCLA) use different MRI scanners, different acquisition parameters, and different operator practices. These differences introduce systematic variance far larger than the biological ASD signal.

**Small within-site samples.** After 5-fold splitting, each test set contains approximately 60 subjects across three sites, which is far too few to detect a subtle, heterogeneous biological effect.

**ASD heterogeneity.** ASD is a spectrum with highly variable connectivity profiles across individuals. Group-level differences are real but small relative to within-group variance.

**What ComBat fixes:** ComBat [11] estimates and removes the additive and multiplicative scanner effects for each site, leaving the biological connectivity differences between ASD and control subjects intact. This is the standard preprocessing step in multi-site neuroimaging and reliably recovers meaningful classification performance [5, 6].

---

### Figure 1: Resting-State Functional Connectivity Matrices

[![Functional connectivity matrices](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig1_connectivity_matrices.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig1_connectivity_matrices.png)

Fisher z-transformed, ComBat-harmonized correlation matrices for a representative neurotypical control (left) and ASD subject (right). Each pixel at position (i, j) shows the strength of functional connectivity between brain region i and brain region j: warm red colors indicate positive correlation, cool blue colors indicate anti-correlation, and white/neutral colors indicate near-zero correlation. The white grid lines divide the 200 ROIs into 8 functional networks. Post-harmonization, between-site scanner artifacts are removed and patterns reflect underlying biology.

---

### Figure 2: Brain Graph Structure

[![Brain graph structure](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig2_graph_structure.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig2_graph_structure.png)

All 200 CC200 ROIs arranged as nodes in a circle, each colored by functional network (see legend). The top 300 edges by absolute connectivity strength are drawn as lines: red = positive functional connectivity, blue = anti-correlation. Post-harmonization, differences in edge count between ASD and control subjects reflect biological connectivity differences rather than scanner-site artifacts.

---

### Figure 3: Node Feature Distributions by Group

[![Node feature distributions](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig3_feature_distributions.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig3_feature_distributions.png)

Violin plots comparing the distribution of all 5 node features between neurotypical controls (blue) and ASD subjects (orange). Each observation is one ROI in one subject (149 x 200 = 29,800 control observations; 154 x 200 = 30,800 ASD observations). The white bar marks the median. Post-harmonization, any visible separation between groups reflects biological differences rather than scanner effects.

---

### Figure 4: Per-Fold Classification Performance

[![Per-fold performance](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig4_performance.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig4_performance.png)

AUC-ROC (blue bars) and accuracy (orange bars) for each of the 5 cross-validation folds after ComBat harmonization. The dashed grey line marks chance level (0.50). Each fold held out approximately 60 subjects. Consistent performance across folds indicates that site harmonization has reduced the fold-to-fold variance that characterized the unharmonized baseline.

---

### Figure 5: ROC Curves

[![ROC curves](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig5_roc_curves.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig5_roc_curves.png)

Receiver Operating Characteristic curves for each of the 5 cross-validation folds plus the interpolated mean with +/- 1 SD band. The x-axis is the false positive rate (fraction of controls incorrectly classified as ASD); the y-axis is the true positive rate (fraction of ASD subjects correctly identified). A perfect classifier reaches the top-left corner. The mean AUC summarizes overall classification performance after harmonization.

---

### Figure 6: Node Importance by Network and Feature

[![Node importance heatmap](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig6_node_importance.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig6_node_importance.png)

Mean absolute difference between ASD and control node features, aggregated by functional network and averaged across 5 held-out folds. Values are normalized to 0-1 per feature column. Darker red = larger ASD-Control difference; pale yellow = near-zero difference. Post-harmonization, this heatmap reflects biological network differences rather than site-confounded artifacts.

**Sensorimotor Network (SMN)** shows the largest differences in mean FC and positive FC, consistent with well-documented sensory processing atypicalities in ASD and altered sensorimotor integration [8].

**Frontoparietal Network (FPN)** shows high importance across clustering coefficient, positive FC, and degree, consistent with disrupted executive function and working memory networks in ASD [9].

**Limbic system** shows consistently high importance across most features, reflecting the role of limbic connectivity in social cognition and emotional processing differences in ASD.

**Visual cortex** shows near-zero importance across all features, consistent with relatively preserved visual processing at the network level in ASD [1].

---

## Summary of Findings

| Question | Finding |
|---|---|
| Does unharmonized multi-site classification work? | No (AUC = 0.51). Scanner variance dominates [11]. |
| Does ComBat harmonization recover signal? | Yes. AUC improves from 0.51 to 0.72 after removing site effects and using PCA features [5, 6, 11]. |
| Which networks show the largest ASD-Control differences? | SMN, FPN, and Limbic show the strongest node importance signals [8, 9]. |
| Are node-level features sufficient alone? | No. Individual ROI features overlap substantially; network-level aggregation is necessary. |

---

## Setup and Usage

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/anirudhramadurai/connectome_gnn.git
cd connectome_gnn

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline with one command
chmod +x run_all.sh
./run_all.sh
```

Or run each step individually:

```bash
python scripts/01_fetch_and_prepare.py   # downloads ABIDE (~500 MB, cached after first run)
python scripts/02_harmonize.py           # ComBat site harmonization
python scripts/03_build_graphs.py
python scripts/04_train_evaluate.py
python scripts/05_figures.py
```

Figures are written to `figures/`. Data files are written to `data/`. Results are written to `results/`.

---

## Requirements

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
pandas>=2.0
nilearn>=0.10
neuroCombat>=0.2.12
```

---

## A note on `models/gcn_numpy.py`

This file contains a from-scratch NumPy implementation of the graph convolution mathematics described in Kipf & Welling (2017). It is **not** the production classifier.

It exists as an educational resource to make the GCN forward pass (symmetric normalized adjacency, two-layer propagation, global mean pooling) transparent and inspectable without requiring PyTorch. The backward pass is numerically unstable on real fMRI data at this graph density and is not used in the pipeline. The actual classifier is gradient boosting on hand-engineered graph features in `04_train_evaluate.py`.

For a production GCN on ABIDE, see BrainGNN [6] and PyTorch Geometric.

---

## Limitations and Future Directions

**Current limitations:**

- Sample size (n = 303 across 3 sites) limits within-site statistical power for a subtle, heterogeneous condition
- The 44-feature graph-level representation discards the full edge structure; a GCN operating on the complete graph may recover additional signal [6, 7]
- Network assignments in the CC200 parcellation are approximate; ROI-to-network mappings are based on Power et al. (2011) applied to CC200 ordering
- Results are correlational; no causal inference is possible from cross-sectional observational data
- ABIDE is cross-sectional (one scan per participant); longitudinal data would allow questions about developmental trajectories of connectivity in ASD

**Future directions:**

- Scale to the full ABIDE cohort (~1,100 subjects across 17 sites) with leave-site-out cross-validation for a more honest estimate of generalizability
- Implement a PyTorch Geometric GCN [6, 7] with proper site stratification and autograd, which achieves AUC ~0.72-0.78 on ABIDE
- Test higher-resolution parcellations (Schaefer-400, Gordon-333) to capture finer-grained connectivity patterns
- Incorporate demographic covariates (age, sex, IQ) to account for biological confounds within sites
- Apply graph attention networks (GAT) to learn which edges are most informative rather than using a fixed threshold

---

## References

1. Di Martino A, Yan C-G, Li Q, et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*. 2014;19(6):659-667. doi:10.1038/mp.2013.78. PMID: 24514918.

2. Craddock RC, James GA, Holtzheimer PE, Hu XP, Mayberg HS. A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*. 2012;33(8):1914-1928. doi:10.1002/hbm.21333.

3. Just MA, Cherkassky VL, Keller TA, Minshew NJ. Cortical activation and synchronization during sentence comprehension in high-functioning autism: evidence of underconnectivity. *Brain*. 2004;127(8):1811-1821. doi:10.1093/brain/awh199.

4. Assaf M, Jagannathan K, Calhoun VD, et al. Abnormal functional connectivity of default mode sub-networks in autism spectrum disorder patients. *NeuroImage*. 2010;53(1):247-256. doi:10.1016/j.neuroimage.2010.05.067.

5. Ktena SI, Parisot S, Ferrante E, et al. Metric learning with spectral graph convolutions on brain connectivity networks. *NeuroImage*. 2018;169:431-442. doi:10.1016/j.neuroimage.2017.12.052.

6. Li X, Zhou Y, Dvornek N, et al. BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis*. 2021;74:102233. doi:10.1016/j.media.2021.102233.

7. Jiang H, Cao P, Xu M, Yang J, Zaiane O. Hi-GCN: A hierarchical graph convolution network for graph embedding learning of brain network. *Computers in Biology and Medicine*. 2020;127:104096. doi:10.1016/j.compbiomed.2020.104096.

8. Marco EJ, Hinkley LBN, Hill SS, Nagarajan SS. Sensory processing in autism: a review of neurophysiologic findings. *Pediatric Research*. 2011;69(5 Pt 2):48R-54R. doi:10.1203/PDR.0b013e3182130c54.

9. Yerys BE, Gordon EM, Abrams DN, et al. Default mode network segregation and social deficits in autism spectrum disorder. *NeuroImage: Clinical*. 2015;9:223-232. doi:10.1016/j.nicl.2015.07.018.

10. Bullmore E, Sporns O. Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*. 2009;10(3):186-198. doi:10.1038/nrn2575.

11. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*. 2007;8(1):118-127. doi:10.1093/biostatistics/kxj037.

12. Fortin JP, Parker D, Tunc B, et al. Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*. 2017;161:149-170. doi:10.1016/j.neuroimage.2017.08.047.

13. Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. *ICLR 2017*. arXiv:1609.02907.

---

## Acknowledgements

Developed as an independent computational neuroscience project. Data from the ABIDE Preprocessed Connectomes Project, accessed via nilearn. ABIDE was supported by grants from the Autism Speaks Foundation and the National Institute of Mental Health.