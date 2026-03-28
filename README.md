# connectome-gnn

**Graph-based classification of Autism Spectrum Disorder from resting-state fMRI functional connectivity data**

---

## Overview

Using publicly available resting-state fMRI data from 303 participants across three ABIDE sites (NYU, USM, UCLA), I built a four-script Python pipeline to ask whether ASD can be detected from whole-brain functional connectivity patterns represented as graphs. After computing 200x200 correlation matrices per subject, converting them to graphs, and extracting 44 biologically grounded graph-level features, a gradient-boosted classifier achieves AUC = 0.51 +/- 0.04 in 5-fold cross-validation, which is essentially chance. This near-chance result is honest and expected on raw multi-site ABIDE data without site harmonization: systematic differences between MRI scanners at different hospitals dominate the signal and mask the biological ASD effect.

The more informative result comes from the node importance analysis (Fig 6): sensorimotor (SMN), frontoparietal (FPN), and limbic network features show the largest ASD-Control differences, consistent with published findings on sensorimotor integration, executive function, and social cognition differences in ASD [8, 9]. The main limitation is the absence of ComBat site harmonization (a statistical correction for scanner differences), which is the standard next step to recover meaningful biological classification performance [11].

---

## What is this project?

This project applies a reproducible Python pipeline to publicly available neuroimaging data to investigate a specific question about ASD: can the condition be detected from the pattern of functional connections between brain regions, when those connections are modelled as a graph?

The full pipeline runs in approximately 10 minutes on a standard laptop, from automatic data download to six figures.

---

## The Biology (plain language first)

### What is ASD and why study brain connectivity?

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition affecting approximately 1-2% of the global population [1], characterized by differences in social communication and restricted or repetitive behavior patterns. It is a spectrum, meaning the condition presents very differently across individuals, ranging from people who are largely independent to those who require substantial daily support.

Unlike conditions such as Alzheimer's disease, where specific brain regions are visibly lost, ASD does not produce obvious structural damage detectable by standard brain imaging. This has led researchers to focus instead on how brain regions **communicate with each other**: the hypothesis is that ASD arises not from failure of any single region, but from disrupted coordination across the brain's large-scale networks [1].

### What is resting-state fMRI and functional connectivity?

**Functional MRI (fMRI)** measures brain activity indirectly. When neurons in a region fire, they demand more oxygen-rich blood. The MRI scanner can detect the resulting change in the ratio of oxygenated to deoxygenated haemoglobin; this is called the **BOLD signal** (Blood Oxygen Level Dependent). The more active a region, the stronger its BOLD signal.

In **resting-state fMRI**, there is no task. Participants simply lie still in the scanner with their eyes open or closed. Even at rest, the brain is highly active: different regions spontaneously fluctuate in activity, and importantly, certain regions tend to fluctuate *together*, activating and deactivating in synchrony.

**Functional connectivity (FC)** is the statistical correlation between the BOLD time series of two brain regions across a scan. If region A and region B consistently go up and down together over the course of several minutes, their functional connectivity is high (correlation close to +1). If they consistently do the opposite, they are anti-correlated (close to -1). A near-zero correlation means the two regions operate independently.

By computing this correlation for every pair of 200 brain regions, we produce a 200x200 **functional connectome**, a matrix capturing the brain's entire large-scale coordination structure for that individual subject.

### What does the ASD connectivity literature say?

The most replicated finding in ASD neuroimaging is **long-range underconnectivity**: regions far apart in the brain, particularly within the Default Mode Network (the network active during self-reflection and social cognition) and between association cortices (regions that integrate information from multiple sources), show weaker-than-typical synchronisation in ASD [3, 4]. This is thought to reflect disrupted information integration across distributed brain networks.

There is also evidence of **local overconnectivity** in sensorimotor regions, meaning elevated synchrony between nearby regions, which may relate to sensory hypersensitivity and motor coordination differences commonly observed in ASD [8].

These connectivity differences are real but subtle at the group level, and highly variable across individuals. This is a central reason why ASD classification from fMRI remains a hard problem.

### Why model the connectome as a graph?

A 200x200 correlation matrix contains 19,900 unique pairwise connections. Using all of them as features for a 303-subject dataset would give a model roughly 66 times more features than samples, a situation called high-dimensional underpowering, where a classifier will overfit to noise and fail completely on new data.

A **graph** offers a more structured alternative. In graph theory, a graph is simply a collection of **nodes** (entities) connected by **edges** (relationships). Here, the 200 brain regions are nodes, and a connection is drawn between two regions if their correlation exceeds a threshold. Each node then has a compact set of local properties (how many connections it has, how tightly clustered those connections are) that summarise its role in the network without using all 19,900 raw edge values. This reduces the feature space from 19,900 to 44 biologically interpretable values per subject.

---

## Research Questions

**Primary:** Can ASD be classified from resting-state fMRI functional connectivity patterns using graph-based features, and which brain networks and connectivity properties drive any observed differences?

**Secondary:** Does the near-chance classification performance on raw multi-site data reflect site-driven variance rather than biological signal, and does node importance analysis recover network differences consistent with the published ASD connectivity literature?

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

**What is a parcellation?** Rather than analyzing millions of individual brain voxels (3D pixels), we divide the brain into larger regions called parcels and treat each parcel as a single unit. The **CC200 parcellation** [2] divides the brain into 200 functionally coherent regions using a statistical clustering algorithm applied to resting-state fMRI data from healthy adults. Regions within a parcel are more similar to each other in their temporal activity patterns than to regions in other parcels. This gives 200 ROIs (regions of interest) as the nodes of our graph.

**What are the 8 functional networks?** Large-scale brain networks are groups of spatially distributed regions that consistently co-activate across tasks and rest. The 8 used here are: Default Mode Network (DMN, active during self-reflection and social thought), Visual, Sensorimotor (SMN), Dorsal Attention (DAN), Ventral Attention (VAN), Frontoparietal (FPN, involved in executive control and working memory), Limbic (emotion and memory), and Subcortical (basal ganglia, thalamus, and related structures).

---

## Pipeline Overview

Four Python scripts run in sequence. Every output is fully reproducible from the automatic ABIDE download.

```
01_fetch_and_prepare.py --> 02_build_graphs.py --> 03_train_evaluate.py --> 04_figures.py
  Download ABIDE fMRI       Threshold matrices,     Extract 44 graph-        6 publication-
  time series, compute      compute 5 node          level features,          quality figures
  connectomes, apply        features per ROI,       train classifier,
  Fisher z-transform        build graph objects     5-fold CV
```

### What each step does

**Step 1: Data retrieval and connectome construction**
Downloads pre-processed ROI time series for each subject using `nilearn`. For each subject, computes the 200x200 Pearson correlation matrix between all pairs of ROI time series; the correlation coefficient r ranges from -1 to +1. The **Fisher z-transform** is then applied: r -> arctanh(r). This converts the bounded correlation values into unbounded, approximately normally distributed Fisher z scores, which are more appropriate for statistical analysis and machine learning. Zero-variance ROIs (brain regions that produced no signal in some subjects, typically due to scanner coverage gaps) produce undefined (NaN) correlations and are replaced with 0.

**Step 2: Graph construction and node feature extraction**
Applies a threshold of |z| > 0.20 to each Fisher z matrix: connections weaker than this are discarded, retaining only the most reliable edges. This threshold is standard in fMRI graph analysis [10]. Five features are then computed per ROI from the resulting sparse graph; see the Node Features section below.

**Step 3: Feature extraction and classification**
Extracts 44 graph-level summary features per subject by aggregating node features across the 8 functional networks. A **gradient-boosted classifier** is trained under **5-fold cross-validation**, explained in detail below. Missing values from the NaN imputation step are filled using median imputation before classification.

**What is a gradient-boosted classifier?** Gradient boosting is an ensemble machine learning method that builds many simple decision trees sequentially, where each tree corrects the mistakes of the previous one. The result is a strong, nonlinear classifier that performs well on tabular feature data. It is a standard, well-validated approach and does not require the large datasets that deep learning methods need.

**What is 5-fold cross-validation?** To get an honest estimate of how well a model generalises to new subjects, we split the 303 subjects into 5 equal groups (folds). The model is trained on 4 folds and tested on the 1 held-out fold, then this is repeated 5 times so every subject appears in the test set exactly once. The final performance metrics are averaged across all 5 test folds. This prevents the model from being evaluated on subjects it trained on, which would give artificially inflated results.

**Step 4: Figures**
Generates six figures covering raw connectivity matrices, graph topology, feature distributions, classification performance, ROC curves, and node importance.

---

## Node Features

Five features are computed per ROI from the thresholded graph:

| Feature | Description | Biological meaning |
|---|---|---|
| Mean FC | Average Fisher z to all other ROIs | Overall connectivity strength; how embedded the region is in the network |
| Degree | Number of edges above threshold | Hubness; how many significant connections the region maintains |
| Clustering coefficient | Fraction of a node's neighbors that are also connected to each other | Local cliquishness; whether nearby regions form tightly knit clusters |
| Positive FC | Mean of positive correlations only | Co-activation profile; regions that tend to be active together |
| Negative FC | Mean of anti-correlations only | Competing connectivity; regions that suppress each other's activity |

**Clustering coefficient explained:** If region A connects to regions B, C, and D, the clustering coefficient asks: are B, C, and D also connected to each other? A high clustering coefficient means A sits in a tightly interconnected local cluster. A low one means A's neighbors do not talk to each other, making A a bridge between otherwise separate communities.

---

## Graph-Level Feature Extraction

Rather than using all 19,900 pairwise edge values as features (far too many for 303 subjects), the pipeline extracts 44 biologically interpretable features per subject by aggregating node features across networks:

- Between-network mean FC difference for each pair of 8 networks (28 values, one per unique network pair)
- Mean node degree per network (8 values, representing how connected each network is on average)
- Mean clustering coefficient per network (8 values, representing how locally clustered each network is)

This gives one 44-dimensional vector per subject that the classifier can actually learn from.

---

## Results

5-fold stratified cross-validation (Gradient Boosting with median imputation and standard scaling):

| Metric | Mean | SD |
|---|---|---|
| Accuracy | 0.515 | 0.050 |
| AUC-ROC | 0.514 | 0.042 |
| Sensitivity (ASD recall) | 0.566 | 0.052 |
| Specificity (CTRL recall) | 0.462 | 0.087 |

**What do these metrics mean?**

- **Accuracy** is the fraction of subjects correctly classified as ASD or control. 0.515 means the model is right 51.5% of the time, barely above the 50.7% you would get by always guessing the majority class.
- **AUC-ROC** (Area Under the Receiver Operating Characteristic curve) measures how well the model ranks ASD subjects above controls, independent of any particular decision threshold. AUC = 0.5 means the model is no better than random; AUC = 1.0 means perfect separation. Our AUC of 0.51 is at chance.
- **Sensitivity** is the fraction of ASD subjects correctly identified (true positive rate). 0.566 means we catch 56.6% of ASD cases.
- **Specificity** is the fraction of controls correctly identified (true negative rate). 0.462 means we correctly rule out ASD in 46.2% of controls.

### Why AUC ~0.51?

This result is honest and expected. ABIDE classification on raw unharmonized data is extremely difficult for three reasons:

**Site effects dominate.** The three acquisition sites (NYU, USM, UCLA) use different MRI scanners, different acquisition parameters, and different operator practices. These technical differences introduce systematic variance in the connectivity data that is far larger than the biological ASD signal. A classifier trained on mixed-site data learns to recognize scanner fingerprints more than disease biology. This is analogous to trying to detect a small gene expression difference while using three different microarray platforms without correcting for platform effects.

**Small within-site samples.** After 5-fold splitting, each test set contains approximately 60 subjects spread across three sites, which is far too few to detect a subtle, heterogeneous biological effect.

**ASD heterogeneity.** ASD is a spectrum with highly variable connectivity profiles across individuals. Group-level differences are real but small relative to within-group variance.

**What would fix this?** Applying **ComBat** [11], a statistical method originally developed for genomics batch correction, to remove scanner-site variance before classification. This is standard in multi-site neuroimaging and reliably improves AUC to 0.65-0.78 on ABIDE [5, 6]. This pipeline establishes a transparent, reproducible baseline.

---

### Figure 1: Resting-State Functional Connectivity Matrices

[![Functional connectivity matrices](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig1_connectivity_matrices.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig1_connectivity_matrices.png)

Fisher z-transformed correlation matrices for a representative neurotypical control (left) and ASD subject (right). Each pixel at position (i, j) shows the strength of functional connectivity between brain region i and brain region j: warm red colors indicate positive correlation (regions that activate together), cool blue colors indicate anti-correlation (regions that suppress each other), and white/neutral colors indicate near-zero correlation. The white grid lines divide the 200 ROIs into 8 functional networks.

The ASD subject's matrix appears more uniformly warm with a visible white diagonal grid pattern. This reflects zero-variance ROIs: brain regions where the UCLA scanner produced no signal for that subject, causing their rows and columns to be set to zero. This is a known artifact in some ABIDE preprocessed files and is handled by NaN imputation in the pipeline.

---

### Figure 2: Brain Graph Structure

[![Brain graph structure](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig2_graph_structure.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig2_graph_structure.png)

All 200 CC200 ROIs are arranged as nodes in a circle, each colored by its functional network (see legend). The top 300 edges by absolute connectivity strength are drawn as lines: red lines represent positive functional connectivity (co-activating regions), blue lines represent anti-correlations (competing regions).

The ASD subject (right, 35,718 edges) has more edges above the |z| > 0.20 threshold than the control (left, 16,620 edges). In unharmonized ABIDE data, this likely reflects scanner-site differences rather than a true biological overconnectivity effect, as the UCLA acquisition protocol can produce inflated correlation values. In properly harmonized multi-site data, ASD typically shows *fewer* long-range connections (the underconnectivity hypothesis [3]).

---

### Figure 3: Node Feature Distributions by Group

[![Node feature distributions](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig3_feature_distributions.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig3_feature_distributions.png)

Violin plots comparing the distribution of all 5 node features between neurotypical controls (blue) and ASD subjects (orange). Each observation is one ROI in one subject, so there are 149 x 200 = 29,800 observations in the control group and 154 x 200 = 30,800 in the ASD group. The white bar inside each violin marks the median; the width of the violin at any value shows how many observations have that value.

All five features show substantial overlap between the two groups. This is the key insight behind the near-chance classification result: when measured at the individual node level, ASD and control brains are nearly indistinguishable. The differences emerge only at the network level, in how regions coordinate across the whole brain, which is why the graph-level aggregated features are necessary.

---

### Figure 4: Per-Fold Classification Performance

[![Per-fold performance](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig4_training_loss.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig4_training_loss.png)

AUC-ROC (blue bars) and accuracy (orange bars) for each of the 5 cross-validation folds. The dashed grey line marks chance level (0.50). Each fold used approximately 60 subjects as a held-out test set, with the remaining ~243 subjects used for training.

Fold 4 achieves the best performance (AUC = 0.56, Accuracy = 0.58); Fold 5 falls below chance (AUC = 0.44, Accuracy = 0.45). This variance across folds (a range of 0.12 AUC units) is a direct signature of site effects. When Fold 5's test set happened to contain a site distribution that the training data did not represent well, performance collapsed. This is exactly the kind of generalisation failure that site harmonization is designed to prevent.

---

### Figure 5: ROC Curves

[![ROC curves](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig5_roc_curves.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig5_roc_curves.png)

The Receiver Operating Characteristic (ROC) curve is a standard tool for evaluating binary classifiers. For each possible decision threshold (how confident the model needs to be before predicting ASD), we compute two quantities: the **true positive rate** (y-axis, the fraction of ASD subjects correctly identified) and the **false positive rate** (x-axis, the fraction of controls incorrectly labeled as ASD). As the threshold decreases, both rates increase. A perfect classifier reaches the top-left corner (100% sensitivity with 0% false positives). A classifier no better than chance traces the diagonal dashed line from bottom-left to top-right.

Each colored line shows one fold's ROC curve. The thick black line is the mean, with the grey band showing +/- 1 standard deviation across folds. A mean AUC of 0.51 +/- 0.04 means the classifier performs no better than randomly guessing, confirmed by the mean curve closely following the diagonal.

---

### Figure 6: Node Importance by Network and Feature

[![Node importance heatmap](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig6_node_importance.png)](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig6_node_importance.png)

This heatmap shows which brain networks and node features differ most between ASD and control subjects. Each cell shows the mean absolute difference between ASD and control group averages for a given network (column) and feature (row), computed on held-out subjects across all 5 folds and then normalized to a 0-1 scale per feature column. Darker red = larger ASD-Control difference; pale yellow = near-zero difference.

Several patterns are biologically interpretable:

**Sensorimotor Network (SMN)** shows the largest differences in mean FC (0.89) and positive FC (0.88). This is consistent with well-documented sensory processing atypicalities in ASD and altered sensorimotor integration [8].

**Frontoparietal Network (FPN)** shows high importance across clustering coefficient, positive FC (0.86), and degree (0.57). The FPN is a key network for executive function (working memory, flexible reasoning, and cognitive control), which are often affected in ASD [9].

**Limbic system** shows the highest importance for mean FC (1.00), degree (1.00), and positive FC (1.00). The limbic system, which encompasses the amygdala, hippocampus, and cingulate cortex, is deeply involved in emotional processing and social cognition, two domains that are characteristically different in ASD.

**Visual cortex** shows near-zero importance across all five features. This is consistent with the published literature: visual processing is largely preserved in ASD at the network level, and visual cortex connectivity does not reliably distinguish ASD from controls [1].

---

## Summary of Findings

| Question | Finding |
|---|---|
| Can ASD be classified from raw multi-site connectome features? | No (AUC = 0.51). Site-driven scanner variance dominates the signal [11]. |
| Which networks show the largest ASD-Control differences? | SMN, FPN, and Limbic show the strongest node importance signals. |
| Are node-level features sufficient for classification? | No. Individual ROI features overlap substantially; network-level aggregation is necessary. |
| What is the next step to improve performance? | ComBat site harmonization [11], which recovers AUC 0.65-0.78 in published models [5, 6]. |

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
python scripts/02_build_graphs.py
python scripts/03_train_evaluate.py
python scripts/04_figures.py
```

Figures are written to `figures/`. Data files are written to `data/`. Results are written to `results/`.

---

## Limitations and Future Directions

**Current limitations:**

- Site effects between NYU, USM, and UCLA dominate the connectivity signal and mask the biological ASD effect without explicit harmonization [11]
- Sample size (n = 303 across 3 sites) limits within-site statistical power for a subtle, heterogeneous condition
- The 44-feature graph-level representation discards the full edge structure; a GCN operating on the complete graph may recover more signal [6, 7]
- Network boundaries in the CC200 parcellation are approximate; ROI-to-network assignments are based on Power et al. (2011) mapped onto CC200 ordering
- Results are correlational; no causal inference is possible from cross-sectional observational data
- ABIDE is cross-sectional (one scan per participant); longitudinal data would allow questions about developmental trajectories of connectivity in ASD

**Future directions:**

- Apply ComBat [11] or neuroCombat site harmonization before classification to remove scanner-driven variance
- Scale to the full ABIDE cohort (~1,100 subjects across 17 sites) with leave-site-out cross-validation for a more honest estimate of generalisability
- Implement a PyTorch Geometric GCN [6, 7] with proper site stratification, which achieves AUC ~0.72-0.78 on ABIDE
- Test higher-resolution parcellations (Schaefer-400, Gordon-333) to capture finer-grained connectivity patterns
- Incorporate demographic covariates (age, sex, IQ) as features to account for biological confounds within sites

---

## References

1. Di Martino A, Yan C-G, Li Q, et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*. 2014;19(6):659-667. doi:10.1038/mp.2013.78. PMID: 24514918.

2. Craddock RC, James GA, Holtzheimer PE, Hu XP, Mayberg HS. A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*. 2012;33(8):1914-1928. doi:10.1002/hbm.21333.

3. Just MA, Cherkassky VL, Keller TA, Minshew NJ. Cortical activation and synchronization during sentence comprehension in high-functioning autism: evidence of underconnectivity. *Brain*. 2004;127(8):1811-1821. doi:10.1093/brain/awh199.

4. Assaf M, Jagannathan K, Calhoun VD, et al. Abnormal functional connectivity of default mode sub-networks in autism spectrum disorder patients. *NeuroImage*. 2010;53(1):247-256. doi:10.1016/j.neuroimage.2010.05.067.

5. Ktena SI, Parisot S, Ferrante E, et al. Metric learning with spectral graph convolutions on brain connectivity networks. *NeuroImage*. 2018;169:431-442. doi:10.1016/j.neuroimage.2017.12.052.

6. Li X, Zhou Y, Dvornek N, et al. BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis*. 2021;74:102233. doi:10.1016/j.media.2021.102233.

7. Jiang H, Cao P, Xu M, Yang J, Zaiane O. Hi-GCN: A hierarchical graph convolution network for graph embedding learning of brain network and its application to the detection of Alzheimer's disease. *Computers in Biology and Medicine*. 2020;127:104096. doi:10.1016/j.compbiomed.2020.104096.

8. Marco EJ, Hinkley LBN, Hill SS, Nagarajan SS. Sensory processing in autism: a review of neurophysiologic findings. *Pediatric Research*. 2011;69(5 Pt 2):48R-54R. doi:10.1203/PDR.0b013e3182130c54.

9. Yerys BE, Gordon EM, Abrams DN, et al. Default mode network segregation and social deficits in autism spectrum disorder: evidence from non-medicated children. *NeuroImage: Clinical*. 2015;9:223-232. doi:10.1016/j.nicl.2015.07.018.

10. Bullmore E, Sporns O. Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*. 2009;10(3):186-198. doi:10.1038/nrn2575.

11. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*. 2007;8(1):118-127. doi:10.1093/biostatistics/kxj037.

---

## Acknowledgements

Developed as an independent computational neuroscience project. Data from the ABIDE Preprocessed Connectomes Project, accessed via nilearn. ABIDE was supported by grants from the Autism Speaks Foundation and the National Institute of Mental Health.