# Beyond the LC25000 benchmark

Code, data partitions, per-image predictions and summary statistics for the manuscript:

> S. Rahuman, *Beyond the LC25000 benchmark: source-group evaluation, external testing, and a capacity-controlled analysis of lightweight CNN–Transformer models for lung histopathology* (submitted to *Computers in Biology and Medicine*).

## Study design

| | Protocol A (image-level split) | Protocol B (source-group split) |
|---|---|---|
| Partitioning | class-stratified 70/15/15 | whole LC25000-clean source groups per partition, 70/15/15 |
| Partitions | seeds 42–46 | seeds 42–46 |
| Fine-tuned networks | MobileNetV2, EfficientNetB0, hybrid EfficientNetB0–Transformer, approximately parameter-matched TokenMLP (20) | the same four (20) |
| Frozen-feature baselines + logistic regression | Phikon, ImageNet ViT-B/16, ImageNet EfficientNetB0 (15) | the same three (15) |

All 40 fine-tuned networks and 30 frozen-feature classifiers were evaluated without retraining on LungHist700 (691 images, 45 patients). An initial Protocol A run on the seed-42 partition supplied the training curves, efficiency measurements, saliency analysis and external sensitivity analyses.

## Repository structure

```
notebooks/
  initial_run/1_Unified_Benchmark_LC25000.ipynb        initial seed-42 run: training and efficiency (Table 2, training curves)
  initial_run/2_External_Validation_LungHist700.ipynb  initial run: external sensitivity analyses (Table 5)
  initial_run/3_XAI_Corrected_Analysis.ipynb           saliency evaluation (Table 8, saliency figure)
  4_Protocol_B_Source_Group_5Seeds.ipynb               Protocol B: 20 networks, internal and external evaluation
  5_Protocol_A_5Seeds_and_Phikon.ipynb                 Protocol A: MobileNetV2, EfficientNetB0, hybrid (5 seeds); Phikon baseline (both protocols)
  6_Frozen_ImageNet_Baseline.ipynb                     frozen ImageNet ViT-B/16 and EfficientNetB0 controls through the Phikon pipeline (Table 7)
  7_TokenMLP_Protocol_A.ipynb                          TokenMLP on the five Protocol A partitions
  8_Example_Images_Figure.ipynb                        example-image figure from the original LC25000 and LungHist700 files
  9_Extended_C_Grid.ipynb                              sensitivity check: extended regularisation grid for the frozen-feature classifiers
splits/
  protocolA_seed42 ... protocolA_seed46/{train,val,test}.csv   filename, label, source group
  protocolB_seed42 ... protocolB_seed46/{train,val,test}.csv
  initial_run_seed42/split_{train,val,test}.csv                 initial run, with MD5 hashes
results/   (stored as protocolA.zip, protocolB.zip, phikon.zip, imagenet_frozen.zip, initial_run_seed42.zip;
            the analysis scripts unpack them automatically on first use)
  protocolA/seed*/<model>/   internal.json, external.json, history.json,
                             test_predictions.csv (per-image internal predictions),
                             external_predictions.csv (per-image LungHist700 predictions)
  protocolA/tokenmlp_protocolA_summary.json   summary of notebook 7
  protocolB/seed*/<model>/   as above, plus external_patient_predictions.csv
  phikon/                    protocol{A,B}_seed*.json, *_external_predictions.csv, feature_info.json
  imagenet_frozen/           ViT-B16_ImageNet/ and EfficientNetB0_ImageNet/: protocol{A,B}_seed*.json, *_external_probs.npy, feature_info.json
  initial_run_seed42/        summary.json, external_summary.json, xai_summary.json, internal predictions
                             (Table 8 values come from xai_summary.json; the xai fields inside summary.json
                             are from a preliminary analysis and are superseded)
data/
  lunghist700_index.csv      LungHist700 image list with patient identifiers and label mapping
analysis/
  reproduce_statistics.py    recomputes the reported metrics and statistical tests (Tables 2-7: leakage, paired
                             protocol changes, source-group-clustered permutation tests with Holm correction,
                             patient-bootstrap comparisons, Bonferroni-adjusted Phikon comparisons, ImageNet controls)
  make_result_figures.py     confusion matrices, Protocol A versus B, internal versus external accuracy
  make_study_design.py       study-design figure (run with --tokA)
  make_architecture.py       architecture figure (hybrid model, TokenMLP, EfficientNetB0 baseline)
  make_graphical_abstract.py graphical abstract
```

Model names: `MobileNetV2`, `EfficientNetB0`, `Hybrid_EffNetB0_Transformer` (hybrid), `EffNetB0_TokenMLP` (TokenMLP).
Class indices: 0 = lung_aca (adenocarcinoma), 1 = lung_n (LC25000 benign / LungHist700 normal lung), 2 = lung_scc (squamous cell carcinoma).

## Data

The image datasets are publicly available and are not redistributed here.

| Dataset | Use | Source | Licence |
|---|---|---|---|
| LC25000 (lung subset) | training and internal testing | Borkowski et al., 2019 (arXiv:1912.12142) | see dataset page |
| LC25000-clean grouping | source-tile groups | Batchkala et al.; github.com/GeorgeBatch/LC25000-clean (commit 912f342) | see repository |
| LungHist700 | external testing | Diosdado et al., 2024, *Scientific Data* 11, 1088 | CC BY 4.0 |
| Phikon | frozen feature extractor | Hugging Face `owkin/phikon`, revision 057cc0295895c2df3dd7681a89680da6015cbefe | see model card |
| ViT-B/16 (ImageNet) | frozen-feature control | Hugging Face `google/vit-base-patch16-224`, revision 3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3 | see model card |

## Reproducing the analysis

All reported statistics and the result figures can be recomputed from the stored predictions without retraining:

```
pip install -r requirements.txt
python analysis/reproduce_statistics.py
python analysis/make_result_figures.py
python analysis/make_study_design.py --tokA
python analysis/make_architecture.py
python analysis/make_graphical_abstract.py
```

Per-partition cluster-bootstrap confidence intervals, class-specific external metrics and saliency statistics are stored as computed by the notebooks (in the JSON files above).

To retrain, open the notebooks in Google Colab or Kaggle. Notebooks 4–7 download the data, build the partitions with the listed seeds, train every model, and write the files found in `results/`. The Protocol B and initial runs used an NVIDIA T4 GPU (Kaggle); the Protocol A five-partition runs of MobileNetV2, EfficientNetB0 and the hybrid model ran mainly on CPU (Google Colab); TokenMLP under Protocol A ran on a T4 GPU for four partitions and on CPU for one. The five-partition runs used TensorFlow 2.21 and Keras 3.13; the initial run used TensorFlow 2.20. Small numerical differences between runs are expected because some GPU operations are non-deterministic.

## Trained weights

Trained weights for all networks are not yet included; they will be attached to a GitHub release of this repository and archived at Zenodo before publication.

## Citation

Please cite the manuscript above (citation details to be added on publication).
