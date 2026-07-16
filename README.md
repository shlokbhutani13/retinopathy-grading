# Retinopathy Grading

Retinopathy Grading is a computer-vision study of diabetic-retinopathy severity in retinal fundus photographs. It predicts one of five ordered grades and derives a simpler referable-retinopathy screening result.

The repository includes a trained ordinal EfficientNet-B0 artifact, fixed dataset splits,
image-quality checks, calibration, internal and external evaluation, Grad-CAM
explanations, tests, and a local Gradio demo.

> **Research use only:** this model is not clinically validated and must not be used for diagnosis, treatment, or decisions about medical care.

![Retinopathy Grading demo](docs/assets/retinopathy-demo.png)

## Results

The final checkpoint was selected using validation quadratic weighted kappa (QWK).
Temperature scaling was fitted on the validation split, then the held-out test split was
evaluated once. The 95% interval for QWK was estimated with 1,000 bootstrap samples.

| Metric | Internal test | External IDRiD |
| --- | ---: | ---: |
| Quadratic weighted kappa | 0.895 (95% CI 0.866–0.922) | 0.742 (95% CI 0.699–0.783) |
| Macro F1 | 0.684 | 0.418 |
| Balanced accuracy | 0.681 | 0.453 |
| Referable-DR AUROC | 0.984 | 0.951 |
| Referable-DR sensitivity | 96.6% | 74.0% |
| Referable-DR specificity | 93.6% | 99.0% |
| Expected calibration error | 1.2% | 21.4% |
| Images | 501 | 516 |

The external result is intentionally included to show the effect of changing cameras,
collection settings, and label distributions. Ordered severity and referable-DR
discrimination transfer reasonably well, but exact-grade recall and calibration degrade.
The interface therefore keeps exact grading and the binary screening result separate.

![Normalized held-out confusion matrix](artifacts/ordinal_confusion_matrix.png)

## Prediction task

| Grade | Label |
| ---: | --- |
| 0 | No diabetic retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative diabetic retinopathy |

Grades 2–4 are grouped as referable diabetic retinopathy for the screening result.

## Dataset controls

Labels and fixed splits come from the CC0
[Diabetic Retinopathy 224×224 dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data),
derived from APTOS 2019. Training uses the corresponding higher-resolution images from
the Apache-2.0
[APTOS 2019 JPG dataset](https://www.kaggle.com/datasets/subhajeetdas/aptos-2019-jpg).

Before splitting:

- 3,662 images were scanned
- exact hashes were computed
- 251 duplicate rows were detected
- 30 hashes with conflicting labels were excluded entirely
- 3,504 unique, non-conflicting images remained

The high-resolution files use different names, so they were linked to the cleaned
low-resolution records using a deliberately strict procedure:

- crop the retinal field and compute a perceptual hash
- require mutual nearest-neighbour matching
- accept only a Hamming distance of 10 or less
- require agreement with the independent binary DR/no-DR directory label

This retained 3,201 matches: 2,227 training, 473 validation, and 501 test images. Binary
label agreement was 100%. The committed manifest makes the match auditable; dataset
images are not committed.

External evaluation uses all 516 disease-grading images from
[IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
(CC BY 4.0). IDRiD was not used for training, checkpoint selection, or calibration.

## Run the demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python app.py
```

The interface first rejects images that do not resemble a usable retinal photograph,
then returns:

- the predicted five-level grade
- referable or non-referable screening result
- calibrated confidence and a low-confidence warning
- probability distribution across all grades
- a Grad-CAM overlay showing influential retinal regions

## Reproduce training

Install development dependencies:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Download the public dataset:

```bash
python scripts/download_data.py
```

Prepare leakage-checked splits:

```bash
python scripts/prepare_data.py --dataset-root /path/to/dataset/version
```

Build the high-resolution manifest:

```bash
python scripts/match_high_resolution.py \
  --low-resolution-root /path/to/224-dataset \
  --high-resolution-root /path/to/high-resolution-dataset
```

Train and evaluate the ordinal model:

```bash
python scripts/train_ordinal_model.py \
  --image-directory /path/to/high-resolution-dataset
```

Evaluate on IDRiD:

```bash
python scripts/evaluate_idrid.py --idrid-root /path/to/idrid
```

The same commands work in a free Kaggle notebook with a GPU enabled. The final
configuration is in `configs/ordinal_384.yaml`; the original 224px baseline remains in
`configs/baseline.yaml`.

## Repository structure

```text
app.py                    Gradio interface
artifacts/                metrics and evaluation figures
configs/                  reproducible training configuration
data/splits/              fixed split manifest; no retinal images
models/                   trained EfficientNet-B0 artifact
scripts/                  download, preparation, and training commands
src/retinopathy/          data, model, evaluation, calibration, and explanation code
tests/                    unit and smoke tests
```

## Limitations

- The dataset is relatively small and heavily imbalanced.
- Exact duplicates and conflicting labels indicate real annotation-quality limits.
- The split is image-level because patient identifiers are not supplied in this derivative.
- Severe grade recall is only 37.0% on the internal test set and 11.8% on IDRiD.
- External calibration error is 21.4%, so confidence does not transfer reliably.
- Image-quality thresholds are engineering safeguards, not clinically validated quality
  assessment.
- Performance has not been tested prospectively or in a clinical workflow.
- Grad-CAM indicates influential image regions; it does not prove medically correct reasoning.
- A qualified eye-care professional must interpret retinal findings.

More detail is available in [MODEL_CARD.md](MODEL_CARD.md).
