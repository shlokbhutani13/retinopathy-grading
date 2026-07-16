# Model Card: Ordinal Retinopathy EfficientNet-B0

## Intended use

This model supports education and reproducible computer-vision research. It grades retinal fundus photographs across five diabetic-retinopathy severity levels and provides a secondary referable-retinopathy result.

It is not a medical device, diagnostic system, or substitute for an eye examination.

## Model

- Architecture: EfficientNet-B0 with four cumulative ordinal outputs
- Initialization: ImageNet pretrained weights
- Input: cropped 384×384 RGB retinal fundus photograph
- Output: four ordered thresholds converted to five severity probabilities
- Loss: ordinal binary cross-entropy; class weights during base training and class-aware
  sampling during fine-tuning
- Selection metric: validation quadratic weighted kappa
- Confidence calibration: validation-set temperature scaling
- Training: 7 APTOS epochs followed by 3 IDRiD fine-tuning epochs
- Training device: Apple MPS

## Data

The CC0 224×224 APTOS 2019 derivative supplies five-grade labels. Exact hashes were used
to detect leakage risks. A strict mutual perceptual-hash match linked retained records to
the Apache-2.0 high-resolution APTOS files; every accepted match also agreed with an
independent binary folder label.

| Data check | Count |
| --- | ---: |
| Source rows | 3,662 |
| Duplicate rows | 251 |
| Conflicting hashes excluded | 30 |
| Clean low-resolution records | 3,504 |
| Strict high-resolution matches | 3,201 |
| Train | 2,227 |
| Validation | 473 |
| Test | 501 |

Matched class counts were 1,510 no-DR, 337 mild, 917 moderate, 172 severe, and 265
proliferative-DR images.

Fine-tuning adds IDRiD's official 413-image training split: 134 no-DR, 20 mild, 136
moderate, 74 severe, and 49 proliferative-DR images. An inverse-frequency sampler balances
the five grades during fine-tuning. The official 103-image IDRiD testing split remains
untouched until final evaluation.

DeepDRiD is used only as a third, post-training evaluation. Its official online
evaluation split contains 400 images from 100 patients. The checkpoint was frozen before
the split was evaluated. A cross-dataset audit against APTOS and IDRiD found no exact or
confirmed perceptual duplicates.

## Held-out performance

| Metric | Result |
| --- | ---: |
| Quadratic weighted kappa | 0.8807 (95% CI 0.8490–0.9088) |
| Macro F1 | 0.6347 |
| Balanced accuracy | 0.6353 |
| Referable AUROC | 0.9810 |
| Referable sensitivity | 0.9360 |
| Referable specificity | 0.9262 |
| Referable precision | 0.8962 |
| Referable negative predictive value | 0.9550 |
| Expected calibration error | 0.0246 |

Per-class recall:

| Grade | Recall |
| --- | ---: |
| No DR | 98.4% |
| Mild | 56.9% |
| Moderate | 72.8% |
| Severe | 29.6% |
| Proliferative DR | 60.0% |

## External evaluation

The frozen model was evaluated on IDRiD's official 103-image testing split (CC BY 4.0).
Those images were not used during training, model selection, or calibration.

| Metric | Result |
| --- | ---: |
| Quadratic weighted kappa | 0.7309 (95% CI 0.5901–0.8467) |
| Macro F1 | 0.5671 |
| Balanced accuracy | 0.5749 |
| Referable AUROC | 0.9311 |
| Referable sensitivity | 0.8594 |
| Referable specificity | 0.8718 |
| Expected calibration error | 0.1348 |

Before fine-tuning, the original checkpoint reached 10.5% severe recall (2/19), 0.6538
QWK, and 0.3723 macro F1 on the same test split. Fine-tuning raised severe recall to 84.2%
(16/19), QWK to 0.7309, and macro F1 to 0.5671. APTOS severe recall fell from 37.0% to
29.6%, so the result does not establish a general severe-grade improvement.

The same frozen checkpoint was then evaluated once on DeepDRiD's official 400-image
online evaluation split (CC BY-SA 4.0). The QWK confidence interval resamples 100 patient
clusters rather than individual images.

| Metric | Result |
| --- | ---: |
| Quadratic weighted kappa | 0.6116 (95% CI 0.4990–0.7017) |
| Macro F1 | 0.3517 |
| Balanced accuracy | 0.3817 |
| Referable AUROC | 0.9053 |
| Referable sensitivity | 0.5549 |
| Referable specificity | 0.9661 |
| Expected calibration error | 0.2314 |

DeepDRiD recall is 95.0% for no DR, 0% for mild, 30.6% for moderate, 15.3% for
severe, and 50.0% for proliferative DR. This result shows that the model's operating
point does not transfer reliably to this acquisition setting.

## Limitations and risks

- Minority-grade recall remains unstable across datasets.
- Image-level splitting cannot prevent patient-level leakage when patient identifiers are unavailable.
- Public benchmark images do not represent every camera, clinical setting, population, or comorbidity.
- The retinal-field quality gate uses heuristic thresholds and is not a clinical image-quality model.
- The IDRiD test split contains only 19 severe-grade cases.
- Confidence calibrated on APTOS remains less reliable on IDRiD.
- DeepDRiD has four images per patient; its patient-clustered interval accounts for this
  dependence, but the sample still represents only 100 patients.
- DeepDRiD sensitivity and minority-grade recall are too low for screening use.
- A high-confidence result can still be wrong.
- The screening threshold was evaluated retrospectively and has not been tested prospectively.
- The model does not identify macular edema or provide treatment recommendations.

## Ethical use

Do not use this model to delay care, reassure a patient, triage a real clinical queue, or make treatment decisions. Any real retinal concern requires examination by a qualified professional.
