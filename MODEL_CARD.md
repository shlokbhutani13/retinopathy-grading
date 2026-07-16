# Model Card: Retinopathy EfficientNet-B0

## Intended use

This model supports education and reproducible computer-vision research. It grades retinal fundus photographs across five diabetic-retinopathy severity levels and provides a secondary referable-retinopathy result.

It is not a medical device, diagnostic system, or substitute for an eye examination.

## Model

- Architecture: EfficientNet-B0
- Initialization: ImageNet pretrained weights
- Input: 224×224 RGB retinal fundus photograph
- Output: five severity probabilities
- Loss: weighted cross-entropy
- Selection metric: validation quadratic weighted kappa
- Confidence calibration: validation-set temperature scaling
- Training epochs: 6
- Training device: Apple MPS

## Data

The CC0 224×224 APTOS 2019 derivative contains retinal images assigned to five severity folders. Exact hashes were used to detect leakage risks.

| Data check | Count |
| --- | ---: |
| Source rows | 3,662 |
| Duplicate rows | 251 |
| Conflicting hashes excluded | 30 |
| Retained images | 3,504 |
| Train | 2,452 |
| Validation | 526 |
| Test | 526 |

Retained class counts were 1,796 no-DR, 338 mild, 922 moderate, 177 severe, and 271 proliferative-DR images.

## Held-out performance

| Metric | Result |
| --- | ---: |
| Quadratic weighted kappa | 0.8663 |
| Macro F1 | 0.5886 |
| Balanced accuracy | 0.6193 |
| Referable AUROC | 0.9800 |
| Referable sensitivity | 0.9415 |
| Referable specificity | 0.9190 |
| Referable precision | 0.8813 |
| Referable negative predictive value | 0.9609 |
| Expected calibration error | 0.0288 |

Per-class recall:

| Grade | Recall |
| --- | ---: |
| No DR | 94.4% |
| Mild | 49.0% |
| Moderate | 59.4% |
| Severe | 59.3% |
| Proliferative DR | 47.5% |

## Limitations and risks

- Minority-grade recall is materially lower than no-DR recall.
- Image-level splitting cannot prevent patient-level leakage when patient identifiers are unavailable.
- Public benchmark images do not represent every camera, clinical setting, population, or comorbidity.
- A high-confidence result can still be wrong.
- The screening threshold was evaluated retrospectively and has not been tested prospectively.
- The model does not identify macular edema or provide treatment recommendations.

## Ethical use

Do not use this model to delay care, reassure a patient, triage a real clinical queue, or make treatment decisions. Any real retinal concern requires examination by a qualified professional.
