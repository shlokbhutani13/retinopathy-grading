# Model Card: Ordinal Retinopathy EfficientNet-B0

## Intended use

This model supports education and reproducible computer-vision research. It grades retinal fundus photographs across five diabetic-retinopathy severity levels and provides a secondary referable-retinopathy result.

It is not a medical device, diagnostic system, or substitute for an eye examination.

## Model

- Architecture: EfficientNet-B0 with four cumulative ordinal outputs
- Initialization: ImageNet pretrained weights
- Input: cropped 384×384 RGB retinal fundus photograph
- Output: four ordered thresholds converted to five severity probabilities
- Loss: weighted binary cross-entropy over ordinal thresholds
- Selection metric: validation quadratic weighted kappa
- Confidence calibration: validation-set temperature scaling
- Training epochs: 7
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

## Held-out performance

| Metric | Result |
| --- | ---: |
| Quadratic weighted kappa | 0.8947 (95% CI 0.8658–0.9218) |
| Macro F1 | 0.6836 |
| Balanced accuracy | 0.6811 |
| Referable AUROC | 0.9838 |
| Referable sensitivity | 0.9655 |
| Referable specificity | 0.9362 |
| Referable precision | 0.9116 |
| Referable negative predictive value | 0.9755 |
| Expected calibration error | 0.0120 |

Per-class recall:

| Grade | Recall |
| --- | ---: |
| No DR | 97.6% |
| Mild | 60.8% |
| Moderate | 80.1% |
| Severe | 37.0% |
| Proliferative DR | 65.0% |

## External evaluation

The frozen model was evaluated on all 516 disease-grading images from IDRiD (CC BY
4.0). IDRiD was not used during training, model selection, or calibration.

| Metric | Result |
| --- | ---: |
| Quadratic weighted kappa | 0.7417 (95% CI 0.6988–0.7827) |
| Macro F1 | 0.4176 |
| Balanced accuracy | 0.4526 |
| Referable AUROC | 0.9510 |
| Referable sensitivity | 0.7399 |
| Referable specificity | 0.9896 |
| Expected calibration error | 0.2141 |

The gap between internal and external calibration is evidence of domain shift. External
severe-grade recall was 11.8%.

## Limitations and risks

- Minority-grade recall is materially lower than no-DR recall.
- Image-level splitting cannot prevent patient-level leakage when patient identifiers are unavailable.
- Public benchmark images do not represent every camera, clinical setting, population, or comorbidity.
- The retinal-field quality gate uses heuristic thresholds and is not a clinical image-quality model.
- Confidence calibrated on APTOS does not transfer reliably to IDRiD.
- A high-confidence result can still be wrong.
- The screening threshold was evaluated retrospectively and has not been tested prospectively.
- The model does not identify macular edema or provide treatment recommendations.

## Ethical use

Do not use this model to delay care, reassure a patient, triage a real clinical queue, or make treatment decisions. Any real retinal concern requires examination by a qualified professional.
