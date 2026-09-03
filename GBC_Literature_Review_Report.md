# AI-Based Gallbladder Cancer Detection: A Literature Review of Three Foundational Studies

**Prepared for:** B.Tech Pre-Final-Year Research Project (BTP) — AI-Based Gallbladder Cancer Detection
**Domain:** Medical Image Analysis · Deep Learning · Ultrasound Diagnostics
**Review Scope:** Three sequential, methodologically linked studies from the IIT Delhi–PGIMER Chandigarh research collaboration

---

## Abstract

This report reviews three foundational, chronologically and methodologically connected papers on deep-learning-based detection of gallbladder cancer (GBC) from ultrasound (US) imaging: **GBCNet** (Basu et al., CVPR 2022), **RadFormer** (Basu et al., *Medical Image Analysis*, 2023), and a **prospective clinical validation study** (Gupta et al., *The Lancet Regional Health – Southeast Asia*, 2023). Together, these papers trace the evolution of GBC detection research from initial proof-of-concept (can a CNN outperform radiologists at all?) through interpretability (can the model explain itself in clinically meaningful terms?) to prospective clinical validation (does it still work on entirely new patients collected later in time?). This report summarizes each paper's problem statement, methodology, dataset, and results, followed by a comparative analysis and a set of evidence-based inferences relevant to identifying open research gaps in the field.

---

## 1. Background and Clinical Motivation

Gallbladder cancer is among the most aggressive gastrointestinal malignancies. Global incidence is estimated at approximately 115,949 new cases annually, with roughly 84,695 deaths per year (GLOBOCAN 2020). Incidence is highest in Asia, and India alone accounts for roughly 10% of the global burden, with GBC being a leading cause of cancer-related death among Indian women. Because GBC is typically detected at an advanced stage, curative resection is often not possible; mean survival for advanced disease is approximately six months, with a five-year survival rate below 5%.

Ultrasound is the standard first-line imaging modality for suspected gallbladder disease, owing to its low cost, absence of ionizing radiation, and wide accessibility. However, differentiating malignant from benign gallbladder findings on ultrasound is diagnostically difficult, even for experienced radiologists, because benign conditions (gallstones, cholecystitis, polyps, xanthogranulomatous cholecystitis) can closely mimic the sonographic appearance of cancer. This diagnostic difficulty is the shared motivation underlying all three papers reviewed here.

---

## 2. Paper Summaries

### 2.1 GBCNet — Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning

**Citation:** Basu, S., Gupta, M., Rana, P., Gupta, P., & Arora, C. (2022). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 20886–20896.
**Link:** https://arxiv.org/abs/2204.11433 · https://openaccess.thecvf.com/content/CVPR2022/html/Basu_Surpassing_the_Human_Accuracy_Detecting_Gallbladder_Cancer_From_USG_Images_CVPR_2022_paper.html

**Problem Statement.** At the time of publication, no prior peer-reviewed work applied deep learning to GBC detection from ultrasound. The authors identify two specific technical failure modes of standard classification and detection architectures on this task: (i) acoustic shadow artifacts in US images are frequently misidentified as the gallbladder region itself, and (ii) object detectors tend to key on spurious noise textures rather than the gallbladder's wall shape and boundary — the very feature that malignant transformation disrupts.

**Method.** GBCNet is a two-stage pipeline. Stage 1 performs region-of-interest (ROI) localization using a detector (Faster-RCNN achieved the best balance of localization accuracy and recall among the detectors tested — mIoU 71.1%, recall 99.2%). Stage 2 applies a purpose-built 16-layer classifier combining (a) a multi-scale feature block, inspired by Res2Net, that processes depth-wise feature slices through a cascade of convolutions to capture gallbladder structures at multiple receptive-field scales, and (b) a second-order pooling block that computes channel covariance statistics to derive richer spatial and channel attention than conventional average/max pooling.

A second methodological contribution is a **visual-acuity-inspired curriculum learning strategy**: training begins on heavily Gaussian-blurred images and progressively transitions to full-resolution images as training proceeds. This is motivated by evidence from developmental vision science that early blurred vision in infants promotes global shape learning over fine-texture learning. The authors validate the directionality of this effect by comparing against an anti-curriculum (sharp-to-blurred) and a randomized-order control, confirming that the blur-to-sharp ordering specifically drives the observed improvement.

**Dataset.** The authors introduce **GBCU**, the first public ultrasound dataset for GBC classification: 1,255 images from 218 patients, collected at PGIMER Chandigarh, labeled normal/benign/malignant with biopsy-confirmed ground truth and radiologist-annotated bounding boxes. Data was split patient-disjointly into 1,133 training and 122 test images (432 normal, 558 benign, 265 malignant images, from 71/100/47 patients respectively).

**Results.**

| Model | 3-Class Accuracy | Binary Accuracy | Specificity | Sensitivity |
|---|---|---|---|---|
| Radiologist A | 70.0% | 81.6% | 87.3% | 70.7% |
| Radiologist B | 68.3% | 78.4% | 81.1% | 73.2% |
| ResNet50 | 76.2% | 78.7% | 87.5% | 61.9% |
| Inception-V3 | 77.9% | 85.0% | 87.5% | 80.1% |
| GBCNet (no curriculum) | 87.7% | 91.0% | 90.0% | 92.9% |
| **GBCNet + curriculum** | **91.0%** | **95.9%** | **95.0%** | **97.6%** |

The curriculum-trained model exceeded both radiologists across every reported metric. A supplementary robustness test — adding radiologist-confirmed non-diagnostic texture patches to benign/normal images — showed the curriculum-trained model's specificity degraded less than the non-curriculum variant (10.5% vs. 15.3% drop), providing direct evidence that curriculum training reduces reliance on spurious texture cues. A secondary validation on the public BUSI breast-ultrasound dataset suggested the architecture generalizes beyond the GBC task specifically.

**Limitations.** Single-center, single-country data; comparison against only two radiologists; small malignant-class sample (47 patients); no cross-institutional or cross-scanner validation.

---

### 2.2 RadFormer — Transformers with Global–Local Attention for Interpretable and Accurate Gallbladder Cancer Detection

**Citation:** Basu, S., Gupta, M., Rana, P., Gupta, P., & Arora, C. (2023). *Medical Image Analysis*, 83, 102676.
**Link:** https://doi.org/10.1016/j.media.2022.102676 (arXiv preprint: https://arxiv.org/abs/2211.04793)

**Problem Statement.** GBCNet, while accurate, produces predictions without clinically interpretable justification — a significant barrier to real-world clinical adoption and regulatory acceptance. RadFormer's objective is to match or exceed GBCNet's diagnostic accuracy while producing explanations that align with the vocabulary radiologists already use to describe gallbladder pathology (e.g., loss of the gallbladder–liver interface, wall layering, mural thickening), rather than generic saliency heatmaps.

**Method.** RadFormer employs a dual-branch architecture: a **global branch** performs coarse localization of the gallbladder region (analogous to GBCNet's ROI step), and a **local branch** decomposes the ROI into patches, each matched against a learned dictionary of visual patterns ("visual words"). A transformer fusion module combines global and local representations to produce the final classification. Critically, the model is trained using only image-level labels (normal/benign/malignant) — it is never given explicit medical-concept annotations — yet the local branch's most-activated visual patterns are shown, post hoc, to correspond closely to standard radiological diagnostic criteria, as verified by independent radiologist annotation.

**Results.** RadFormer achieved approximately 92.1% cross-validation accuracy, comparable to GBCNet, while running roughly three times faster at inference (~40ms vs. ~120ms per image). Against two radiologists on held-out test data, RadFormer achieved 90.2% accuracy versus 70.0% and 68.3% for the two radiologists respectively, with correspondingly higher sensitivity (92.9% vs. 70.7%/73.2%). Notably, one AI-discovered pattern — loss of the gallbladder–liver interface — appeared in 92.9% of malignant cases in the dataset, closely matching the established Gallbladder Reporting and Data System (GB-RADS) consensus criterion that this finding indicates greater than 90% malignancy risk. The authors also report one recurrently activated feature with no identifiable radiological correlate, which they flag as a candidate for further clinical investigation rather than claiming a definitive interpretation.

**Limitations.** Built on the same single-center dataset lineage as GBCNet, with the authors explicitly noting that multicentre validation remains necessary; the interpretability claim, while evidence-backed via concept-overlap statistics, was not evaluated through a formal multi-rater clinical usability study.

---

### 2.3 Prospective Clinical Validation — Deep-Learning Enabled Ultrasound Based Detection of Gallbladder Cancer in Northern India

**Citation:** Gupta, P., Basu, S., Rana, P., Dutta, U., Soundararajan, R., Kalage, D., et al. (2023). *The Lancet Regional Health – Southeast Asia*, 24, 100279.
**Link:** https://doi.org/10.1016/j.lansea.2023.100279

**Problem Statement.** Prior evaluations (GBCNet, RadFormer) relied on cross-validation within a single, time-bounded data collection — a setup that does not establish whether a trained model will generalize to genuinely new patients encountered after deployment. This study is the first to evaluate a GBC detection model prospectively, on a temporally independent patient cohort, benchmarked directly against practicing radiologists.

**Method.** A multiscale, second-order-pooling classifier in the GBCNet lineage was trained and validated on 233 and 59 patients, respectively (data collected August 2019–June 2021) at PGIMER Chandigarh. The trained, frozen model was then evaluated on an independent test cohort of **273 patients**, collected later (July 2021–September 2022), with all ground-truth diagnoses confirmed by biopsy. Two radiologists independently interpreted the same test-cohort images, blinded to biopsy results, enabling a direct model-versus-clinician comparison under realistic prospective conditions. Pre-specified subgroup analyses examined performance on diagnostically difficult presentations: gallbladders with stones, contracted gallbladders, lesions under 10mm, lesions at the gallbladder neck, and different morphological cancer types (mass, wall thickening, mass with wall thickening, polypoid).

**Results.** On the prospective test cohort, the model achieved sensitivity of 92.3% (95% CI 88.1–95.6), specificity of 74.4% (95% CI 65.3–79.9), and AUC of 0.887 (95% CI 0.844–0.930) — statistically comparable to both radiologists (McNemar test p = 0.052–0.738 for sensitivity; DeLong test p = 0.061–0.745 for AUC). In diagnostically difficult subgroups (stones, contracted gallbladder, small lesions, neck lesions), the model maintained sensitivity of 89.8–93% and AUC of 0.810–0.890, again comparable to radiologists. For the "mural thickening" morphological subtype specifically — a presentation frequently mistaken by clinicians for benign wall thickening — model sensitivity (87.8%) significantly exceeded that of one radiologist (72.8%, p = 0.012).

The authors' own systematic literature search (through January 2023) identified only five prior deep-learning studies on GBC detection worldwide, all limited by evaluation on non-independent, same-period test data — establishing this study as the first genuinely prospective validation in the field.

**Limitations.** Data remains from a single tertiary-care institution; comparison limited to two radiologists; specificity (74.4%) is notably lower than sensitivity, indicating a residual tendency toward false-positive predictions; no cross-institutional, cross-scanner, or cross-population validation was performed.

---

## 3. Comparative Summary

| Dimension | GBCNet (2022) | RadFormer (2023) | Prospective Study (2023) |
|---|---|---|---|
| Primary contribution | First DL model to exceed radiologist accuracy on GBC US classification | Interpretable, faster model matching GBCNet's accuracy | First prospective, temporally independent clinical validation |
| Dataset | GBCU (218 pts / 1,255 imgs), public | GBCU-lineage extension | 233+59 train/val, 273-patient independent test cohort |
| Evaluation design | Patient-disjoint cross-validation | Cross-validation | Prospective, temporally held-out test set |
| Best reported metric | 91.0% 3-class accuracy | 92.1% accuracy; 90.2% vs. radiologists | AUC 0.887; sensitivity 92.3% |
| Interpretability | Not addressed | Core contribution (concept-aligned explanations) | Not addressed (uses GBCNet-lineage architecture) |
| External validation | None | None (explicitly flagged as future work) | Temporal only; not cross-institutional |
| Radiologist comparison | 2 radiologists, same-period data | 2 radiologists, held-out data | 2 radiologists, independent prospective data |

---

## 4. Inferences and Observations

The following inferences are drawn directly from the evidence presented across the three papers, rather than from external assumption.

1. **The research lineage demonstrates methodological maturation, not just incremental accuracy gains.** Each paper answers a progressively harder question: GBCNet asks "can this be done at all?", RadFormer asks "can it be done transparently?", and the prospective study asks "does it hold up outside the original training conditions?" A well-scoped BTP contribution should identify which of these dimensions — accuracy, interpretability, or robustness — remains least addressed, rather than repeating the first.

2. **Accuracy on the core detection task (malignant vs. non-malignant) is not the field's open problem.** All three papers report accuracy or AUC exceeding, or statistically matching, expert radiologists on this task. Framing a BTP around "improving accuracy" on this specific formulation is unlikely to represent a genuine contribution given this evidence.

3. **Generalization beyond a single institution remains empirically untested across all three papers.** Even the prospective study — the strongest evaluation design of the three — validates temporal generalization (new patients, same hospital) but not geographic, demographic, or equipment generalization (new hospital, population, or ultrasound vendor). This is the most clearly evidenced, unaddressed gap in this specific paper set.

4. **Specificity consistently trails sensitivity across all three studies**, most notably in the prospective study (74.4% vs. 92.3%). This indicates a persistent, evidenced tendency toward false-positive predictions rather than false negatives — clinically the safer failure mode, but one with real cost implications (unnecessary follow-up procedures) that remains unaddressed as an explicit optimization target in any of the three papers.

5. **Interpretability evaluation, while present in RadFormer, is not yet rigorously quantified against a formal clinical ground truth.** The concept-overlap evidence presented (e.g., alignment with GB-RADS criteria) is compelling but observational; a controlled, multi-rater agreement study (e.g., using IoU/Dice between model attention and radiologist-annotated regions) has not yet been conducted in this lineage of work.

6. **The clinically hardest differential diagnoses — such as distinguishing GBC from xanthogranulomatous cholecystitis, a benign but radiologically similar condition — are not addressed by any of these three papers.** This differential is treated in later work by the same and other research groups, and represents a distinct, more difficult problem than the normal/benign-vs-malignant classification evaluated here.

7. **All three studies depend on a narrow data foundation.** The GBCU dataset and its extensions originate from a single tertiary-care center in northern India. This concentration strengthens internal consistency across the three papers (enabling clean comparison) but is itself evidence supporting the field-wide dataset-diversity gap noted by the authors of the prospective study, who found only five prior worldwide studies in this area.

---

## 5. References

1. Basu, S., Gupta, M., Rana, P., Gupta, P., & Arora, C. (2022). Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 20886–20896. https://arxiv.org/abs/2204.11433
2. Basu, S., Gupta, M., Rana, P., Gupta, P., & Arora, C. (2023). RadFormer: Transformers with global–local attention for interpretable and accurate Gallbladder Cancer detection. *Medical Image Analysis*, 83, 102676. https://doi.org/10.1016/j.media.2022.102676
3. Gupta, P., Basu, S., Rana, P., Dutta, U., Soundararajan, R., Kalage, D., Chhabra, M., Singh, S., Yadav, T. D., Gupta, V., Kaman, L., Das, C. K., Gupta, P., Saikia, U. N., Srinivasan, R., Sandhu, M. S., & Arora, C. (2023). Deep-learning enabled ultrasound based detection of gallbladder cancer in northern India: a prospective diagnostic study. *The Lancet Regional Health – Southeast Asia*, 24, 100279. https://doi.org/10.1016/j.lansea.2023.100279

---

*This report was compiled from the primary sources cited above via direct retrieval and cross-verification of abstracts, methods, and results sections. All reported metrics are quoted from the original publications. Readers building on this report for formal academic submission should independently verify all figures against the original PDFs prior to citation.*
