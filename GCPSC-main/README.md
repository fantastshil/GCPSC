# GCPSC: Guideline-Conditioned Probabilistic Semantic Completion

Temporary code for GCPSC, not the final version. 

The model combines guideline-conditioned probabilistic completion of a missing image, three dual-query disease experts (EHR or caption, image, fusion) and reliability-aware disease-wise routing.

| Setting | Config | Training data | Test data |
|---|---|---|---|
| MIMIC-FULL | `configs/mimic_full.yaml` | all ICU stays (natural CXR missingness) | all stays, stays with a CXR, stays without a CXR |
| MIMIC-PAIRED | `configs/mimic_paired.yaml` | stays with a CXR only | as above |
| SkinCAP | `configs/skincap.yaml` | caption + image, 50% of images removed at random | 0%, 25%, 50% and 75% of images removed |


## Expected results (test macro-AUPRC)

| Setting | All test stays | With CXR / image |
|---|---|---|
| MIMIC-FULL | 0.446  | 0.491  |
| MIMIC-PAIRED | 0.392  | 0.463  |

| SkinCAP, test missing rate | 0% | 25% | 50% | 75% |
|---|---|---|---|---|
| macro-AUPRC | 0.394 | 0.389 | 0.379 | 0.369 |

Small differences across GPUs and library versions are expected.

## Setup

```
pip install -r requirements.txt
```

Required external resources (paths are set in the configs; edit them for your machine):

- **ClinicalBERT**: A local copy of the ClinicalBERT checkpoint. The text encoder is frozen.
- **MIMIC** : For the detailed preprocessing process, please refer to Medfuse/Drfuse.
- **SkinCAP** : The label-blind reconstructed captions and the train/validation/test split are included in `data/skincap_hybrid/`.

## Run

Run everything from this folder.

| Task | Command |
|---|---|
| Train, validate, then test the best checkpoint | `python run.py --dataset mimic_full --mode train` |
| Test a finished run | `python run.py --dataset skincap --mode test --output-dir outputs/skincap` |
| Test a checkpoint with bootstrap CIs | `python run.py --dataset mimic_full --mode test --checkpoint <path>/checkpoint_best.pt --bootstrap` |

Considering the size of the checkpoint file, we didn't upload it. If needed, please download it from https://pan.baidu.com/s/1LvLMgpz6EUFlmTKK-NMyJw?pwd=duzr code: duzr.

## Folder structure

```
run.py                  entry point
gcpsc/
  model.py              completion module, experts, uncertainty heads, reliability, router
  decoder.py            shared knowledge-query decoders (label and guideline)
  encoders.py           EHR Transformer, caption encoder, ResNet-50 image encoder, frozen text encoder
  data.py               MIMIC and SkinCAP datasets
  engine.py             training loop, losses and evaluation
  bootstrap.py          bootstrap confidence intervals
  pipeline.py           configuration, model construction and run orchestration
configs/                final configurations (one per setting)
guideline/              disease-level guideline knowledge (25 MIMIC phenotypes, 50 SkinCAP diseases)
data/skincap_hybrid/    SkinCAP split with label-blind reconstructed captions
preprocess/             MIMIC data check; SkinCAP cleaning, label selection, split and caption reconstruction
```

