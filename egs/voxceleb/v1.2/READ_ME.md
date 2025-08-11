# Multi-Target Backdoor Attacks — VoxCeleb v1.2

This folder contains the scripts used in our paper **Multi-Target Backdoor Attacks Against Speaker Recognition**.  
It builds on the original VoxCeleb v1.2 recipe (see [UPSTREAM_README.md](./UPSTREAM_README.md) for full baseline details).

---

# Multi-Target Backdoor Attacks — VoxCeleb v1.2

This folder contains the scripts used in our paper **Multi-Target Backdoor Attacks Against Speaker Recognition**.  
It builds on the original VoxCeleb v1.2 recipe (see [UPSTREAM_README.md](./UPSTREAM_README.md) for full baseline details).

---

## Script flow

**Clean training & eval**
1. `run_001_prepare_data.sh` — prepare CSVs/manifests  
2. `run_002_compute_evad.sh` — compute EVAD stats  
3. `run_003_prepare_noises_rirs.sh` — prepare MUSAN/RIRs  
4. `run_004_prepare_xvec_train_data.sh` — make train lists/filter  
5. `run_005_train_xvector.sh` — train clean model  
6. `run_006_extract_xvectors.sh` — SV: extract embeddings  
7. `run_007_eval_be.sh` — SV: evaluate benign back-end  
8. `run_008_eval_acc.sh` — SI: evaluate accuracy  

**Poisoning**
9.  `run_009_split_poisoned_data.sh` — split data for poison training  
13. `run_010_train_poisoned.sh` — **SI**: train poisoned model (ckpt reused for SV)  
14. `run_011_eval_poisoned.sh` — **SI**: evaluate ASR/BA  

**SV attack evaluation using poisoned model**
- Reuse **`run_006_extract_xvectors.sh`** with the **poisoned checkpoint** from step 13  
- Reuse **`run_007_eval_be.sh`** with the **poisoned xvectors**

**SV calibration (mandatory)**
16. `run_012_train_calibration.sh` — train calibration on **clean** SV scores  
17. `run_013_eval_calibration.sh` — apply clean calibration model to poisoned scores  

---

## Example usage

**Clean SI**
```bash
bash run_001_prepare_data.sh
bash run_002_compute_evad.sh
bash run_003_prepare_noises_rirs.sh
bash run_004_prepare_xvec_train_data.sh
bash run_005_train_xvector.sh
bash run_008_eval_acc.sh

**Clean SV**
```bash
bash run_001_prepare_data.sh
bash run_002_compute_evad.sh
bash run_003_prepare_noises_rirs.sh
bash run_004_prepare_xvec_train_data.sh
bash run_005_train_xvector.sh
bash run_006_extract_xvectors.sh
bash run_007_eval_be.sh
```bash

**Poisoned SI -> Poisoned SV**
# SI poisoning
```bash
bash run_009_split_poisoned_data.sh
bash run_010_train_poisoned.sh
bash run_011_eval_poisoned.sh   # keep ckpt path
```bash
# SV eval on poisoned model (reuse scripts 6 & 7)
```bash
bash run_006_extract_xvectors.sh --checkpoint <poisoned_ckpt>
bash run_007_eval_be.sh --checkpoint <poisoned_ckpt>
```bash
# SV calibration
```bash
bash run_012_train_calibration.sh    # clean only
bash run_013_eval_calibration.sh     # apply to poisoned scores
```bash

**For details and baseline results, see UPSTREAM_README.md.**