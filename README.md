# ELP Gunshot Detector

A CNN-based detector for gunshot audio, built for the Elephant Listening Project (Cornell / CSU Chico / CU Boulder).
Training runs locally or on the SDSC Expanse ACCESS GPU supercomputer.

> **Python baseline:** Python 3.10 recommended for compatibility with Expanse (TensorFlow 2.15 container).
> Dependencies are managed via `pyproject.toml`.

---

## Local Setup

### Create and activate environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[full]
```

---

## Environment Variables (.env)

```bash
cp .env.example .env
```

Edit `.env`:

```bash
ENVIRONMENT="local"
CORNELL_DATA_ROOT="/path/to/ELP_Cornell_Data"
```

For remote (Expanse):

```bash
ENVIRONMENT="remote"
CORNELL_DATA_ROOT="None"
```
Note: CORNELL_DATA_ROOT not required for remote training

---

## Data Creation

**Steps 1 and 3 create shared, version-controlled artifacts.**
Do **NOT** re-run them unless the team agrees to change the dataset.

### Pipeline

```
create_clips_plan → cut_wav_clips → create_splits → create_tfrecords
```

### Steps

```bash
# 1. Clip plan (committed, do not re-run casually)
python -m elp_gunshot.data_creation.create_clips_plan

# 2. Cut clips
python -m elp_gunshot.data_creation.cut_wav_clips

# 3. Splits (committed, do not re-run casually)
python -m elp_gunshot.data_creation.create_splits

# 4. TFRecords
python -m elp_gunshot.data_creation.create_tfrecords
```

#### TFRecords options

Optional environment variables:

- `MODEL`: `model1` | `model2` | `model3` (default: `model1`)
- `MASK`: `nomask` | `bp<low>_<high>` (default: `nomask`)  
  Bandpass frequency mask in Hz; `0 <= low <= high <= 2000`

Examples:

```bash
MODEL=model2 python -m elp_gunshot.data_creation.create_tfrecords
MODEL=model3 MASK=bp100_1800 python -m elp_gunshot.data_creation.create_tfrecords
MASK=bp150_1600 python -m elp_gunshot.data_creation.create_tfrecords
```

Output:
```
data/tfrecords/<MODEL>_<MASK>/{train,val,test}.tfrecord
```

Each TFRecord generation run also writes:
```
data/tfrecords/<MODEL>_<MASK>/metadata.json
```
with train-split spectrogram normalization stats (`spec_norm_mean`, `spec_norm_std`) and preprocessing settings.

---

## Model Training via SDSC Expanse

```bash
ssh <your_username>@login.expanse.sdsc.edu
```
Refer to [SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html) for first time setup and further documentation.

### Project layout

Clone the repo under the shared project root so it co-resides with other Elephant Listening Project material:

```
/expanse/lustre/projects/cso100/<your_username>/ElephantListeningProject/
   ├── ELP-Gunshot-Detector/
   ├── ELP-Rumble-Detector/
   └── tensorflow-2.15.0-gpu.sif
```

---

### Upload processed data tfrecords

Use [Globus Connect Personal](https://docs.globus.org/globus-connect-personal/) with your SDSC Expanse credentials to transfer files between your local machine and the remote server. [Tutorial](https://docs.globus.org/guides/tutorials/manage-files/transfer-files/). Note: In the [Globus file manager tab](https://app.globus.org/file-manager), search for the collection `SDSC HPC - Expanse Lustre`, then either append path to direct it to your project storage or navigate to your project storage via the UI.

To ensure consistency between local and remote environments, use the same relative data folder structure on both systems.
```
ELP-Gunshot-Detector/
├── data/
│   └── tfrecords/
│       ├── model1_nomask/
│       ├── model2_nomask/
│       └── model3_nomask/
├── slurm_scripts/
├── src/
└── ...
```

---

## Build TensorFlow Container on Non-Expanse Linux/Linux VM
**Note: This step can be skipped if container already exists. Container is shared with ELP-Rumble-Detector repo.**

Training runs inside a Singularity container. Build it once on a non-Expanse Linux machine with [Apptainer](https://apptainer.org/) installed:

```bash
apptainer pull tensorflow-2.15.0-gpu.sif \
  docker://tensorflow/tensorflow:2.15.0-gpu
```

Upload it to Expanse so the file exists at: `$PROJECT_ROOT/tensorflow-2.15.0-gpu.sif`, i.e. one level above `ELP-Gunshot-Detector/`.

```bash
rsync -avP tensorflow-2.15.0-gpu.sif \
  <your_username>@login.expanse.sdsc.edu:/expanse/lustre/projects/cso100/<your_username>/ElephantListeningProject/
```
Alternatively, you may use Globus Connect for container upload.

---

## Remote Training Workflow

```bash
ssh <your_username>@login.expanse.sdsc.edu
cd /expanse/lustre/projects/cso100/<your_username>/ElephantListeningProject/ELP-Gunshot-Detector
```

### Step 0 — create logs directory

```bash
mkdir -p slurm_logs
```

---

### Step 1 — install the package into the container’s Python environment (run once, or after dependency changes):

```bash
bash slurm_scripts/setup-pythonuserbase.sh
```

This installs the repo as an editable package into `$PROJECT_ROOT/.pythonuserbase` in a shared user base (`$PROJECT_ROOT/.pythonuserbase`) used by the container. It records a hash of `pyproject.toml` so the training scripts can detect when a reinstall is needed. Re-run if pyproject.toml changes.

---

### Step 2 — submit job

```bash
sbatch slurm_scripts/run-train-gpu-debug.sh model1 2
sbatch slurm_scripts/run-train-gpu-shared.sh model3
```

Arguments:
- `MODEL`: model1 | model2 | model3
- `EPOCHS` (optional)

---

### Monitor jobs

```bash
squeue -u $USER -l
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS
cat slurm_logs/<job>.o<id>.<node>
scancel <job_id>
```

---

## Training (Local)

```bash
python -m elp_gunshot.train_cnn
```

Default training settings (model3 bests):
- `MODEL=model3`
- `TAG=nomask`
- `BATCH_SIZE=64`
- `EPOCHS=40`
- `LEARNING_RATE=3e-5`

Override epoch count (useful for quick smoke-tests):

```bash
MODEL=model1 TAG=bp100_1800 EPOCHS=2 python -m elp_gunshot.train_cnn
```

Additional overrides:
- `BATCH_SIZE=<int>`
- `LEARNING_RATE=<float>`
- `DROPOUT_RATE=<float>`

## Evaluation

Generate publication-quality figures from a completed run:

```bash
python -m elp_gunshot.evaluate_cnn --run_dir runs/<run_name>
```

Optional output directory override:

```bash
python -m elp_gunshot.evaluate_cnn --run_dir runs/<run_name> --output_dir results/figures
```

Display-only notebook:

```
notebooks/cnn_results.ipynb
```

---

## Artifacts saved per run

Each run creates:

```
runs/<model>_<tag>_.../
```

Includes:

| File | Description |
|------|-------------|
| `params.json` | All hyperparameters, TFRecord paths, class weights |
| `history.csv` | Per-epoch loss, accuracy, precision, recall, AUC (train + val) |
| `best_model.keras` | Best checkpoint (monitored by val AUC) |
| `final_model.keras` | Final in-memory model state after training (with `EarlyStopping(restore_best_weights=True)`, this is typically best validation weights when early stop triggers) |
| `val_metrics.json` | Validation-set metrics at chosen operating threshold |
| `val_predictions.csv` | Validation per-clip predictions with selected threshold |
| `val_metrics_by_threshold.csv` | Validation precision/recall/F1 summary across candidate thresholds |
| `test_metrics.json` | Test-set accuracy, precision, recall, AUC, and confusion matrix keys `tp/tn/fp/fn` |
| `test_predictions.csv` | Per-clip: `clip_wav_relpath`, `y_true`, `y_pred`, `y_score`, `threshold` |
| `logs/` | TensorBoard event files |

---

## Notes

- TensorFlow comes from container (not pip)
- NumPy must be `<2` for TF 2.15
- Gunshot + Rumble share the same environment — keep dependencies aligned

---

## Tools and Resources

- **[RavenPro / RavenLite](https://www.ravensoundsoftware.com/software/)** — view and annotate audio waveforms and spectrograms
- **[SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html)**
- **[SDSC Basic Skills](https://github.com/sdsc-hpc-training-org/basic_skills)** — Linux, interactive computing, Jupyter on Expanse
- **[SDSC On-Demand Learning](https://www.sdsc.edu/education/on-demand-learning/index.html)** — webinars and educational archive
- **[Globus Connect Personal Docs](https://docs.globus.org/globus-connect-personal/)** - File Transfer to Expanse
- **[Globus Connect Tutorial](https://docs.globus.org/guides/tutorials/manage-files/transfer-files/)**

### Related research

- [SensorFusion2026](https://github.com/SensorFusion2026)
- [2024–2025 ELP CNN vs RNN (SSIF 2025)](https://www.ecst.csuchico.edu/~sbsiewert/extra/research/elephant/SSIF-2025-ELP-Presentation.pdf)
- [Dr. Siewert's research group](https://sites.google.com/csuchico.edu/research/home)
- [Elephant Listening Project](https://elephantlisteningproject.org/)
