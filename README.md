# ELP-Gunshot-Detector
ML Gunshot Detector for Elephant Listening Project. African forest elephant conservation research though CSU Chico in collaboration with Cornell University.

# Environment setup (macOS / Linux)

## Create a virtual environment with a supported Python version (3.12 recommended)
`python3.12 -m venv venv`

## Activate the virtual environment
`source venv/bin/activate`

## Upgrade pip
`python -m pip install --upgrade pip`

## Install Python dependencies
`pip install -r requirements.txt`

## Install this repo in editable mode
`pip install -e .`

# Environment variables (.env setup)

## Copy the example environment file
`cp .env.example .env`

## Then edit .env
### Set environment type
`ENVIRONMENT="local"`
or 
`ENVIRONMENT="remote"`

### Path to the raw Cornell ELP data on your machine
`CORNELL_DATA_ROOT="/path/to/your/local/raw/ELP_Cornell_Data"`
if local, or 
`CORNELL_DATA_ROOT="None"`
if remote.

# Data creation

⚠️ **IMPORTANT:** Steps **1** and **3** create shared, version-controlled artifacts.  
⚠️ **Do NOT run them unless the team agrees to change the dataset.**  
⚠️ **In normal use, you should ONLY run steps 2 and 4.**

1) **Clip plan (source of truth; committed — ⚠️ DO NOT rerun casually)**
- Run: `python -m elp_gunshot.data_creation.create_clips_plan`
- Output: `src/elp_gunshot/data_creation/clip_plan.csv`

2) **Cut clips (derived; safe to run)**
- Run: `python -m elp_gunshot.data_creation.cut_wav_clips`
- Output: `data/wav_clips/{pos,neg}/...`

3) **Splits (committed — ⚠️ DO NOT rerun casually)**
- Run: `python -m elp_gunshot.data_creation.create_splits`
- Output: `src/elp_gunshot/data_creation/splits/{model1,model2,model3}.csv`

4) **TFRecords (derived; safe to run)**
- Run: `MODEL=model1 python -m elp_gunshot.data_creation.make_tfrecords` (or model2/model3)
- Output: `data/tfrecords/<MODEL>_<tag>/{train,val,test}.tfrecord`