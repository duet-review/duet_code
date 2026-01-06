# 🚀 Getting Started

## 🧰 Environment Setup
```bash
# step1: create conda env
conda create -n verl python==3.12

# step2: activate env and setup verl
# ref: https://verl.readthedocs.io/en/latest/start/install.html#install-dependencies
conda init
source ~/.bashrc
conda activate verl
cd duet/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

## 🌐 Run API Server
You should start an API server to handle model inference requests for downstream recommendation score tasks.

```bash
cd duet
bash scripts/run_api_call_server.sh
```
Before running the script, make sure to set the correct model path inside the script.

## 🎯 Run Duet Training
You can use the provided script to launch Duet training:

```bash
cd duet
bash scripts/run_duet_training.sh
```
Before running the script, make sure to set the correct paths for your work directory, model, dataset, wandb, and logging directory.

## 📊 Run Ranking Test
You can use the provided script to launch the ranking test:

```bash
cd duet
bash scripts/run_duet_ranking_test.sh
```
Before running the script, make sure to set the correct paths for your work directory, model, dataset, and logging directory.
