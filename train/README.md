# ICEdit Training Repository

This repository contains the training code for ICEdit, a model for image editing based on text instructions. It utilizes conditional generation to perform instructional image edits.

This codebase is based heavily on the [OminiControl](https://github.com/Yuanshi9815/OminiControl) repository. We thank the authors for their work and contributions to the field!

## Setup and Installation

```bash
# Create a new conda environment
conda create -n train python=3.10
conda activate train

# Install requirements
pip install -r train/requirements.txt
```

## Extra modification for NPU

1. The official version of `pytorch-lightning` doesn't support `NPU` device. We can follow this [PR](https://github.com/Lightning-AI/pytorch-lightning/pull/19308/commits/70397c9ad20b7b1ab5299b91e6d3ceefad47e874) to add `NPU` support.
2. You should set some environment variables before training. Just put the following lines in your `~/.bashrc` file.
```bash
export ASCEND_PATH=/usr/local/Ascend/ascend-toolkit/8.0.RC2  # CANN Installation Path may need to change
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest # may need to change
export PYTHONPATH=${ASCEND_PATH}/python/site-packages:$PYTHONPATH 
export PATH=$ASCEND_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_PATH/lib64:$ASCEND_PATH/acllib/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=$ASCEND_PATH/opp
source /usr/local/Ascend/ascend-toolkit/set_env.sh # may need to change
```
3. Refer to [torch_npu](https://github.com/Ascend/pytorch) for matching the CANN and PyTorch versions.

## Project Structure

- `src/`: Source code directory
  - `train/`: Training modules
    - `train.py`: Main training script
    - `data.py`: Dataset classes for handling different data formats
    - `model.py`: Model definition using Flux pipeline
    - `callbacks.py`: Training callbacks for logging and checkpointing
  - `flux/`: Flux model implementation
- `assets/`: Asset files
- `parquet/`: Parquet data files
- `requirements.txt`: Dependency list

## Datasets

Download training datasets (part of OmniEdit) to the `parquet/` directory. You can use the provided scripts `parquet/prepare.sh`.

```bash
cd parquet
bash prepare.sh
```

## Training

```bash
bash train/script/train.sh # 6 NPUs train

bash train/script/train_single.sh # Single NPU train
```

You can modify the training configuration in `train/config/normal_lora.yaml`. 
