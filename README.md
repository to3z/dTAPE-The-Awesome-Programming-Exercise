---

# Improvements on Deterministic TAPE for Heterogeneous MARL

- **Course Project**: Reinforcement Learning (Fall 2025), Peking University  
- **Topic**: Research and Improvement of dTAPE
- **Team Memebers**: Lian Hanchun (连含春), Li Shuhua (李抒桦)

## Introduction

This repository contains the implementation and experimental results for the Reinforcement Learning course project at Peking University. Our work focuses on the **Deterministic TAPE (dTAPE)** algorithm from the AAAI 2024 paper *"TAPE: Leveraging Agent Topology for Cooperative Multi-Agent Policy Gradient"*.

Specifically, we address the credit assignment failure in highly heterogeneous scenarios (e.g., `wzsy`, `swct`). We propose an **Attentional Mixer** to replace the original aggregation mechanism. Our experimental results demonstrate that this improvement significantly enhances the "Diversionary Tactic" (e.g., in `wzsy`) and achieves stable win rates in complex tactical maps.

## Codebase & Acknowledgements

This repository is built upon the following open-source projects:

1.  **[TAPE](https://github.com/LxzGordon/TAPE)**: Leveraging Agent Topology for Cooperative Multi-Agent Policy Gradient.



## Environment

'smac' package has been uploaded on https://course.pku.edu.cn/

## How to Run

### 1. Training

**For Deterministic TAPE (Baseline & Ours):**

```bash
# choose one of the 12 maps, here 'adcc' as an example

# Baseline
python src/main.py --config=d_tape --env-config=sc2te with env_args.map_name=adcc_te

# With improved mixer
python src/main.py --config=lsh_d_tape --env-config=sc2te with env_args.map_name=adcc_te

# With attention-based agent topology
python src/main.py --config=d_tape_atten --env-config=sc2te with env_args.map_name=adcc_te

# Both
python src/main.py --config=amalgam --env-config=sc2te with env_args.map_name=adcc_te

```

*Note: Ensure you have StarCraft II and SMAC installed and the maps are placed in the correct directory.*

### 2. Evaluation

To evaluate a trained model and generate replays for visualization.

**Evaluate a saved model:**
You need to point `checkpoint_path` to the directory where your models are saved.

```bash
# Evaluate model
python src/evaluate.py --config=d_tape_atten --env-config=sc2te with env_args.map_name=adcc_te --checkpoint_path=...
```
