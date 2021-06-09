# ALBERT (A Lite BERT)
PyTorch Implementation of "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"

Original paper can be found [here](https://arxiv.org/abs/1909.11942).

This repository will contain a PyTorch implementation of ALBERT and various wrappers around it for pre-training and fine-tuning tasks.

I will use 🤗Tokenizers and 🤗Datasets for tokenization and dataset
preprocessing. A native Python implementation for this simply cannot compete
speed wise. 

## Repository Structure
- `albert/`: directory containing ALBERT architecture implementation and
  associated wrapper modules.
- `trainer/`: directory containing trainer classes for different language model
  tasks.
- `main-TASK.py`: main script to run the task `TASK`.
- `hps.py`: Hyperparameter configuration file.

## Usage
`TBA`

### Data Preparation
`TBA`

### Configuration
`TBA`

### Model Pre-training
`TBA`

### Model Fine-tuning
`TBA`

### Inference
`TBA`

## Modifications
- Option of using Nyströmformer self-attention approximations rather than
  softmax attention. Defaults to Nyströmformer.

## Checkpoints
`TODO: Add model checkpoints`

---

### Roadmap
- [X] ALBERT Core
- [X] BERT alternative option
- [X] Pretraining Wrappers
- [ ] Finetuning Wrappers
- [ ] Preprocessing Pipeline
    - [ ] 🤗Version (Faster)
    - [ ] Native (Slower)
- [ ] Training Scripts
- [ ] Inference Scripts
- [ ] More attention approximation options
- [ ] Fancy Logging
- [ ] Automatic Mixed-Precision Operations
- [ ] Distributed Training

### Citations
**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**
```
@misc{lan2020albert,
      title={ALBERT: A Lite BERT for Self-supervised Learning of Language Representations}, 
      author={Zhenzhong Lan and Mingda Chen and Sebastian Goodman and Kevin Gimpel and Piyush Sharma and Radu Soricut},
      year={2020},
      eprint={1909.11942},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention**
```
@misc{xiong2021nystromformer,
      title={Nystr\"omformer: A Nystr\"om-Based Algorithm for Approximating Self-Attention}, 
      author={Yunyang Xiong and Zhanpeng Zeng and Rudrasis Chakraborty and Mingxing Tan and Glenn Fung and Yin Li and Vikas Singh},
      year={2021},
      eprint={2102.03902},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
