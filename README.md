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
### Configuration
Configurations are provided via `types.SimpleNamespace` to generate namespaces from Python dictionaries.

- `_common` contains shared hyperparameters for all tasks.
- `_pretrain` contains hyperparameters for pretraining tasks.
- Dictionaries with `_albert_` as a suffix are for specific model configurations
  which can be selected with the `--model` flag in training scripts.
- The exception to the above is `_albert_shared` which is shared hyperparameters
  for all configurations.
- More specific configurations override common configurations

To select a configuration, import `HPS` from `hps.py` then retrieve the
namespace using a `(task, model)` key-tuple.

### Model Pre-training
Start the pre-training script:
```
python main-pretrain.py
```

With optional flags:
```
--cpu           # do not use GPU acceleration, only use the CPU.
--tqdm-off      # disable the use of tqdm loading bars. useful when running on a server.
--small         # use a smaller split of the dataset. useful for debugging.
--model         # select model to use. defaults to `base`. see `hps.py` for details.
--no-save       # do not save outputs to disk. all data will be lost on termination. note that HuggingFace may still cache results.
```
The script will save outputs to `runs/pretrain-{DATE}_{TIME}/` where you can retrieve model checkpoints.

### Model Fine-tuning
`TODO: Instructions on fine-tuning for downstream tasks`

### Inference
`TODO: Instructions on model inference`

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
    - [X] 🤗Version (Faster)
    - [ ] Native (Slower)
- [X] Pre-training Scripts
- [ ] Fine-tuning Scripts
- [ ] Inference Scripts
- [ ] More attention approximation options
- [ ] Fancy Logging
- [ ] Automatic Mixed-Precision Operations
- [ ] Distributed Training

### Code References
- [Nyströmformer self-attention approximation implementation](https://github.com/lucidrains/nystrom-attention) - [@lucidrains](https://github.com/lucidrains)

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
