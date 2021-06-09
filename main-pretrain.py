import torch
import torch.nn.functional

import datasets
import tokenizers
import transformers
import transformers.data.data_collator

import numpy as np

import tqdm
import argparse
import datetime
import functools
from pathlib import Path

log = lambda s: print(f"> {s}") # TODO: Colour!

from hps import HPS
from helper import get_device, get_parameter_count
from trainer.pretrain import PretrainTrainer

"""
    Create workspace and return pointers 
    to the created directories
"""
def create_workspace(save_id):
    runs_dir = Path("runs")
    root_dir = runs_dir / f"pretrain-{save_id}"
    chk_dir = root_dir / "checkpoints"
    log_dir = root_dir / "log_dir"

    runs_dir.mkdir(exist_ok=True)
    root_dir.mkdir(exist_ok=True)
    chk_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    return root_dir, chk_dir, log_dir

"""
    Loading English Wikipedia dataset
    1. Download dataset
    2. Remove title column
    3. Chunk text into smaller sequences
    4. Filter shorter sequences
    5. Create train-test split
"""
def load_dataset(tokenizer, small=False, chunk_size=2000, test_split=.1):
    log("Loading English Wikipedia dataset.")
    wiki = datasets.load_dataset('wikipedia', '20200501.en', split='train[:1%]' if small else 'train')
    wiki.remove_columns("title")
    dataset = wiki

    def _chunk_text(batch, chunk_size=2000):
        chunks = []
        for s in batch['text']:
            chunks += [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]
        return {'chunks': chunks}
    log("Chunking text to maximum sequence_length")
    dataset = dataset.map(functools.partial(_chunk_text, chunk_size=chunk_size), batched=True, num_proc=cfg.nb_workers, remove_columns=dataset.column_names)
    log("Filtering short sequences")
    dataset = dataset.filter(lambda e: len(e['chunks']) >= chunk_size, num_proc=cfg.nb_workers)

    log("Creating train-eval split")
    dataset = dataset.train_test_split(test_size=test_split)

    return dataset

"""
    Load pretrained tokenizer
"""
def load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained('albert-base-v2')
    return tokenizer

"""
    Given a batch, process it into a batch of tensors
    1. Pick a random split
    2. Pick a random ordering
    3. Tokenize sentence pairs
    4. Mask tokens
    5. Cast to tensors

    TODO: Generally, make this more efficient
"""
def process_batch(batch, tokenizer, collator, mask_prob, chunk_size=2000, seq_len=1024):
    chunks = batch['chunks']
    batch_size = len(chunks)
    random_deltas = torch.randint(-chunk_size // 4, chunk_size // 4, (batch_size,))

    mid = chunk_size // 2

    sentence_pairs = [(c[:mid+random_deltas[i]], c[mid+random_deltas[i]:]) for i, c in enumerate(chunks)]
    ordering = torch.randint(0, 2, (batch_size,))

    sentence_pairs = [(c[0],c[1]) if ordering[i] else (c[1], c[0]) for i, c in enumerate(sentence_pairs)]

    tokenized_pairs = tokenizer([s0 for (s0,_) in sentence_pairs], [s1 for (_,s1) in sentence_pairs], 
                        padding='max_length', max_length=seq_len, pad_to_multiple_of=8, truncation=True,
                        return_special_tokens_mask=True, return_tensors='pt')
    masked_input, token_labels = collator.mask_tokens(tokenized_pairs['input_ids'], tokenized_pairs['special_tokens_mask'])

    return masked_input,\
           tokenized_pairs['token_type_ids'],\
           tokenized_pairs['attention_mask'],\
           token_labels,\
           ordering

class EpochMetric:
    def __init__(self):
        self.loss = 0.0

        self.cls_loss = 0.0
        self.cls_accuracy = 0.0

        self.token_loss = 0.0
        self.token_accuracy = 0.0

        self.nb_updates = 0

    def update(self, loss, cls, token):
        self.loss += loss

        self.cls_loss += cls[0]
        self.cls_accuracy += cls[1]

        self.token_loss += token[0]
        self.token_accuracy += token[1]

        self.nb_updates += 1

    def __str__(self):
        s = ""
        s += f"loss: {self.loss / self.nb_updates} "
        s += f"| cls: [loss: {self.cls_loss / self.nb_updates}, accuracy: {100.0 * self.cls_accuracy / self.nb_updates:.2f}%] "
        s += f"| token: [loss: {self.token_loss / self.nb_updates}, accuracy: {100.0 * self.token_accuracy / self.nb_updates:.2f}%]"
        return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--tqdm-off', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    cfg = HPS[('pretrain', args.model)]

    device = get_device(args.cpu)
    log(f"device: {device}")

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not args.no_save:
        root_dir, chk_dir, log_dir = create_workspace(save_id)

    log("Loading pretrained tokenizer")
    tokenizer = load_tokenizer()
    cfg.vocab_size = len(tokenizer) + (8 - len(tokenizer) % 8) # pad to multiple of 8 for tensor core optimisation
    collator = transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=cfg.mask_prob)
    
    log("Loading Wikipedia Dataset")
    # TODO: Convert to dataloader version
    dataset = load_dataset(tokenizer, small=args.small, chunk_size=cfg.chunk_size, test_split=cfg.test_size)

    trainer = PretrainTrainer(cfg, args)

    update_frequency = cfg.batch_size // cfg.mini_batch_size
    ts = 0

    while ts < cfg.nb_updates:
        print()
        epoch_metrics = EpochMetric()
        pb = tqdm.tqdm(range(cfg.nb_train_batches), disable=args.tqdm_off)
        for _ in pb:
            idx = np.random.choice(len(dataset['train']), cfg.batch_size)

            for i in range(update_frequency):
                # TODO: function to collect minibatch
                b = dataset['train'][idx[i*cfg.mini_batch_size:(i+1)*cfg.mini_batch_size]]
                b = process_batch(b, tokenizer, collator, cfg.mask_prob, seq_len=cfg.seq_len)
                metrics = trainer.train_step(b)
                epoch_metrics.update(*metrics)

            if not args.no_save and (ts+1) % cfg.save_frequency == 0:
                trainer.save_checkpoint(chk_dir / f"albert-{args.model}-checkpoint-{str(ts).zfill(7)}.pt")

            ts += 1
            display = f"training | ts: {str(ts).zfill(7)} | {str(epoch_metrics)}"
            pb.set_description(display)
            pb.update(1)

        if args.tqdm_off:
            log(display)

        epoch_metrics = EpochMetric()
        pb = tqdm.tqdm(range(cfg.nb_eval_batches*update_frequency), disable=args.tqdm_off)
        for i in pb:
            idx = np.random.choice(len(dataset['test']), cfg.mini_batch_size)
            b = dataset['test'][idx]
            b = process_batch(b, tokenizer, collator, cfg.mask_prob, seq_len=cfg.seq_len)
            metrics = trainer.eval_step(b)
            epoch_metrics.update(*metrics)

            display = f"evaluation | ts: {str(ts).zfill(7)} | {str(epoch_metrics)}"
            pb.set_description(display)
            pb.update(1)

        if args.tqdm_off:
            log(display)
