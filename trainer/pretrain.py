import torch
import torch.nn.functional as F

from math import ceil

from albert.model import ALBERT
from albert.wrappers import PretrainTransformerWrapper
from helper import get_device

class PretrainTrainer:
    def __init__(self, cfg, args=None):
        self.cfg = cfg
        self.device = get_device(args.cpu if args else False)
        self.net = ALBERT(
            nb_in       = cfg.vocab_size,
            seq_len     = cfg.seq_len,
            mlp_dim     = cfg.mlp_dim,
            emb_dim     = cfg.embedding_dim,
            nb_heads    = cfg.nb_heads,
            head_dim    = cfg.head_dim,
            nb_layers   = cfg.nb_layers,
            nb_seg      = 2,
            dropout     = cfg.dropout,
            norm        = cfg.layer_norm,
            attention_type = cfg.attention_type,
        ).to(self.device)
        self.net = PretrainTransformerWrapper(self.net).to(self.device)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.opt.zero_grad()

        self.scaler = torch.cuda.amp.GradScaler() # for automatic mixed-precision

        self.update_frequency = ceil(cfg.batch_size / cfg.mini_batch_size)
        self.train_steps = 0

    """
        Calculates Masked LM and SOP loss
    """
    @torch.cuda.amp.autocast()
    def _calculate_loss(self, 
            X:              torch.LongTensor, 
            seg:            torch.LongTensor, 
            attention_mask: torch.BoolTensor,
            token_labels:   torch.LongTensor,
            sentence_order: torch.LongTensor,
        ):
        cls_logits, token_logits = self.net(X, seg, mask = attention_mask)

        cls_loss = F.cross_entropy(cls_logits, sentence_order)

        cls_prediction = torch.argmax(cls_logits, dim=-1) # for accuracy calculating, not training
        cls_correct = (cls_prediction == sentence_order).sum()
        cls_total = sentence_order.shape[0]
        cls_accuracy = cls_correct / cls_total

        masked_token_idx = (token_labels >= 0)
        masked_targets = token_labels[masked_token_idx]
        masked_token_logits = token_logits[masked_token_idx]

        token_loss = F.cross_entropy(masked_token_logits, masked_targets)

        masked_token_prediction = torch.argmax(masked_token_logits, dim=-1)
        token_correct = (masked_token_prediction == masked_targets).sum()
        token_total = masked_targets.shape[0]
        token_accuracy = token_correct / token_total

        loss = cls_loss + token_loss

        return loss, (cls_loss, cls_accuracy), (token_loss, token_accuracy)

    """
        Unpack batch and cast to device
        TODO: Feels a tad unnecessary, could just cast before and 
        pass as a tuple
    """
    def _unpack_batch(self, batch):
        batch = [b.to(self.device) for b in batch]
        return batch

    """
        Perform one step of training, accumulate gradients and return metrics
    """
    def train_step(self, batch):
        X, seg, attention_mask, token_labels, sentence_order = self._unpack_batch(batch)
        self.net.train()
        loss, *metrics = self._calculate_loss(
            X, seg, attention_mask.bool(),
            token_labels, sentence_order,
        )
        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()
        return (loss, *metrics)

    """
        Perform one step of evaluation and return metrics
    """
    @torch.no_grad()
    def eval_step(self, batch):
        X, seg, attention_mask, token_labels, sentence_order = self._unpack_batch(batch)
        self.net.eval()
        loss, *metrics = self._calculate_loss(
            X, seg, attention_mask.bool(),
            token_labels, sentence_order,
        )

        return (loss, *metrics)

    """
        Use accumulated gradients to step `self.opt`, updating parameters.
    """
    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

    """
        TODO: proper checkpointing, not just model state_dict
    """
    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)
    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))
