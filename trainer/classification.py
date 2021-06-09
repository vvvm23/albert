import torch
import torch.nn.functional as F

from math import ceil

from albert.wrappers import ClassificationTransformerWrapper
from helper import get_device

"""
    Trainer for multi-label, sentence-level classification
"""
class MultiLabelTrainer:
    def __init__(self, base: torch.nn.Module, cfg, args=None):
        self.cfg = cfg
        self.device = get_device(args.cpu if args else False)
        self.net = ClassificationTransformerWrapper(base.to(self.device), cfg.nb_classes).to(self.device)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.opt.zero_grad()

        self.scaler = torch.cuda.amp.GradScaler() # for automatic mixed-precision

        self.update_frequency = ceil(cfg.batch_size / cfg.mini_batch_size)
        self.train_steps = 0

    """
        Calculate class prediction loss
        TODO: Some other metrics aside from accuracy? Not the best
    """
    @torch.cuda.amp.autocast()
    def _calculate_loss(self,
            X:              torch.LongTensor, 
            seg:            torch.LongTensor, 
            attention_mask: torch.BoolTensor,
            targets:        torch.FloatTensor,
            weights:        torch.FloatTensor   = None,
        ):
        class_logits = self.net(X, seg, mask = attention_mask)
        class_loss = F.binary_cross_entropy_with_logits(class_logits, targets, pos_weight=weights)

        class_prediction = torch.sigmoid(class_logits) > .5
        class_correct = (class_prediction == (targets > .5)).sum()
        class_accuracy = class_correct / targets.numel()

        return class_loss, class_accuracy

    """
        Perform one step of training, accumulate gradients and return metrics
    """
    def train_step(self,
            X:              torch.LongTensor, 
            seg:            torch.LongTensor, 
            attention_mask: torch.BoolTensor,
            targets:        torch.FloatTensor,
            weights:        torch.FloatTensor   = None,
        ):
        self.net.train()
        loss, class_accuracy = self._calculate_loss(
            X, seg,
            attention_mask, targets,
            weights = weights,
        )
        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()
        return loss.item(), class_accuracy.item()

    """
        Perform one step of evaluation and return metrics
    """
    @torch.no_grad()
    def eval_step(self,
            X:              torch.LongTensor, 
            seg:            torch.LongTensor, 
            attention_mask: torch.BoolTensor,
            targets:        torch.FloatTensor,
            weights:        torch.FloatTensor   = None,
        ):
        self.net.eval()
        loss, class_accuracy = self._calculate_loss(
            X, seg,
            attention_mask, targets,
            weights = weights,
        )
        return loss.item(), class_accuracy.item()

    """
        Use accumulated gradients to step `self.opt`, updating parameters.
    """
    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)
    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))

"""
    Trainer for multi-class, sentence-level classification
"""
class MultiClassTrainer:
    def __init__(self, base: torch.nn.Module, cfg, args=None):
        self.cfg = cfg
        self.device = get_device(args.cpu if args else False)
        self.net = ClassificationTransformerWrapper(base.to(self.device), cfg.nb_classes).to(self.device)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.opt.zero_grad()

        self.scaler = torch.cuda.amp.GradScaler() # for automatic mixed-precision

        self.update_frequency = ceil(cfg.batch_size / cfg.mini_batch_size)
        self.train_steps = 0

    """
        Calculate class prediction loss
        TODO: Some other metrics aside from accuracy? Not the best
    """
    @torch.cuda.amp.autocast()
    def _calculate_loss(self,
            X:              torch.LongTensor, 
            seg:            torch.LongTensor, 
            attention_mask: torch.BoolTensor,
            targets:        torch.LongTensor,
            weights:        torch.FloatTensor   = None,
        ):
        class_logits = self.net(X, seg, mask = attention_mask)
        class_loss = F.nll_loss(class_logits, targets, pos_weight=weights)

        class_prediction = class_logits.argmax(dim=-1)

        class_correct = (class_prediction == targets).sum()
        class_accuracy = class_correct / targets.numel()

        return class_loss, class_accuracy

    """
        Perform one step of training, accumulate gradients and return metrics
    """
    def train_step(self,
            X:              torch.LongTensor, 
            seg:            torch.LongTensor, 
            attention_mask: torch.BoolTensor,
            targets:        torch.FloatTensor,
            weights:        torch.FloatTensor   = None,
        ):
        self.net.train()
        loss, class_accuracy = self._calculate_loss(
            X, seg,
            attention_mask, targets,
            weights = weights,
        )
        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()
        return loss.item(), class_accuracy.item()

    """
        Perform one step of evaluation and return metrics
    """
    @torch.no_grad()
    def eval_step(self,
            X:              torch.LongTensor, 
            seg:            torch.LongTensor, 
            attention_mask: torch.BoolTensor,
            targets:        torch.FloatTensor,
            weights:        torch.FloatTensor   = None,
        ):
        self.net.eval()
        loss, class_accuracy = self._calculate_loss(
            X, seg,
            attention_mask, targets,
            weights = weights,
        )
        return loss.item(), class_accuracy.item()

    """
        Use accumulated gradients to step `self.opt`, updating parameters.
    """
    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)
    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))
