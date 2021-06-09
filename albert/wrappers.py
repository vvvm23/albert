import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import HelperModule

"""
    Pre-training wrapper for Transformers
    Implements Mask LM (Cloze) and Sentence-order-prediction (SOP)
"""
class PretrainTransformerWrapper(HelperModule):
    def build(self, base: nn.Module):
        self.base = base
        self.cls_mlp = nn.Linear(self.base.mlp_dim, 2)
        self.ve_weight = self.base.in_emb.ve.weight
        self.eh_weight = self.base.in_emb.eh.weight.t()
        self.eh_bias = self.base.in_emb.eh.bias

    def forward(self, x: torch.FloatTensor, seg: torch.LongTensor, mask: torch.BoolTensor):
        attn = self.base(x, seg, mask)
        cls_attn = attn[:, 0]

        cls_logits = self.cls_mlp(cls_attn)
        token_logits = F.linear(F.linear(attn+self.eh_bias, self.eh_weight), self.ve_weight)
        
        return cls_logits, token_logits

"""
    Sentence Level Classification wrapper for Transformers
    Used for multi-label / multi-class classification on the CLR token 
"""
class ClassificationTransformerWrapper(HelperModule):
    def build(self, base: nn.Module, nb_classes: int, dropout: float = 0.2):
        self.base = base
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.base.mlp_dim, nb_classes),
        )

    def forward(self, x: torch.FloatTensor, seg: torch.LongTensor, mask: torch.BoolTensor):
        c = self.base(x, seg, mask)[:, 0] # only grab [CLS] token
        return self.out(c)

"""
    Per-token classification wrapper for Transformers
"""
class TaggingTransformerWrapper(HelperModule):
    def build(self, base: nn.Module, nb_classes: int, dropout: float = 0.2):
        self.base = base
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.base.mlp_dim, nb_classes),
        )

    def forward(self, x: torch.FloatTensor, seg: torch.LongTensor, mask: torch.BoolTensor):
        c = self.base(x, seg, mask) # grab ALL tokens
        return self.out(c) # apply mlp to all tokens

"""
    Span-start and span-end prediction wrapper for Transformers.
    Predicts the start and end tokens of the answer to a 
    question-paragraph tokenized sentence pair.

    TODO: figure out how two weight matrices work to get 
    distribution over all sequence positions.
"""
class QuestionAnsweringTransformerWrapper(HelperModule):
    def build(self, base: nn.Module):
        pass

    def forward(self, x: torch.FloatTensor, seg: torch.LongTensor, mask: torch.BoolTensor):
        pass
