from types import SimpleNamespace

_common = {
    'seed': 123, # not implemented
    'nb_workers': 8,
    'seq_len': 512,
    'chunk_size': 1024,
}

_pretrain = {
    'save_frequency': 5_000,
    'test_size': 0.1,
    'batch_size': 1024,
    'mini_batch_size': 8, # be a multiple of 8
    'learning_rate': 1e-4,
    'mask_prob': .15,
    'nb_updates': 125_000,
    'nb_train_batches': 32,
    'nb_eval_batches': 4,
}

_albert_shared = {
    'embedding_dim':    128,
    'nb_heads':         8,
    'head_dim':         64,
    'layer_norm':       True,
    'attention_type':   'nystrom',
}

_albert_base = {
    'mlp_dim':          768,
    'nb_layers':        6,
    'dropout':          0.1,
}

_albert_large = {
    'mlp_dim':          1024,
    'nb_layers':        24,
    'dropout':          0.1,
}

_albert_xlarge = {
    'mlp_dim':          2048,
    'nb_layers':        24,
    'dropout':          0.0,
}

HPS = {
    ('pretrain', 'base'): SimpleNamespace(**(_common | _pretrain | _albert_shared | _albert_base)),
    ('pretrain', 'large'): SimpleNamespace(**(_common | _pretrain | _albert_shared | _albert_large)),
    ('pretrain', 'xlarge'): SimpleNamespace(**(_common | _pretrain | _albert_shared | _albert_xlarge)),
}
