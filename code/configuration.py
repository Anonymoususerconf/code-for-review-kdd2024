

config = {
    "attention_probs_dropout_prob": 0.1,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "ps_layer_start_idx": 10,
    'initializer_range': 0.02,
    "vocab_size": 21128,
    "max_len_poi": 512,
    "max_len_token": 512,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "hidden_dropout_prob": 0.1,
    "chunk_size_feed_forward": 0,
    "poi_cate_num": 130,  # 128 for categories + 1 for CLS + 1 for PAD
    "image_size": 256,
    "patch_size": 16,
    "num_grid": 256,
    "visual_vocab_size": 8192,
}
