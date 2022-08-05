from easydict import EasyDict

cfg = EasyDict()
cfg.batch_size = 4
cfg.epoch = 500
cfg.learning_rate = 1e-4
cfg.weight_decay = 1e-5
cfg.patience = 25
cfg.inference_threshold = 0.75

cfg.unetr = EasyDict()
cfg.unetr.img_dim = (128, 128, 128)
cfg.unetr.in_channels = 4
cfg.unetr.base_filter = 32
cfg.unetr.class_num = 3
cfg.unetr.patch_size = 16
cfg.unetr.embedding_dim = 768
cfg.unetr.block_num = 12
cfg.unetr.head_num = 12
cfg.unetr.mlp_dim = 3072
cfg.unetr.z_idx_list = [3, 6, 9, 12]
