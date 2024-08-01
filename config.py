class Config:
    MAX_SEQ = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEADS = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1
    RATIO = 0.4
    DROPOUT_RATE = 0.2 

def get_config():
    config = Config()
    return config 

def set_config_ratio(config, ratio):
    config.RATIO = ratio 
    return config 

