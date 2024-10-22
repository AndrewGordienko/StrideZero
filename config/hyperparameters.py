ACTOR_NETWORK = {
    "FC1_DIMS": 1024,
    "FC2_DIMS": 512,
}

CRITIC_NETWORK = {
    "FC1_DIMS": 1024,
    "FC2_DIMS": 512,
}

AGENT = {
    "ACTOR_LR": 2e-3, 
    "CRITIC_LR": 5e-4,
    "ENTROPY_COEF_INIT": 0.02,
    "ENTROPY_COEF_DECAY": 0.999,
    "GAMMA": 0.995,
    "LAMBDA": 0.95,
    "KL_DIV_THRESHOLD": 0.02,
    "BATCH_SIZE": 64,
    "N_EPOCHS": 10,
    "DEVICE": "cuda"
}

