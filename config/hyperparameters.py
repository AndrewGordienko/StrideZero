ACTOR_NETWORK = {
    "FC1_DIMS": 1024,
    "FC2_DIMS": 512,
}

CRITIC_NETWORK = {
    "FC1_DIMS": 1024,
    "FC2_DIMS": 512,
}

AGENT = {
    "ACTOR_LR": 5e-4, 
    "CRITIC_LR": 1e-3,
    "ENTROPY_COEF_INIT": 0.03,
    "ENTROPY_COEF_DECAY": 0.9995,
    "GAMMA": 0.995,
    "LAMBDA": 0.97,
    "KL_DIV_THRESHOLD": 0.02,
    "BATCH_SIZE": 64,
    "N_EPOCHS": 10,
    "DEVICE": "cuda"
}

