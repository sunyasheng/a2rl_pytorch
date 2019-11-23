from envs_zoo.crop_env import Environment

def create_crop_env(args, scorer):
    return Environment(args, scorer)
