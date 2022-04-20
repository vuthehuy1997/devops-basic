# from omegaconf import OmegaConf
# # loading
# config = OmegaConf.load('./week_2_hydra_config/configs/config_learn.yaml')

# # accessing
# print(config.preferences.user)
# print(config["preferences"]["trait"])

import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="./configs", config_name="config_learn.yaml")
def main(cfg):

    print(cfg)
    # Print the config file using `to_yaml` method which prints in a pretty manner
    print(OmegaConf.to_yaml(cfg))
    print(cfg.preferences.user)

if __name__ == "__main__":
    main()