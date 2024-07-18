from omegaconf import OmegaConf
from pydantic import ValidationError

from python_config_type_safety.train_config import RunningConfig


def load_yaml(file_path):
    yaml_config = OmegaConf.load(file_path)
    return yaml_config


def load_config(yaml_file_path):
    cli_config = OmegaConf.from_cli()
    base_config = load_yaml(yaml_file_path)
    # merge base config and cli config
    config = OmegaConf.merge(base_config, cli_config)
    try:
        config = RunningConfig.model_validate(config)
    except ValidationError as e:
        print(e.json())
        exit(1)
    return config


def main():
    file_path = "src/omegaconf_usage/bad_config.yaml"
    config = load_config(file_path)

    print(config.model.name)

    # OK
    config.train.step = 2000
    # Linter ERROR
    # config.train.step = "2000"

    # OK
    print(config.train.step)
    # ERROR
    # print(config.train.steps)


if __name__ == "__main__":
    main()
