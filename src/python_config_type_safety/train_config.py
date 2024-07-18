from dataclasses import dataclass
from typing import Literal

from omegaconf import MISSING
from pydantic import BaseModel, field_validator

# NOTE: omegaconfはLiteralをサポートしていないため、pydanticのfield_validatorを用いてvalidationを行う
ModelName = Literal[
    "model_name_1",
    "model_name_2",
    "model_name_3",
    "model_name_4",
]

DatasetName = Literal[
    "dataset_name_1",
    "dataset_name_2",
    "dataset_name_3",
    "dataset_name_4",
]


@dataclass
class ModelConfig(BaseModel):
    name: str
    num_classes: int

    @field_validator("name")
    def validate_name(cls, v):
        if v not in ModelName.__args__:
            raise ValueError(f"Invalid model name: {v}")
        return v


@dataclass
class DatasetConfig(BaseModel):
    name: str
    train_batch_size: int

    @field_validator("name")
    def validate_name(cls, v):
        if v not in DatasetName.__args__:
            raise ValueError(f"Invalid dataset name: {v}")
        return v


@dataclass
class TrainConfig(BaseModel):
    # NOTE: 実行時に動的に決定したいがconfigの構造内に入れたい場合はMISSINGにしておく？
    step: int = MISSING
    val_step: int = MISSING
    lr: float


CanUseDeviceId = Literal[0, 1, 2, 3, 4, 5, 6, 7]


@dataclass
class RunningConfig(BaseModel):
    seed: int
    use_device_id: list[int]
    model: ModelConfig
    dataset: DatasetConfig
    train: TrainConfig

    @field_validator("use_device_id")
    def validate_use_device_id(cls, v):
        if not all([i in CanUseDeviceId.__args__ for i in v]):
            raise ValueError(f"Invalid device id: {v}")
        return v
