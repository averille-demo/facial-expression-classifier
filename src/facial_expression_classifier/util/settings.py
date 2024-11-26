"""Pydantic settings facial expressions model."""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import computed_field, field_validator, model_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_BATCH_SIZE: int = 10
MAX_BATCH_SIZE: int = 2048
MIN_EPOCHS: int = 1
MAX_EPOCHS: int = 2048


# pylint: disable=[missing-function-docstring,too-few-public-methods,missing-class-docstring]
class ModelSettings(BaseSettings):
    """Setting for facial expressions model."""

    model_config = SettingsConfigDict(case_sensitive=True)

    debug: bool = False
    version: str
    project_root: Path = Path(__file__).resolve().parent.parent.parent.parent

    src_img_folder: Path = Path(project_root, "data", "orig")
    aug_img_folder: Path = Path(project_root, "data", "augmented")
    dst_plot_folder: Path = Path(project_root, "plots")
    dst_model_folder: Path = Path(project_root, "models")

    epochs: int
    batch_size: int
    image_size: int = 48
    validation_split: float = 0.2
    kernel_size: Tuple[int, int] = (3, 3)
    activation: str = "relu"
    pool_size: Tuple[int, int] = (2, 2)
    drop_out_rate: float = 0.20
    labels: List[str] = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

    @computed_field
    def model_name(self) -> str:
        """Name of model based on project name with versioning."""
        return self.project_root.stem.replace(" ", "_").replace("-", "_")

    @computed_field
    def model_file(self) -> Path:
        """Path to persistent model .keras file."""
        return self.dst_model_folder.joinpath("model_optimal.keras")

    @computed_field
    def history_file(self) -> Path:
        """Path to persistent history .json file."""
        return self.dst_model_folder.joinpath(f"{self.model_name}_history.json")

    @computed_field
    def summary_file(self) -> Path:
        """Path to persistent model summary .json file."""
        return self.dst_model_folder.joinpath(f"{self.model_name}_summary.json")

    @field_validator("epochs", mode="before", check_fields=True)
    def check_epochs_range(cls, v: int) -> int:
        """Validate number of epochs."""
        if v < MIN_EPOCHS or v > MAX_EPOCHS:
            raise ValueError(f"invalid number of epochs: '{v}' must be between {MIN_EPOCHS}-{MAX_EPOCHS}")
        return v

    @field_validator("batch_size", mode="before", check_fields=True)
    def check_batch_size_range(cls, v: int) -> int:
        """Validate batch size."""
        if v < MIN_BATCH_SIZE or v > MAX_BATCH_SIZE:
            raise ValueError(f"invalid batch size: '{v}' must be between {MIN_BATCH_SIZE}-{MAX_BATCH_SIZE}")
        return v

    @field_validator("version", mode="before", check_fields=True)
    def check_version_format(cls, v) -> str:
        if not re.match(r"^(\d+\.)?(\d+\.)?(\*|\d+)$", v):
            raise ValueError(f"invalid model version number: {v}")
        return v

    @field_validator(
        "src_img_folder",
        "aug_img_folder",
        "dst_plot_folder",
        "dst_model_folder",
        mode="before",
        check_fields=True,
    )
    def check_folder_paths(
        cls,
        path,
    ) -> Path:
        """Validate local folder path."""
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Serialize model with selected/ordered fields."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "validation_split": self.validation_split,
            "labels": self.labels,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "pool_size": self.pool_size,
            "drop_out_rate": self.drop_out_rate,
        }
