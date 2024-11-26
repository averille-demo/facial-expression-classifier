"""Pipeline to create augmented data and train sequential model for facial expression classification."""

import click
import pendulum
import structlog
from util.settings import MAX_BATCH_SIZE, MAX_EPOCHS, MIN_BATCH_SIZE, MIN_EPOCHS, ModelSettings

structlog.configure(
    cache_logger_on_first_use=True,
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)


@click.command()
@click.option(
    "--epochs",
    "-e",
    show_default=True,
    default=60,
    help=f"choose number of epochs between {MIN_EPOCHS} and {MAX_EPOCHS}",
)
@click.option(
    "--batch-size",
    "-b",
    show_default=True,
    default=64,
    help=f"choose batch size between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}",
)
@click.option(
    "--train-model",
    "-t",
    is_flag=True,
    show_default=True,
    default=False,
    help="apply boolean flag to train model",
)
@click.option(
    "--plot-history",
    "-p",
    is_flag=True,
    show_default=True,
    default=False,
    help="apply boolean flag to plot model training history",
)
def trigger_pipeline(
    epochs: int,
    batch_size: int,
    train_model: bool,
    plot_history: bool,
):
    """Driver for pipeline."""
    settings = ModelSettings(epochs=epochs, batch_size=batch_size, version="1.0.0")
    print(f"starting: '{settings.model_name}' {train_model=} {plot_history=}")
    print(f"settings: {settings.model_dump()}")

    print(f"completed: '{settings.model_name}' {pendulum.now().to_datetime_string()}")


if __name__ == "__main__":
    log = structlog.get_logger()
    trigger_pipeline()
