"""Pipeline to create augmented data and train sequential model for facial expression classification."""

import json

import click
import pandas as pd
import pendulum
import structlog
from keras.src.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models.sequential import Sequential
from keras.src.optimizers import Adam
from keras.src.saving import load_model

from facial_expression_classifier.util.settings import (
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MIN_BATCH_SIZE,
    MIN_EPOCHS,
    ModelSettings,
)
from facial_expression_classifier.util.vizualizations import plot_accuracy, plot_confusion_matrix, plot_loss

structlog.configure(
    cache_logger_on_first_use=True,
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)


def build_model(
    settings: ModelSettings,
) -> Sequential:
    """Setup sequential model with Rectified Linear Unit (ReLU) activation function."""
    model = Sequential(name=settings.model_name)
    model.add(Input(shape=(settings.image_size, settings.image_size, 1)))

    model.add(Conv2D(filters=32, kernel_size=settings.kernel_size, activation=settings.activation))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=settings.kernel_size, activation=settings.activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=settings.pool_size))
    model.add(Dropout(rate=settings.drop_out_rate))

    model.add(Conv2D(filters=128, kernel_size=settings.kernel_size, activation=settings.activation))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=settings.kernel_size, activation=settings.activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=settings.pool_size))
    model.add(Dropout(rate=settings.drop_out_rate))

    model.add(Conv2D(filters=256, kernel_size=settings.kernel_size, activation=settings.activation))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=settings.kernel_size, activation=settings.activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=settings.pool_size))
    model.add(Dropout(rate=settings.drop_out_rate))

    model.add(Flatten())
    model.add(Dense(units=256, activation=settings.activation))
    model.add(BatchNormalization())
    model.add(Dropout(rate=settings.drop_out_rate))
    model.add(Dense(units=7, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
    return model


def init_training_data(
    settings: ModelSettings,
) -> ImageDataGenerator:
    """Setup training dataset with augmentation."""
    # remove prior augmented images
    for to_purge in settings.aug_img_folder.joinpath("train").glob("*.png"):
        to_purge.unlink()
    # setup augmentation parameters
    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rescale=1.0 / 255,
        validation_split=settings.validation_split,
    )
    train_generator = train_datagen.flow_from_directory(
        directory=settings.src_img_folder.joinpath("train"),
        target_size=(settings.image_size, settings.image_size),
        batch_size=settings.batch_size,
        shuffle=True,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training",
        keep_aspect_ratio=False,
        # save_to_dir=settings.aug_img_folder.joinpath("train"),
    )
    log.info(f"training image count: {len(train_generator)}")
    return train_generator


def init_validation_data(
    settings: ModelSettings,
) -> ImageDataGenerator:
    """Setup validation dataset without augmentation."""
    # remove prior augmented images
    for to_purge in settings.aug_img_folder.joinpath("test").glob("*.png"):
        to_purge.unlink()
    # setup augmentation parameters
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=settings.validation_split,
    )
    validation_generator = validation_datagen.flow_from_directory(
        directory=settings.src_img_folder.joinpath("test"),
        target_size=(settings.image_size, settings.image_size),
        batch_size=settings.batch_size,
        shuffle=True,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation",
        keep_aspect_ratio=False,
        # save_to_dir=settings.aug_img_folder.joinpath("test"),
    )
    log.info(f"validation image count: {len(validation_generator)}")
    return validation_generator


def fit_model(
    model: Sequential,
    settings: ModelSettings,
    train_generator: ImageDataGenerator,
    validation_generator: ImageDataGenerator,
):
    """Train model and save results to file."""
    # train model
    trained_model = model.fit(
        x=train_generator,
        epochs=settings.epochs,
        validation_data=validation_generator,
        # callbacks=[checkpoint_callback],
    )
    # save history as '.json' file
    settings.history_file.write_text(data=json.dumps(trained_model.history, indent=2, default=str), encoding="utf-8")
    settings.summary_file.write_text(data=model.to_json(indent=2), encoding="utf-8")
    # save model as '.keras' file
    model.save(settings.model_file, overwrite=True)
    log.info(f"saved: {settings.model_file.name}")


# pylint: disable=no-value-for-parameter
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
    default=True,
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
    print(f"starting: '{settings.model_name}' {train_model=}")
    print(f"settings: {settings.model_dump()}")

    train_generator = init_training_data(settings=settings)
    validation_generator = init_validation_data(settings=settings)

    if train_model:
        model = build_model(settings=settings)
        fit_model(
            model=model,
            settings=settings,
            train_generator=train_generator,
            validation_generator=validation_generator,
        )
    if plot_history:
        # restore model from file
        model: Sequential = load_model(settings.model_file)
        log.info(f"loaded: {settings.model_file.name}")
        # restore training history from file
        history = json.loads(settings.history_file.read_text(encoding="utf-8"))
        df = pd.DataFrame.from_dict(history, orient="columns")
        df.rename(
            columns={
                "loss": "training_loss",
                "val_loss": "validation_loss",
                "accuracy": "training_accuracy",
                "val_accuracy": "validation_accuracy",
            },
            inplace=True,
        )
        plot_confusion_matrix(
            validation_labels=validation_generator.classes,
            validation_predictions=model.predict(validation_generator),
            class_names=settings.labels,
            path=settings.dst_plot_folder.joinpath("confusion_matrix.png"),
        )
        plot_accuracy(df=df, path=settings.dst_plot_folder.joinpath("accuracy.png"))
        plot_loss(df=df, path=settings.dst_plot_folder.joinpath("loss.png"))
    print(f"completed: '{settings.model_name}' {pendulum.now().to_datetime_string()}")


if __name__ == "__main__":
    log = structlog.get_logger()
    trigger_pipeline()
