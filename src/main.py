"""Main module to run analysis.

Arguments allow to select whether to run fine tuning, evaluation and plotting, and which model.
"""

import argparse
import os
from typing import Literal

import torch

from evaluation.classification_evaluation import ClassifierEvaluation
from fine_tuning.dataset_loader import DatasetLoader
from fine_tuning.eda import EDA
from fine_tuning.fine_tuning import PreTrainedAndClassificationHead, train_model
from fine_tuning.tokenize import tokenize
from silhouette.silhouette import ReducionAndSilhouette
from utils.utils import set_seed, setup_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


#####  PREDEFINED PARAMETERS  #####
#  Seed
RANDOM_STATE = 42
set_seed(RANDOM_STATE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#  Training parameters
TOKEN_MAX_LENGTH = 128
EPOCHS = 1
EPOCHS_CLASSIFICATION_HEAD = 5
LR = 1e-4
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 8

#  Select model
RESULTS_DIR = os.path.join(project_root, "results")
PRETRAINED_MODEL = "xlm-roberta-base"
TOKENIZER = PRETRAINED_MODEL

#  HuggingFace Dataset
DF_STR = "papluca/language-identification"

MODELS_TO_RUN: Literal["both", "base", "fine_tuned"] = "both"
#####  PREDEFINED PARAMETERS  #####

# Setting up logger
logger = setup_logger()

FINE_TUNED_MODEL = f"{PRETRAINED_MODEL}_fine_tuned"
CLASSIFICATION_HEAD_DIR = f"{PRETRAINED_MODEL}_classification_head"


def main(
    models_to_run: str = MODELS_TO_RUN,
    hugging_face_df: str = DF_STR,
    eda: bool = True,
    train: bool = False,
    train_classification_head: bool = False,
    train_full_model: bool = False,
    evaluate: bool = True,
    silhouette: bool = True,
    extract_layers: bool = True,
    extract_probabilities: bool = True,
    plot: bool = False,
    save_plots: bool = True,
    pretrained_model_name: str = PRETRAINED_MODEL,
    fine_tuned_model: str = FINE_TUNED_MODEL,
    tokenizer_name: str = TOKENIZER,
    token_max_length: int = TOKEN_MAX_LENGTH,
    epochs_full_model: int = EPOCHS,
    epochs_classification_head: int = EPOCHS_CLASSIFICATION_HEAD,
    lr: float = LR,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    test_batch_size: int = TEST_BATCH_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Main function

    Args:
        models_to_run (str, optional): indicador of which models to run. Defaults to MODELS_TO_RUN.
        hugging_face_df (str, optional): Name of Hugging face dataset to use. Defaults to DF_STR.
        eda (bool, optional): Whether to run EDA. Defaults to True.
        train (bool, optional): Whether to run training. Defaults to False.
        train_classification_head (bool, optional): Whether to train the classification head model. Defaults to False.
        train_full_model (bool, optional): Whether to run full fine tuning. Defaults to False.
        evaluate (bool, optional): Whether to run evaluation. Defaults to True.
        silhouette (bool, optional): Whether to calculate silhouette metrics. Defaults to True.
        extract_layers (bool, optional): Whether to export the model's embeddings. Defaults to True.
        extract_probabilities (bool, optional): Whether to export classification probabilities. Defaults to True.
        plot (bool, optional): Whether to plot figures. Defaults to False.
        save_plots (bool, optional): Whether to save plots locally. Defaults to True.

        pretrained_model_name (str, optional): _description_. Defaults to PRETRAINED_MODEL.
        fine_tuned_model (str, optional): _description_. Defaults to FINE_TUNED_MODEL.
        tokenizer_name (str, optional): _description_. Defaults to TOKENIZER.
        token_max_length (int, optional): _description_. Defaults to TOKEN_MAX_LENGTH.
        epochs_full_model (int, optional): _description_. Defaults to EPOCHS.
        epochs_classification_head (int, optional): _description_. Defaults to EPOCHS_CLASSIFICATION_HEAD.
        lr (float, optional): _description_. Defaults to LR.
        train_batch_size (int, optional): _description_. Defaults to TRAIN_BATCH_SIZE.
        test_batch_size (int, optional): _description_. Defaults to TEST_BATCH_SIZE.
        random_state (int, optional): _description_. Defaults to RANDOM_STATE.
    """
    #  Loading datasets andp preprocessing
    logger.info("---------------------\nLoading Dataset...")
    dataset_loader = DatasetLoader(hugging_face_df=hugging_face_df, device=DEVICE)

    #  Tokenizing and training
    if eda:
        #  Tokenizing with no truncation for EDA purposes
        logger.info(
            "---------------------\nTokenizing Train and Test for EDA (without truncation)..."
        )
        train_eda = tokenize(df=dataset_loader.train, tokenizer_name=tokenizer_name)
        test_eda = tokenize(dataset_loader.test, tokenizer_name=tokenizer_name)

        eda_dir = f"{RESULTS_DIR}/EDA"

        logger.info("---------------------\nRunning EDA plots...")
        eda = EDA(
            train_eda,
            test_eda,
            id2label=dataset_loader.id2label,
            token_max_length=token_max_length,
            eda_dir=eda_dir,
        )
        eda.run_plots(plot=plot, save_plots=save_plots)

    logger.info("---------------------\nTokenizing Test...")
    test = tokenize(
        df=dataset_loader.test,
        tokenizer_name=tokenizer_name,
        token_max_length=token_max_length,
    )

    if train:
        logger.info("---------------------\nTokenizing Train and fine tuning...")
        train = tokenize(
            df=dataset_loader.train,
            tokenizer_name=tokenizer_name,
            token_max_length=token_max_length,
        )

        if train_classification_head:
            logger.info(
                "---------------------\nTraining default model's classification head..."
            )
            model = PreTrainedAndClassificationHead(
                pretrained_model_name=pretrained_model_name,
                num_labels=dataset_loader.num_labels,
            )
            train_model(
                model=model,
                train=train,
                lr=lr,
                num_epochs=epochs_classification_head,
                train_batch_size=train_batch_size,
                save_dir=CLASSIFICATION_HEAD_DIR,
                train_full_model=False,
                device=DEVICE,
                random_state=random_state,
            )

        if train_full_model:
            logger.info("---------------------\nRunning full fine tuning...")
            model = PreTrainedAndClassificationHead(
                pretrained_model_name=pretrained_model_name,
                num_labels=dataset_loader.num_labels,
            )
            train_model(
                model=model,
                train=train,
                lr=lr,
                num_epochs=epochs_full_model,
                train_batch_size=train_batch_size,
                save_dir=fine_tuned_model,
                train_full_model=True,
                device=DEVICE,
                random_state=random_state,
            )

        logger.info("\nTraining completed!")

    #  Evaluating classification and extracting layers
    logger.info("---------------------\nModel testing")

    model_names = (
        [pretrained_model_name, fine_tuned_model]
        if models_to_run == "both"
        else (
            [pretrained_model_name] if models_to_run == "base" else [fine_tuned_model]
        )
    )

    model_versions = (
        ["Base", "Fine Tuned"]
        if models_to_run == "both"
        else (["Base"] if models_to_run == "base" else ["Fine Tuned"])
    )

    for model_name, model_version in zip(model_names, model_versions):
        logger.info(f"\nModel: {model_name}")

        model_results_dir = f"{RESULTS_DIR}/{model_name}"
        layers_dir = f"{model_results_dir}/layers"

        if evaluate:
            logger.info("\n\tLoading base and fine tuned models for evaluation...")

            logger.info("\n\tRunning evaluation...")
            evaluator = ClassifierEvaluation(
                test=test,
                test_batch_size=test_batch_size,
                num_labels=dataset_loader.num_labels,
                id2label=dataset_loader.id2label,
                random_state=random_state,
            )

            evaluator.load_model(
                pretrained_model_name=pretrained_model_name,
                model_version=model_version,
                num_labels=dataset_loader.num_labels,
                device=DEVICE,
            )

            evaluator.evaluate(
                extract_probabilities=extract_probabilities,
                extract_layers=extract_layers,
            )

            if extract_layers:
                logger.info(
                    "\n\tSaving layers and calculating classification metrics..."
                )
                evaluator.save_layers(layers_dir)
                evaluator.plot_confusion_matrix(
                    model_version=model_version, save_dir=model_results_dir
                )
                evaluator.classification_report(save_dir=model_results_dir)

        if silhouette:
            logger.info("\nRunning dimensionality reduction and plotting...")
            reduction_and_silhouette = ReducionAndSilhouette(
                model_version=model_version, model_results_dir=model_results_dir
            )
            reduction_and_silhouette.run_plots(
                plot=plot, save_plots=save_plots, random_state=random_state
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs selected components of the thesis analysis."
    )
    parser.add_argument(
        "--models_to_run",
        type=str,
        default=MODELS_TO_RUN,
        help="Determines which model to run. Possible values: ['both', 'base', 'fine_tuned'],  (default: 'both')",
    )
    parser.add_argument(
        "--eda", action="store_true", help="Whether to run EDA (default: False)"
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to train models (default: False)"
    )
    parser.add_argument(
        "--train_classification_head",
        action="store_true",
        help="Whether to train the defualt model's classificaiton head (default: False)",
    )
    parser.add_argument(
        "--train_full_model",
        action="store_true",
        help="Whether to do full fine tuning (default: False)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to output and save plots locally (default: False)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to run evaluation pipeline (default: False)",
    )
    parser.add_argument(
        "--silhouette",
        action="store_true",
        help="Whether to run silhouette analysis (default: False)",
    )

    args = parser.parse_args()

    main(
        models_to_run=args.models_to_run,
        eda=args.eda,
        train=args.train,
        train_classification_head=args.train_classification_head,
        train_full_model=args.train_full_model,
        plot=args.plot,
        evaluate=args.evaluate,
        silhouette=args.silhouette,
    )
