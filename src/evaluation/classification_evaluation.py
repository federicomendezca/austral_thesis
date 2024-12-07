import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from fine_tuning.fine_tuning import PreTrainedAndClassificationHead
from utils.utils import set_seed

warnings.filterwarnings("ignore")


class ClassifierEvaluation:
    def __init__(
        self,
        test: Dataset,
        test_batch_size: int,
        num_labels: int,
        id2label: dict,
        random_state: int = 42,
    ):
        set_seed(random_state)

        self.test_dataloader = DataLoader(
            test, batch_size=test_batch_size, shuffle=False
        )
        self.num_labels = num_labels
        self.id2label = id2label

    def load_model(
        self,
        pretrained_model_name: str,
        model_version: str,
        num_labels: int,
        device: str,
    ) -> None:
        self.model = PreTrainedAndClassificationHead(
            pretrained_model_name=pretrained_model_name, num_labels=num_labels
        )

        pretrained_model_classification_head_dir = (
            f"{pretrained_model_name}_classification_head"
        )
        fine_tuned_model_dir = f"{pretrained_model_name}_fine_tuned"

        if model_version == "Base":
            classifier_checkpoint = torch.load(
                f"{pretrained_model_classification_head_dir}/classification_head.pth"
            )
            self.model.classifier.load_state_dict(classifier_checkpoint)
        else:
            pretrained_model_checkpoint = torch.load(
                f"{fine_tuned_model_dir}/pretrained_model.pth"
            )
            classifier_checkpoint = torch.load(
                f"{fine_tuned_model_dir}/classification_head.pth"
            )

            self.model.pretrained_model.load_state_dict(pretrained_model_checkpoint)
            self.model.classifier.load_state_dict(classifier_checkpoint)

        self.model.to(device)

    def evaluate(self, extract_probabilities=True, extract_layers=True) -> None:
        """Runs forward pass of the standard model, and its classifier version. The classifier only doesn't work, since it doesn't output the pooler layer.

        Args:
            extract_layers (bool, optional): Indicator of whether to output and save encoder's layers. Defaults to True.
        """
        self.model.eval()

        #  Output lists (labels, text and embedding layers)
        all_labels = []
        all_predictions = []
        all_probabilities = []
        text = []

        if extract_layers:
            embeddings_hidden_state_mean_1 = []
            embeddings_hidden_state_mean_2 = []
            embeddings_hidden_state_mean_3 = []

            embeddings_hidden_state_max_1 = []
            embeddings_hidden_state_max_2 = []
            embeddings_hidden_state_max_3 = []

            embeddings_pooler_layer = []

        with torch.no_grad():
            with tqdm(self.test_dataloader, desc="Evaluating", unit="batch") as t:
                for batch in t:
                    inputs = batch["input_ids"]
                    attention_masks = batch["attention_mask"]
                    labels = batch["labels"]

                    if extract_layers:
                        output_model, logits = self.model(
                            inputs,
                            attention_mask=attention_masks,
                            output_hidden_states=True,
                        )
                    else:
                        logits = self.model(
                            inputs,
                            attention_mask=attention_masks,
                            output_hidden_states=True,
                        )

                    if extract_layers:
                        #  Hidden States
                        hidden_states_mean_1 = output_model.hidden_states[-1].mean(
                            dim=1
                        )
                        hidden_states_mean_2 = output_model.hidden_states[-2].mean(
                            dim=1
                        )
                        hidden_states_mean_3 = output_model.hidden_states[-3].mean(
                            dim=1
                        )

                        hidden_states_max_1 = output_model.hidden_states[-1].max(dim=1)[
                            0
                        ]  # max() returns a tuple with the max values and its indices
                        hidden_states_max_2 = output_model.hidden_states[-2].max(dim=1)[
                            0
                        ]
                        hidden_states_max_3 = output_model.hidden_states[-3].max(dim=1)[
                            0
                        ]

                        pooler_layer = output_model.pooler_output

                        #  Extending layer lists
                        embeddings_hidden_state_mean_1.extend(
                            hidden_states_mean_1.cpu().numpy()
                        )
                        embeddings_hidden_state_mean_2.extend(
                            hidden_states_mean_2.cpu().numpy()
                        )
                        embeddings_hidden_state_mean_3.extend(
                            hidden_states_mean_3.cpu().numpy()
                        )

                        embeddings_hidden_state_max_1.extend(
                            hidden_states_max_1.cpu().numpy()
                        )
                        embeddings_hidden_state_max_2.extend(
                            hidden_states_max_2.cpu().numpy()
                        )
                        embeddings_hidden_state_max_3.extend(
                            hidden_states_max_3.cpu().numpy()
                        )

                        embeddings_pooler_layer.extend(pooler_layer.cpu().numpy())

                    if extract_probabilities:
                        probabilities = torch.softmax(logits, dim=1)
                        predicted = logits.argmax(-1)

                        all_predictions.extend(predicted.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())

                    #  Labels and text are included regardless of the type of inference
                    all_labels.extend(labels.cpu().numpy())
                    text.extend(batch["text"])

        if extract_probabilities:
            self.accuracy = accuracy_score(all_labels, all_predictions)
            print(f"Accuracy: {accuracy_score(all_labels, all_predictions)}")
            print(
                f"F1 Score: {f1_score(all_labels, all_predictions, average='weighted')}"
            )

            # Create DataFrame with evaluation results
            self.evaluation_df = pd.DataFrame(
                {
                    "text": text,
                    "y_true": all_labels,
                    "y_pred": all_predictions,
                    "predict_proba": [max(probs) for probs in all_probabilities],
                }
            )

            self.evaluation_df["y_true"] = self.evaluation_df["y_true"].map(
                self.id2label
            )
            self.evaluation_df["y_pred"] = self.evaluation_df["y_pred"].map(
                self.id2label
            )

        if extract_layers:
            # Create dict with embedding dataframes
            self.layer_dfs = {}

            def create_embeddings_df(embeddings, labels):
                df = pd.DataFrame(embeddings)
                df["labels"] = labels
                return df

            self.layer_dfs["Third last hidden state (mean reduction)"] = (
                create_embeddings_df(embeddings_hidden_state_mean_3, labels=all_labels)
            )
            self.layer_dfs["Second last hidden state (mean reduction)"] = (
                create_embeddings_df(embeddings_hidden_state_mean_2, labels=all_labels)
            )
            self.layer_dfs["Last hidden state (mean reduction)"] = create_embeddings_df(
                embeddings_hidden_state_mean_1, labels=all_labels
            )

            self.layer_dfs["Third last hidden state (max reduction)"] = (
                create_embeddings_df(embeddings_hidden_state_max_3, labels=all_labels)
            )
            self.layer_dfs["Second last hidden state (max reduction)"] = (
                create_embeddings_df(embeddings_hidden_state_max_2, labels=all_labels)
            )
            self.layer_dfs["Last hidden state (max reduction)"] = create_embeddings_df(
                embeddings_hidden_state_max_1, labels=all_labels
            )

            self.layer_dfs["Pooler Layer"] = create_embeddings_df(
                embeddings_pooler_layer, labels=all_labels
            )

    def save_layers(self, layers_dir: str = None) -> None:
        if layers_dir:
            #  Creating language family
            language_families = {
                "Romance": ["es", "it", "fr", "pt"],
                "Germanic": ["de", "en", "nl", "sw"],
                "Slavic": ["bg", "pl", "ru"],
                "Afro-Asiatic": ["ar"],
                "Japonic": ["ja"],
                "Sino-Tibetan": ["zh"],
                "Indo-Iranian": ["hi", "ur"],
                "Austroasiatic": ["vi"],
                "Turkic": ["tr"],
                "Tai": ["th"],
                "Hellenic": ["el"],
            }

            language_to_family = {
                lang: family
                for family, languages in language_families.items()
                for lang in languages
            }

            if not os.path.exists(layers_dir):
                os.makedirs(layers_dir)

            for k, v in self.layer_dfs.items():
                v["labels"] = v["labels"].map(self.id2label)  #  Getting labels' names
                v["family"] = v["labels"].map(
                    language_to_family
                )  #  Creating language family variable
                v.to_csv(f"{layers_dir}/{k}.csv", index=False)

    def plot_confusion_matrix(self, model_version, save_dir=None) -> None:
        if save_dir:
            #  Saving classification results df
            self.evaluation_df.to_csv(
                f"{save_dir}/classification_results.csv", sep="|", index=False
            )

            confusion_matrix_array = confusion_matrix(
                self.evaluation_df["y_true"], self.evaluation_df["y_pred"]
            )

            confusion_matrix_df = pd.DataFrame(confusion_matrix_array)
            confusion_matrix_df = confusion_matrix_df.rename(
                columns=self.id2label, index=self.id2label
            )

            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            plt.imshow(
                confusion_matrix_array, interpolation="nearest", cmap=plt.cm.Blues
            )
            plt.title(
                f"Confusion Matrix ({model_version}) - {self.accuracy * 100:.2f}% Accuracy",
                fontsize=22,
            )
            plt.colorbar()

            tick_marks = np.arange(len(confusion_matrix_df.columns))
            plt.xticks(tick_marks, confusion_matrix_df.columns, rotation=45)
            plt.yticks(tick_marks, confusion_matrix_df.index)

            # Add text annotations
            threshold = confusion_matrix_array.max() / 2
            for i, j in np.ndindex(confusion_matrix_array.shape):
                plt.text(
                    j,
                    i,
                    format(confusion_matrix_array[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white"
                    if confusion_matrix_array[i, j] > threshold
                    else "black",
                )

            plt.tight_layout()
            plt.ylabel("Actual Label", fontsize=14)
            plt.xlabel("Predicted Label", fontsize=14)

            if save_dir:
                plt.savefig(f"{save_dir}/Confusion Matrix.png", bbox_inches="tight")

            plt.close("all")

    def classification_report(self, save_dir: str = None) -> None:
        """_summary_

        Args:
            save_dir (str, optional): Output directory. Defaults to None.
        """

        if save_dir:
            classification_report_df = pd.DataFrame(
                classification_report(
                    self.evaluation_df["y_true"],
                    self.evaluation_df["y_pred"],
                    digits=3,
                    output_dict=True,
                )
            ).T
            classification_report_df = classification_report_df.iloc[:20].drop(
                columns="support"
            )
            classification_report_df = classification_report_df.reset_index().rename(
                columns={"index": "language"}
            )

            #  Save classification report table
            classification_report_df.to_csv(
                f"{save_dir}/classification_report.csv", sep="|", index=False
            )
