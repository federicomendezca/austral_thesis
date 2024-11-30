import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


class EDA:
    def __init__(
        self, train, test, id2label: dict, eda_dir: str, token_max_length: int = 128
    ):
        self.train = train
        self.test = test
        self.id2label = id2label
        self.token_max_length = token_max_length
        self.eda_dir = eda_dir

        self.preprocess()

    def preprocess(self):
        """Transforms Hugging Face Dataset to Pandas, maps labels and creates a sequence length variable"""
        self.train = self.train.to_pandas()
        self.train["labels"] = self.train["labels"].map(self.id2label)
        self.train["sequence_length"] = self.train["input_ids"].apply(
            lambda x: x[0].size
        )

        self.test = self.test.to_pandas()
        self.test["labels"] = self.test["labels"].map(self.id2label)
        self.test["sequence_length"] = self.test["input_ids"].apply(lambda x: x[0].size)

    def seq_length_hist(
        self, title: str = "Sequence length distribution per dataset", save: bool = True
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.hist(
            self.train["sequence_length"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax2.hist(
            self.test["sequence_length"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        ax1.set_title("Train", fontsize=16)
        ax2.set_title("Test", fontsize=16)
        ax1.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax2.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)

        plt.suptitle(title, fontsize=22)
        plt.tight_layout()

        if save:
            if not os.path.exists(self.eda_dir):
                os.makedirs(self.eda_dir)
            plt.savefig(f"{self.eda_dir}/{title}.png")

        plt.close("all")

    def seq_length_cdfs(
        self, title: str = "Sequence length CDF per dataset", save: bool = True
    ):
        df_train_sorted = self.train.sort_values(by="sequence_length")
        df_test_sorted = self.test.sort_values(by="sequence_length")

        df_train_sorted["cumulative_percentage"] = (
            (np.arange(len(df_train_sorted)) + 1) / len(df_train_sorted) * 100
        )
        df_test_sorted["cumulative_percentage"] = (
            (np.arange(len(df_test_sorted)) + 1) / len(df_test_sorted) * 100
        )

        # Find cumulative percentage at maximum sequence length
        train_cum_percentage_at_max = df_train_sorted.loc[
            df_train_sorted["sequence_length"] <= self.token_max_length,
            "cumulative_percentage",
        ].max()
        test_cum_percentage_at_max = df_test_sorted.loc[
            df_test_sorted["sequence_length"] <= self.token_max_length,
            "cumulative_percentage",
        ].max()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sns.lineplot(
            x="sequence_length",
            y="cumulative_percentage",
            data=df_train_sorted,
            ax=ax1,
            color="blue",
        )
        ax1.set_title("Train", fontsize=14)
        ax1.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax1.set_ylabel("Cumulative Relative Frequency", fontsize=12)

        # Add vertical line at maximum sequence length
        ax1.axvline(
            x=self.token_max_length,
            color="red",
            linestyle="--",
            label=f"Sequence length = {self.token_max_length}",
        )
        ax1.axhline(
            y=train_cum_percentage_at_max,
            color="green",
            linestyle="--",
            label=f"Cumulative percentage = {train_cum_percentage_at_max:.2f}%",
        )

        # Plot CDF for the test dataset
        sns.lineplot(
            x="sequence_length",
            y="cumulative_percentage",
            data=df_test_sorted,
            ax=ax2,
            color="blue",
        )
        ax2.set_title("Test", fontsize=14)
        ax2.set_xlabel("Sequence Length (tokens)", fontsize=12)

        # Add vertical line at maximum sequence length
        ax2.axvline(
            x=self.token_max_length,
            color="red",
            linestyle="--",
            label="Sequence length = 128",
        )
        ax2.axhline(
            y=test_cum_percentage_at_max,
            color="green",
            linestyle="--",
            label=f"Cumulative percentage = {test_cum_percentage_at_max:.2f}%",
        )

        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")

        plt.suptitle(title, fontsize=22)

        # Adjust layout
        plt.tight_layout()

        if save:
            if not os.path.exists(self.eda_dir):
                os.makedirs(self.eda_dir)
            plt.savefig(f"{self.eda_dir}/{title}.png")

        plt.close("all")

    def boxplots(
        self, df: pd.DataFrame, title: str, plot: bool = False, save: bool = True
    ):
        #  Static color map for both targets to keep consistensy throughout plots
        color_map = {
            "nl": "#2E91E5",
            "es": "#E15F99",
            "it": "#1CA71C",
            "ar": "#FB0D0D",
            "ru": "#DA16FF",
            "tr": "#222A2A",
            "fr": "#B68100",
            "el": "#750D86",
            "pl": "#EB663B",
            "ja": "#511CFB",
            "vi": "#00A08B",
            "th": "#FB00D1",
            "pt": "#FC0080",
            "en": "#B2828D",
            "ur": "#6C7C32",
            "zh": "#778AAE",
            "bg": "#862A16",
            "hi": "#A777F1",
            "sw": "#620042",
            "de": "#1616A7",
        }

        fig = px.box(
            df,
            height=1000,
            width=1600,
            x="labels",
            y="sequence_length",
            category_orders={"labels": sorted(df["labels"].unique())},
            color="labels",
            color_discrete_map=color_map,
            labels={
                "labels": "Language",
                "sequence_length": "Sequence Length (tokens)",
            },
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=28)),
            xaxis_title_font=dict(size=18),  # Update X-axis title font size
            yaxis_title_font=dict(size=18),
            title_x=0.5,
            margin=dict(l=80, r=10, t=60, b=60),
            showlegend=False,
        )

        if plot:
            fig.show()

        if save:
            if not os.path.exists(self.eda_dir):
                os.makedirs(self.eda_dir)
            fig.write_image(f"{self.eda_dir}/{title}.jpeg")
            fig.write_html(f"{self.eda_dir}/{title}.html")

    def run_plots(self, plot: bool = False, save_plots: bool = True):
        self.seq_length_hist(save=save_plots)
        self.seq_length_cdfs(save=save_plots)
        self.boxplots(
            self.train,
            title="Sequence length box plots (train)",
            plot=plot,
            save=save_plots,
        )
        self.boxplots(
            self.test,
            title="Sequence length box plots (test)",
            plot=plot,
            save=save_plots,
        )
