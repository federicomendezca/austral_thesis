import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_samples, silhouette_score

from silhouette.dimensionality_reduction import standardization_and_reduction
from utils.utils import setup_logger

logger = setup_logger()

warnings.filterwarnings("ignore")


class ReducionAndSilhouette:
    def __init__(self, model_version: str, model_results_dir: str):
        self.model_results_dir = model_results_dir
        self.projections_dir = f"{model_results_dir}/projections"
        self.model_version = model_version

        layers_dir = f"{model_results_dir}/layers"
        self.dfs = {}
        for file in os.listdir(layers_dir):
            self.dfs[file] = pd.read_csv(f"{layers_dir}/{file}")
            # self.languages = np.unique(dfs[file]["labels"])  # Get unique languages

    @staticmethod
    def plot_scatter(
        df,
        labels: str,
        color_map: dict,
        title="",
        symbol: bool = None,
        legend_title: bool = None,
        save_dir: bool = None,
    ):
        #  Static symbol sequence for consistency throughout plots
        symbol_sequence = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-down",
            "star",
            "hexagram",
            "star-triangle-up",
        ]
        symbol = df[symbol] if symbol else None

        scatter_fig = px.scatter(
            x=df[0],
            y=df[1],
            color=df[labels],
            color_discrete_map=color_map,
            labels=dict(x="Component 1", y="Component 2"),
            symbol=symbol,
            symbol_sequence=symbol_sequence,
        )

        scatter_fig.update_layout(
            height=1000,
            width=1600,
            margin=dict(l=10, r=10, t=80, b=80),
            paper_bgcolor="white",
            title=dict(text=title, font=dict(size=24)),
            title_x=0.5,
            legend=dict(y=0.5),
            legend_title_text=legend_title,
            font=dict(size=14),
        )

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            scatter_fig.write_image(f"{save_dir}/{title}.jpeg")
            scatter_fig.write_html(f"{save_dir}/{title}.html")

        return scatter_fig

    @staticmethod
    def plot_silhouette(
        df: pd.DataFrame,
        labels: str,
        color_map: dict,
        title: str = None,
        metric: str = "euclidean",
        reverse_labels: bool = True,
        legend: bool = False,
        legend_title: str = "Language",
        random_state: int = 42,
        save_dir: bool = None,
    ):
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(
            X=df.drop(columns=["labels", "family"]),
            labels=df[labels],
            metric=metric,
            random_state=random_state,
        )
        sample_silhouette_values = silhouette_samples(
            X=df.drop(columns=["labels", "family"]), labels=df[labels], metric=metric
        )  # silhouette_samples has no random state

        unique_labels = sorted(
            np.unique(df[labels]), reverse=reverse_labels
        )  #  Descending order, to replicate labels' ordering in the silhouette plots
        y_lower = 10  #  Small gap between traces
        traces = []

        for i, label in enumerate(unique_labels):
            # Aggregate the silhouette scores for samples belonging to the same cluster
            ith_cluster_silhouette_values = sample_silhouette_values[
                df[labels] == label
            ]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Create a trace for the silhouette values of the current cluster
            trace = go.Scatter(
                x=ith_cluster_silhouette_values,
                y=np.arange(y_lower, y_upper),
                mode="lines",
                line=dict(width=1),
                fill="tozerox",
                name=f"{label}",
                marker=dict(color=color_map[label]),
                legendgroup=label,
                showlegend=legend,
            )
            traces.append(trace)

            # Label the silhouette plots with their cluster numbers in the middle
            y_lower = y_upper + 10  # 10 for the 0 samples

        # Create silhouette plot layout
        silhouette_layout = go.Layout(
            height=1000,
            width=1600,
            xaxis=dict(title="Silhouette Coefficient Values"),
            yaxis=dict(title="Document Index"),
            title=dict(text=title, font=dict(size=24)),
            title_x=0.5,
            legend_title_text=legend_title,
            legend=dict(y=0.5),
            showlegend=legend,
            font=dict(size=14),
        )

        # Add a red vertical line for average silhouette score
        silhouette_avg_trace = go.Scatter(
            x=[silhouette_avg, silhouette_avg],
            y=[0, y_lower],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Mean Silhouette Score",
        )
        traces.append(silhouette_avg_trace)
        silhouette_fig = go.Figure(data=traces)

        #  Adjusting the position of the Mean silhouette Score legend
        silhouette_traces: list = []
        last_trace: list = []
        for trace in silhouette_fig.data:
            if trace.name == "Mean Silhouette Score":
                last_trace = trace
            else:
                silhouette_traces.append(trace)

        silhouette_traces.sort(key=lambda x: x.name)

        # Reconstruct the data list with the last trace at the end
        sorted_traces = silhouette_traces + [last_trace]
        silhouette_fig = go.Figure(data=sorted_traces, layout=silhouette_layout)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            silhouette_fig.write_image(f"{save_dir}/{title}.jpeg")
            silhouette_fig.write_html(f"{save_dir}/{title}.html")

        return silhouette_fig

    @staticmethod
    def plot_scatter_silhouette(
        scatter_fig,
        silhouette_fig,
        title: str = "",
        legend_title: str = "Language Family",
        plot: bool = False,
        save_dir: bool = None,
    ):
        # Combine both plots into a single figure with two subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Scatterplot", "Silhouette Diagram"),
            vertical_spacing=0.15,
        )

        # Add scatter plot to the first row
        for scatter_trace in scatter_fig.data:
            fig.add_trace(scatter_trace, row=1, col=1)

        # Add silhouette plots to the second row
        for trace in silhouette_fig.data:
            fig.add_trace(trace, row=2, col=1)

        # Update layout for the first row
        fig.update_xaxes(title_text=scatter_fig.layout.xaxis.title.text, row=1, col=1)
        fig.update_yaxes(title_text=scatter_fig.layout.yaxis.title.text, row=1, col=1)

        # Update layout for the second row
        fig.update_xaxes(
            title_text=silhouette_fig.layout.xaxis.title.text, row=2, col=1
        )
        fig.update_yaxes(
            title_text=silhouette_fig.layout.yaxis.title.text, row=2, col=1
        )

        # Update overall title and legend

        fig.update_layout(
            height=1000,
            width=1600,
            margin=dict(l=50, r=50, t=120, b=50),
            title=dict(text=title, font=dict(size=24)),
            coloraxis_showscale=False,
            showlegend=True,
            title_x=0.5,
            legend=dict(y=0.5),
            legend_title_text=legend_title,
            font=dict(size=14),
        )

        fig.update_annotations(
            dict(font=dict(size=18), yshift=3)  # Adjust y-shift for subplot titles
        )

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.write_image(f"{save_dir}/{title}.jpeg")
            fig.write_html(f"{save_dir}/{title}.html")

        if plot:
            fig.show()

    def plot_and_save_reduction(
        self,
        df: pd.DataFrame,
        title: str,
        reduction: str = "pca",
        plot: bool = False,
        save: bool = False,
        random_state: int = 42,
    ):
        reduction_dir = f"{self.projections_dir}/{reduction}" if save else None
        silhouette_title = (
            f"{self.model_version} - {title} - High Dim - Silhouette Diagram"
        )

        #  PCA also outputs total variance captured by first 2 principal components
        if reduction == "pca":
            df_reduction, variance = standardization_and_reduction(
                df, reduction=reduction, random_state=random_state
            )
            scatter_title = f"{self.model_version} - {title} - {reduction.upper()} ({variance}% total variance)"
        else:
            df_reduction = standardization_and_reduction(
                df, reduction=reduction, random_state=random_state
            )
            scatter_title = f"{self.model_version} - {title} - {reduction.upper()}"

        #  Saving df
        if save:
            if not os.path.exists(reduction_dir):
                os.makedirs(reduction_dir)
            df_reduction.to_csv(
                f"{reduction_dir}/{self.model_version} - {title} - {reduction.upper()}.csv",
                index=False,
            )

        #  Plotting and saving outputs
        for labels, symbol in zip(
            ["labels", "family"], ["family", "labels"]
        ):  #  Alternating language and family
            #  Sorting to get labels and silhouette traces in alphabetical order
            df_reduction = df_reduction.sort_values(labels)

            high_dim_silhouette_dir = (
                f"{self.model_results_dir}/silhouette/{labels}" if save else None
            )
            reduction_silhouette_dir = (
                f"{reduction_dir}/silhouette/{labels}" if reduction_dir else None
            )
            scatter_dir = f"{reduction_dir}/scatter" if reduction_dir else None
            legend_title = "Language" if labels == "labels" else "Language family"

            #  Static color map for both targets to keep consistensy throughout plots
            if labels == "labels":
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

            else:
                color_map = {
                    "Germanic": "#2E91E5",
                    "Romance": "#E15F99",
                    "Afro-Asiatic": "#1CA71C",
                    "Slavic": "#FB0D0D",
                    "Turkic": "#DA16FF",
                    "Hellenic": "#222A2A",
                    "Japonic": "#B68100",
                    "Austroasiatic": "#750D86",
                    "Tai": "#EB663B",
                    "Indo-Iranian": "#511CFB",
                    "Sino-Tibetan": "#00A08B",
                }

            #  Two scatter figures (one with labels and family to save independently, and one with only labels to pair with the silhouette diagram)
            scatter_fig_save = ReducionAndSilhouette.plot_scatter(
                df_reduction,
                labels=labels,
                color_map=color_map,
                symbol=symbol,
                legend_title="Family, language",
                title=scatter_title,
                save_dir=scatter_dir,
            )
            scatter_fig = ReducionAndSilhouette.plot_scatter(
                df_reduction,
                labels=labels,
                color_map=color_map,
                symbol=None,
                save_dir=None,
            )

            #  1) Plot reduction's scatterplot and silhouette diagrams
            reduction_silhouette_fig = ReducionAndSilhouette.plot_silhouette(
                df_reduction,
                labels=labels,
                color_map=color_map,
                random_state=random_state,
            )

            ReducionAndSilhouette.plot_scatter_silhouette(
                scatter_fig=scatter_fig,
                silhouette_fig=reduction_silhouette_fig,
                title=f"{scatter_title} (by {legend_title})",
                legend_title=f"{legend_title}",
                plot=plot,
                save_dir=reduction_silhouette_dir,
            )

            #  2) Plot original embeddings' silhouette diagrams
            ReducionAndSilhouette.plot_silhouette(
                df,
                labels=labels,
                color_map=color_map,
                reverse_labels=True,
                metric="cosine",  #  Using cosine similarity for high dimensionality distances
                title=f"{silhouette_title} (by {legend_title})",
                legend=True,
                legend_title=legend_title,
                save_dir=high_dim_silhouette_dir,
                random_state=random_state,
            )

    def run_plots(
        self, plot: bool = False, save_plots: bool = True, random_state: int = 42
    ):
        for key, df in self.dfs.items():
            title = key[:-4]  #  Removing ".csv" from file name

            logger.info(f"\n-----------------\nProcessing {title}\n")

            for reduction in ["pca", "tsne", "umap"]:
                logger.info(f"\tCurrent reduction: {reduction}")
                self.plot_and_save_reduction(
                    df=df,
                    title=title,
                    reduction=reduction,
                    plot=plot,
                    save=save_plots,
                    random_state=random_state,
                )
