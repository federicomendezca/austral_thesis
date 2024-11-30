import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def standardization_and_reduction(
    df: pd.DataFrame,
    reduction: str = "tsne",
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Standardizes using Z-Score and applies dimensionality reduction.

    Args:
        df (pd.DataFrame): Embeddings dataframe.
        reduction (str, optional): Dimensionality reduction to be applied. Defaults to "tsne".
        n_components (int, optional): Output dimensionality. Defaults to 2.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        pd.DataFrame: Low dimensoinality representation of the embeddings, with labels.
    """

    labels_df = df[["labels", "family"]]
    df = df.drop(columns=["labels", "family"])

    standard_scaler = StandardScaler()
    df = standard_scaler.fit_transform(df)

    if reduction == "pca":
        pca = PCA(
            n_components=n_components, svd_solver="full", random_state=random_state
        )
        reduction_output = pca.fit_transform(df)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = round(np.cumsum(explained_variance)[-1] * 100, 2)

        df_reduction = pd.DataFrame(reduction_output)
        df_reduction = pd.concat([df_reduction, labels_df], axis=1)
        return df_reduction, cumulative_variance

    elif reduction == "tsne":
        reduction_output = TSNE(
            n_components=n_components,
            learning_rate="auto",
            init="pca",
            early_exaggeration=4,
            perplexity=50,
            random_state=random_state,
        ).fit_transform(df)

    elif reduction == "umap":
        reduction_output = umap.UMAP(
            n_components=n_components,
            n_neighbors=90,
            min_dist=0.8,
            random_state=random_state,
        ).fit_transform(df)

    df_reduction = pd.DataFrame(reduction_output)
    df_reduction = pd.concat([df_reduction, labels_df], axis=1)
    return df_reduction
