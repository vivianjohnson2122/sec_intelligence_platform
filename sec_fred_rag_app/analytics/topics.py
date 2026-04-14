"""
Lightweight topic modeling for SEC filing sections using TF-IDF + LDA.

We deliberately avoid BERTopic to keep dependencies minimal and inference fast.
sklearn's LatentDirichletAllocation is well-suited for the bag-of-words nature
of regulatory filing language.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)

# Common financial/legal stop words to add on top of sklearn's default list
_EXTRA_STOP_WORDS: list[str] = [
    "company", "companies", "year", "years", "fiscal", "quarter",
    "annual", "period", "ended", "december", "january", "march", "june",
    "september", "item", "form", "sec", "report", "filing", "pursuant",
    "section", "herein", "thereof", "hereby", "thereto", "therein",
    "shall", "may", "including", "following", "certain", "however",
    "also", "approximately", "million", "billion", "thousand",
]


class TopicModeler:
    """
    Fit a TF-IDF + LDA topic model on SEC filing text chunks.

    Usage:
        modeler = TopicModeler(n_topics=8)
        modeler.fit(texts)
        topics = modeler.get_topics(n_words=10)
        dist = modeler.transform(new_texts)
        topic_id, words, prob = modeler.get_dominant_topic(single_text)
    """

    def __init__(
        self,
        n_topics: int = 8,
        max_features: int = 5000,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            n_topics: Number of latent topics to learn.
            max_features: Maximum vocabulary size for TF-IDF.
            random_state: Seed for reproducibility.
        """
        self.n_topics = n_topics
        self._fitted = False

        stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words())
        stop_words += _EXTRA_STOP_WORDS

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=20,
            learning_method="batch",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, texts: list[str]) -> "TopicModeler":
        """
        Fit the TF-IDF vectorizer and LDA model on a list of text strings.

        Args:
            texts: List of filing section text strings.

        Returns:
            self, for method chaining.
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list.")
        logger.info("[Topics] Fitting LDA on %d texts with %d topics…", len(texts), self.n_topics)
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.lda.fit(tfidf_matrix)
        self._fitted = True
        logger.info("[Topics] LDA fit complete. Perplexity: %.1f", self.lda.perplexity(tfidf_matrix))
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TopicModeler has not been fitted. Call fit() first.")

    def get_topics(
        self,
        n_topics: Optional[int] = None,
        n_words: int = 10,
    ) -> list[tuple[int, list[str]]]:
        """
        Return the top words for each topic.

        Args:
            n_topics: Number of topics to return (default: all).
            n_words: Number of top words per topic.

        Returns:
            List of (topic_id, [top_words]) tuples sorted by topic_id.
        """
        self._check_fitted()
        vocab = self.vectorizer.get_feature_names_out()
        n = n_topics or self.n_topics
        result = []
        for topic_id, components in enumerate(self.lda.components_[:n]):
            top_indices = components.argsort()[: -(n_words + 1) : -1]
            words = [vocab[i] for i in top_indices]
            result.append((topic_id, words))
        return result

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Return topic-probability distributions for a list of texts.

        Args:
            texts: List of strings to transform.

        Returns:
            2D numpy array of shape (len(texts), n_topics).
        """
        self._check_fitted()
        tfidf = self.vectorizer.transform(texts)
        return self.lda.transform(tfidf)

    def get_dominant_topic(
        self, text: str
    ) -> tuple[int, list[str], float]:
        """
        Return the single dominant topic for a text string.

        Args:
            text: Input string.

        Returns:
            (topic_id, top_words, probability)
        """
        self._check_fitted()
        dist = self.transform([text])[0]
        topic_id = int(np.argmax(dist))
        probability = float(dist[topic_id])
        _, words = self.get_topics(n_words=10)[topic_id]
        return topic_id, words, probability


def plot_topic_distribution(
    topic_df: pd.DataFrame,
    group_col: str = "ticker",
    title: str = "Topic Distribution by Company",
) -> go.Figure:
    """
    Stacked bar chart of topic proportions per company (or other grouping).

    Args:
        topic_df: DataFrame with columns [group_col, topic_0, topic_1, …, topic_N].
                  Each row represents one document/filing; topic columns hold
                  probability values from TopicModeler.transform().
        group_col: Column to group/aggregate by (default: "ticker").
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    topic_cols = [c for c in topic_df.columns if c.startswith("topic_")]
    if not topic_cols:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="No topic columns found")
        return fig

    grouped = topic_df.groupby(group_col)[topic_cols].mean().reset_index()

    colors = [
        "#6366f1", "#22c55e", "#f59e0b", "#ef4444",
        "#06b6d4", "#a855f7", "#f97316", "#14b8a6",
    ]

    fig = go.Figure()
    for i, col in enumerate(topic_cols):
        topic_num = col.replace("topic_", "")
        fig.add_trace(
            go.Bar(
                name=f"Topic {topic_num}",
                x=grouped[group_col],
                y=grouped[col],
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        barmode="stack",
        xaxis_title=group_col.title(),
        yaxis_title="Average Topic Proportion",
        legend_title="Topic",
        height=450,
    )
    return fig
