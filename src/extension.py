import logging
import knime.extension as knext
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from flair.embeddings import TransformerDocumentEmbeddings
import gensim.downloader as api
import pandas as pd
import hdbscan
from sklearn.cluster import KMeans
from utils import knutils as kutil

@knext.node(
    name="BERTopic Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/bertopic_learner.png",
    category="/Text/BERT"
)
@knext.input_table("Input Table", "Table containing the text column for topic modeling.")
@knext.output_table("Document-Topic Probabilities", "Document-topic distribution with probabilities.")
@knext.output_table("Word-Topic Probabilities", "Topic-word probabilities for each topic.")
@knext.output_table("Model Fit Summary", "Basic statistics and evaluation metrics from model fitting.")
class BERTopicLearner:
    """Train a BERTopic model using a selected embedding backend."""

    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents.",
        column_filter=lambda col: col.type == knext.string,
        include_none_column=False
    )

    embedding_model_param = knext.StringParameter(
        label="Embedding Model",
        description="Type of embedding model to use for BERTopic.",
        default_value="SentenceTransformers",
        enum=["SentenceTransformers", "Flair", "Spacy", "Gensim"],
        is_advanced=True
    )

    clustering_method = knext.StringParameter(
        label="Clustering Method",
        description="Clustering algorithm to use within BERTopic.",
        default_value="HDBSCAN",
        enum=["HDBSCAN", "KMeans"],
        is_advanced=True
    )

    nr_topics = knext.IntParameter(
        "Number of Topics",
        "Set to -1 to let BERTopic decide, or a fixed number to reduce topics.",
        default_value=-1
    )

    def configure(self, config_context, input_schema):
        schema1 = input_schema.append(knext.Column(knext.int64(), "Topics"))
        schema2 = knext.Schema([
            knext.int64(),
            knext.list_(inner_type=knext.string()),
            knext.list_(inner_type=knext.double())
        ], ["Topic ID", "Term", "Weight"])
        schema3 = knext.Schema([
            knext.int64(),  # Num Topics
            knext.int64(),  # Num Documents
            knext.int64()   # Custom Topic Reduction
        ], ["Num Topics", "Num Documents", "Custom Topic Reduction"])
        return schema1, schema2, schema3

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        docs = df[self.text_column]

        # Load selected embedding backend
        embedder = kutil.get_embedding_model(self.embedding_model_param)

        # Choose clustering algorithm
        cluster_model = kutil.get_clustering_model(self.clustering_method, self.nr_topics)

        # Fit BERTopic
        model = BERTopic(embedding_model=embedder, cluster_model=cluster_model, nr_topics=self.nr_topics)
        topics, probs = model.fit_transform(docs.tolist())

        # Output 1: Document-Topic Probabilities with "Topics" column
        df["Topics"] = topics
        df["Probabilities"] = probs.tolist()
        df = df[[self.text_column, "Topics", "Probabilities"]]
        doc_topic_df = df

        # Output 2: Word-Topic Probabilities
        all_topics = model.get_topics()
        all_topics_df = pd.DataFrame(
            [(key, [w for w, _ in val], [p for _, p in val]) for key, val in all_topics.items()],
            columns=['Topic ID', 'Term', 'Weight']
        )

        # Output 3: Model Fit Summary
        model_fit_df = pd.DataFrame({
            "Num Topics": [len(model.get_topics())],
            "Num Documents": [len(df)],
            "Custom Topic Reduction": [self.nr_topics]
        })

        return (
            knext.Table.from_pandas(doc_topic_df),
            knext.Table.from_pandas(all_topics_df),
            knext.Table.from_pandas(model_fit_df)
        )