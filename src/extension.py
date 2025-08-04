import logging
import knime.extension as knext
from bertopic import BERTopic
import pandas as pd
from utils import knutils as kutil
from sentence_transformers import SentenceTransformer
from flair.embeddings import TransformerDocumentEmbeddings
import spacy
import gensim.downloader as api
import hdbscan
from sklearn.cluster import KMeans
LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Topic Extractor (BERTopic)", 
    node_type=knext.NodeType.LEARNER, 
    icon_path="icons/icon.png", 
    category="/")
@knext.input_table(name="Input Table", description="Table containing the text column for topic modeling.")
@knext.output_table(name="Document-Topic Probabilities",description="Document-topic distribution with probabilities.")
@knext.output_table(name="Word-Topic Probabilities",description="Topic-word probabilities for each topic.")
@knext.output_table(name="Model Fit Summary",description="Basic statistics and evaluation metrics from model fitting.")
class TemplateNode:
    """Use BERTopic to extract topics from documents. 
    TODO Long description of the node.
    """
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents.",
        port_index=0, 
        column_filter=kutil.is_string
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
    
    def execute(self, exec_context, input_1):
        input_1_pandas = input_1.to_pandas()
        
        # Compute the topics and output as new column
        topic_model = BERTopic(language=self.language_param, calculate_probabilities=self.probabilities_param, nr_topics=20)
        topics, probs = topic_model.fit_transform(input_1_pandas[self.document_column_param].to_list())
        input_1_pandas['Topics'] = topics

        # Compute the topics and get the most frequent words
        # TODO I am very positive that this can be sped up and/or simplified
        all_topics = topic_model.get_topics()
'''
    def execute(self, exec_context, input_table):
            df = input_table.to_pandas()
            docs = df[self.text_column]

            # Load selected embedding backend
            embedder = self._get_embedding_model(self.embedding_model_param)

            # Choose clustering algorithm
            cluster_model = self._get_clustering_model(self.clustering_method, self.nr_topics)

            # Fit BERTopic
            model = BERTopic(embedding_model=embedder, cluster_model=cluster_model, nr_topics=self.nr_topics)
            topics, probs = model.fit_transform(docs.tolist())

            # Output 1: Document-Topic Probabilities with "Topics" column
            df["Topics"] = topics
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

def _get_embedding_model(self, model_type: str):
    if model_type == "SentenceTransformers":
        return SentenceTransformer("all-MiniLM-L6-v2")
    elif model_type == "Flair":
        return TransformerDocumentEmbeddings('bert-base-uncased')
    elif model_type == "Spacy":
        return spacy.load("en_core_web_md")
    elif model_type == "Gensim":
        return api.load("glove-wiki-gigaword-50")
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")

def _get_clustering_model(self, method: str, nr_topics: int):
    if method == "HDBSCAN":
        return hdbscan.HDBSCAN()
    elif method == "KMeans":
        return KMeans(n_clusters=nr_topics if nr_topics > 0 else 10)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
'''