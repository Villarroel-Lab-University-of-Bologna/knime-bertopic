import logging
import knime.extension as knext
from bertopic import BERTopic
import pandas as pd
from utils import knutils as kutil
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