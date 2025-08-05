import logging
import knime.extension as knext
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
import utils.knutils as kutil

LOGGER = logging.getLogger(__name__)

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError as e:
    BERTOPIC_AVAILABLE = False
    print(f"BERTopic import failed: {e}")

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Topic Extractor (BERTopic)", 
    node_type=knext.NodeType.LEARNER, 
    icon_path="icons/icon.png", 
    category="/")

@knext.input_table(name="Input Table", description="Table containing the text column for topic modeling.")
@knext.output_table(name="Document-Topic Probabilities", description="Document-topic distribution with probabilities.")
@knext.output_table(name="Word-Topic Probabilities", description="Topic-word probabilities for each topic.")
@knext.output_table(name="Model Fit Summary", description="Basic statistics and evaluation metrics from model fitting.")
class BERTopicNode:
    """Use BERTopic to extract topics from documents."""
    
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents.",
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
    
    language_param = knext.StringParameter(
        label="Language",
        description="Language for topic modeling.",
        default_value="english",
        enum=["english", "multilingual"],
        is_advanced=True
    )
    
    probabilities_param = knext.BoolParameter(
        label="Calculate Probabilities",
        description="Whether to calculate topic probabilities for documents.",
        default_value=True,
        is_advanced=True
    )
    
    nr_topics_param = knext.IntParameter(
        label="Number of Topics",
        description="Number of topics to extract. Use 'auto' for automatic selection.",
        default_value=20,
        is_advanced=True
    )
    def configure(self, config_context, input_schema):
        # Simple output: input + topic column
        return input_schema.append(knext.Column(knext.int64(), "Topic"))

    def execute(self, exec_context, input_table):
        if not BERTOPIC_AVAILABLE:
            raise RuntimeError("BERTopic is not available. Please check installation.")
            
        # Convert to pandas
        df = input_table.to_pandas()
        
        # Get documents
        documents = df[self.text_column].dropna().astype(str).tolist()
        
        if not documents:
            raise ValueError("No valid documents found.")
        
        # Simple BERTopic usage with defaults
        topic_model = BERTopic()
        topics, _ = topic_model.fit_transform(documents)
        
        # Add topics back to dataframe
        df['Topic'] = -1  # Default for all rows
        valid_indices = df[self.text_column].dropna().index
        df.loc[valid_indices, 'Topic'] = topics
        

        return knext.Table.from_pandas(df)