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
    category="/",
    description="Extract topics from documents using BERTopic.")
@knext.input_table(name="Document table", description="Data table with the document collection to analyze. Each row contains one document.")
@knext.output_table(name="Document table with topics", description="The document collection with topic assignments and the probability for each document to belong to a certain topic.")
@knext.output_table(name="Topic terms", description="The topic models with the terms and their weight per topic.")
class TemplateNode:
    """Use BERTopic to extract topics from documents. 
    TODO Long description of the node.
    """

    # Language in input documents
    language_param = knext.StringParameter(label="Input language", description="", default_value="english")

    # Calculate Probabilities
    probabilities_param = knext.BoolParameter(label="Calculate Probabilities", description="Output probabilities", default_value=False)
    
    # Document column
    document_column_param = knext.ColumnParameter(label="Document column", description="Documents from which topics should be extracted", port_index=0, column_filter=kutil.is_string)

    # TODO Embedding model to use for topic extraction
    embedding_model_param = knext.StringParameter(label="Embedding Model", description="The options to choose from.", default_value="SentenceTransformers", enum=["SentenceTransformers", "Flair", "Spacy", "Gensim"], is_advanced=True)

    # def configure(self, configure_context, input_schema_1):
    def configure(self, configure_context, input_schema_1):  ### Tutorial step 11: Uncomment to configure the new port (and comment out the previous configure header)
        schema1 = input_schema_1.append(knext.Column(knext.int64(), "Topics"))
        # return schema1
        schema2 = knext.Schema([knext.int64(), knext.list_(inner_type=knext.string()), knext.list_(inner_type=knext.double())], ["Topic ID", "Term", "Weight"])
        return schema1, schema2
 
    # def execute(self, exec_context, input_1):
    def execute(self, exec_context, input_1):  ### Tutorial step 11: Uncomment to accept the new port (and comment out the previous execute header)
        input_1_pandas = input_1.to_pandas()
        
        # Compute the topics and output as new column
        topic_model = BERTopic(language=self.language_param, calculate_probabilities=self.probabilities_param, nr_topics=20)
        topics, probs = topic_model.fit_transform(input_1_pandas[self.document_column_param].to_list())
        input_1_pandas['Topics'] = topics

        # Compute the topics and get the most frequent words
        # TODO I am very positive that this can be sped up and/or simplified
        all_topics = topic_model.get_topics()
        all_topics_df = pd.DataFrame(
            [(key, list(t[0] for t in val), list(w[1] for w in val)) for key, val in all_topics.items()],
            columns=['Topic ID', 'Term', 'Weight']
        )
        return knext.Table.from_pandas(input_1_pandas), knext.Table.from_pandas(all_topics_df)
