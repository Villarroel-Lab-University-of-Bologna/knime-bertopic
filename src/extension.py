import logging
import knime.extension as knext
import pandas as pd

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Simple Text Processor", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path="icons/icon.png", 
    category="/")
@knext.input_table(name="Input Table", description="Table containing text data.")
@knext.output_table(name="Output Table", description="Table with processed text.")
class SimpleTextNode:
    """A simple text processing node for testing."""
    
    text_column = knext.ColumnParameter(
        "Text Column",
        "Select the column containing text data.",
        port_index=0,
        column_filter=lambda col: col.ktype == knext.string()
    )

    def configure(self, config_context, input_schema):
        # Return the same schema with an additional column
        return input_schema.append(knext.Column(knext.string(), "Processed_Text"))

    def execute(self, exec_context, input_table):
        # Convert to pandas
        df = input_table.to_pandas()
        
        # Simple text processing - add length
        df['Processed_Text'] = df[self.text_column].astype(str) + " [Length: " + df[self.text_column].astype(str).str.len().astype(str) + "]"
        
        # Convert back to KNIME table
        return knext.Table.from_pandas(df)