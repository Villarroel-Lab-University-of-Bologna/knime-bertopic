import knime.extension as knext
import logging
import pickle

LOGGER = logging.getLogger(__name__)

# pre-define values factory strings
# link to all supported ValueFactoryStrings
# https://github.com/knime/knime-python/blob/49aeaba3819edd635519641c81cc7f9541cf090e/org.knime.python3.arrow.types/plugin.xml

ZONED_DATE_TIME_ZONE_VALUE = "org.knime.core.data.v2.time.ZonedDateTimeValueFactory2"
LOCAL_TIME_VALUE = "org.knime.core.data.v2.time.LocalTimeValueFactory"
LOCAL_DATE_VALUE = "org.knime.core.data.v2.time.LocalDateValueFactory"
LOCAL_DATE_TIME_VALUE = "org.knime.core.data.v2.time.LocalDateTimeValueFactory"

PNG_IMAGE_VALUE = "org.knime.core.data.image.png.PNGImageValueFactory"


def is_zoned_datetime(column: knext.Column) -> bool:
    """
    Checks if date&time column has the timezone or not.
    @return: True if selected date&time column has time zone
    """
    return __is_type_x(column, ZONED_DATE_TIME_ZONE_VALUE)


def is_datetime(column: knext.Column) -> bool:
    """
    Checks if a column is of type Date&Time.
    @return: True if selected column is of type date&time
    """
    return __is_type_x(column, LOCAL_DATE_TIME_VALUE)


def is_time(column: knext.Column) -> bool:
    """
    Checks if a column is of type Time only.
    @return: True if selected column has only time.
    """
    return __is_type_x(column, LOCAL_TIME_VALUE)


def is_date(column: knext.Column) -> bool:
    """
    Checks if a column is of type date only.
    @return: True if selected column has date only.
    """
    return __is_type_x(column, LOCAL_DATE_VALUE)


def boolean_or(*functions):
    """
    Return True if any of the given functions returns True
    @return: True if any of the functions returns True
    """

    def new_function(*args, **kwargs):
        return any(f(*args, **kwargs) for f in functions)

    return new_function


def is_type_timestamp(column: knext.Column):
    """
    This function checks on all the supported timestamp columns in KNIME.
    Note that legacy date&time types are not supported.
    @return: True if timestamp column is compatible with the respective logical types supported in KNIME.
    """

    return boolean_or(is_time, is_date, is_datetime, is_zoned_datetime)(column)


def __is_type_x(column: knext.Column, type: str) -> bool:
    """
    Checks if column contains the given type
    @return: True if Column Type is of that type
    """
    return isinstance(column.ktype, knext.LogicalType) and type in column.ktype.logical_type


def is_string(column: knext.Column) -> bool:
    """
    Check if column is of type string
    @return: True if Column Type is of type string
    """
    return column.ktype == knext.string()


def is_nominal(column: knext.Column) -> bool:
    """
    Check if column is a nominal datatype (string or a boolean)
    @return: True if Column Type is of type string or a boolean
    """
    return column.ktype == knext.string() or column.ktype == knext.bool_()


def is_numeric(column: knext.Column) -> bool:
    """
    Checks if column is numeric e.g. int, long or double.
    @return: True if Column is numeric
    """
    return column.ktype == knext.double() or column.ktype == knext.int32() or column.ktype == knext.int64()


def is_boolean(column: knext.Column) -> bool:
    """
    Checks if column is boolean
    @return: True if Column is boolean
    """
    return column.ktype == knext.boolean()


def is_numeric_or_string(column: knext.Column) -> bool:
    """
    Checks if column is numeric or string
    @return: True if Column is numeric or string
    """
    return boolean_or(is_numeric, is_string)(column)


def is_int_or_string(column: knext.Column) -> bool:
    """
    Checks if column is int or string
    @return: True if Column is numeric or string
    """
    return column.ktype in [
        knext.int32(),
        knext.int64(),
        knext.string(),
    ]


def is_binary(column: knext.Column) -> bool:
    """
    Checks if column is of binary object
    @return: True if Column is binary object
    """
    return column.ktype == knext.blob()


def is_png(column: knext.Column) -> bool:
    """
    Checks if column contains PNG image
    @return: True if Column is image
    """
    return __is_type_x(column, PNG_IMAGE_VALUE)


def check_canceled(exec_context: knext.ExecutionContext) -> None:
    """
    Checks if the user has canceled the execution and if so throws a RuntimeException
    """
    if exec_context.is_canceled():
        raise RuntimeError("Execution canceled")

class BERTopicModelObjectSpec(knext.PortObjectSpec):
    """
    Specification of BERTopic model port.
    """

    def __init__(
        self,
        text_column_name: str,
        n_topics: int,
        embedding_method: str,
        clustering_method: str,
        use_mmr: bool,
    ) -> None:
        self._text_column_name = text_column_name
        self._n_topics = n_topics
        self._embedding_method = embedding_method
        self._clustering_method = clustering_method
        self._use_mmr = use_mmr

    def serialize(self) -> dict:
        return {
            "text_column_name": self._text_column_name,
            "n_topics": self._n_topics,
            "embedding_method": self._embedding_method,
            "clustering_method": self._clustering_method,
            "use_mmr": self._use_mmr,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "BERTopicModelObjectSpec":
        return cls(
            data["text_column_name"],
            data["n_topics"],
            data["embedding_method"],
            data["clustering_method"],
            data["use_mmr"],
        )

    @property
    def text_column_name(self) -> str:
        return self._text_column_name

    @property
    def n_topics(self) -> int:
        return self._n_topics

    @property
    def embedding_method(self) -> str:
        return self._embedding_method

    @property
    def clustering_method(self) -> str:
        return self._clustering_method

    @property
    def use_mmr(self) -> bool:
        return self._use_mmr


class BERTopicModelObject(knext.PortObject):
    """
    Port object containing the trained BERTopic model.
    """

    def __init__(
        self,
        spec: BERTopicModelObjectSpec,
        model,
        documents: list = None,
    ) -> None:
        super().__init__(spec)
        self._model = model
        self._documents = documents  # Optional: store training documents for reference

    def serialize(self) -> bytes:
        """
        Serialize the BERTopic model and associated data.
        """
        return pickle.dumps(
            {
                "model": self._model,
                "documents": self._documents,
            }
        )

    @property
    def spec(self) -> BERTopicModelObjectSpec:
        return super().spec

    @property
    def model(self):
        """Returns the BERTopic model instance."""
        return self._model

    @property
    def documents(self) -> list:
        """Returns the training documents if stored."""
        return self._documents

    @classmethod
    def deserialize(
        cls, spec: BERTopicModelObjectSpec, data: bytes
    ) -> "BERTopicModelObject":
        """
        Deserialize the BERTopic model and associated data.
        """
        deserialized_data = pickle.loads(data)
        return cls(
            spec,
            deserialized_data["model"],
            deserialized_data.get("documents"),
        )

    def transform_documents(self, documents: list):
        """
        Transform new documents using the trained model.
        Returns topics and probabilities.
        """
        if self._model is None:
            raise ValueError("Model is not set.")

        topics, probabilities = self._model.transform(documents)
        return topics, probabilities

    def get_topic_info(self):
        """
        Get information about all topics in the model.
        """
        if self._model is None:
            raise ValueError("Model is not set.")

        return self._model.get_topic_info()

    def get_topics(self):
        """
        Get all topics with their word representations.
        """
        if self._model is None:
            raise ValueError("Model is not set.")

        return self._model.get_topics()

    def get_topic(self, topic_id: int):
        """
        Get a specific topic's word representation.
        """
        if self._model is None:
            raise ValueError("Model is not set.")

        return self._model.get_topic(topic_id)


# Define the BERTopic model port type
bertopic_model_port_type = knext.port_type(
    name="BERTopic Model",
    object_class=BERTopicModelObject,
    spec_class=BERTopicModelObjectSpec,
)
