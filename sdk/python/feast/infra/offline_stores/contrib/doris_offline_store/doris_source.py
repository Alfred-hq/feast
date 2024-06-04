import json
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
from feast.data_source import DataSource
from feast.errors import DataSourceNoNameException
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.protos.feast.core.SavedDataset_pb2 import (
    SavedDatasetStorage as SavedDatasetStorageProto,
)
from feast.repo_config import RepoConfig
from feast.saved_dataset import SavedDatasetStorage
from feast.value_type import ValueType
from feast.infra.utils.doris.connection_util import _get_conn
from feast.infra.utils.doris.doris_config import doris_type_to_feast_value_type
from typeguard import typechecked


@typechecked
class DorisSQLSource(DataSource):
    def __init__(
        self,
        name: Optional[str] = None,
        query: Optional[str] = None,
        table: Optional[str] = None,
        timestamp_field: Optional[str] = "",
        created_timestamp_column: Optional[str] = "",
        field_mapping: Optional[Dict[str, str]] = None,
        description: Optional[str] = "",
        tags: Optional[Dict[str, str]] = None,
        owner: Optional[str] = "",
    ):
        self._doris_options = DorisSQLOptions(name=name, query=query, table=table)
        if name is None and table is None:
            raise DataSourceNoNameException()
        name = name or table
        assert name

        super().__init__(
            name=name,
            timestamp_field=timestamp_field,
            created_timestamp_column=created_timestamp_column,
            field_mapping=field_mapping,
            description=description,
            tags=tags,
            owner=owner,
        )

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, DorisSQLSource):
            raise TypeError(
                "Comparisons should only involve DorisSQLSource class objects."
            )

        return (
            super().__eq__(other)
            and self._doris_options._query == other._doris_options._query
            and self.timestamp_field == other.timestamp_field
            and self.created_timestamp_column == other.created_timestamp_column
            and self.field_mapping == other.field_mapping
        )

    @staticmethod
    def from_proto(data_source: DataSourceProto):
        assert data_source.HasField("custom_options")

        doris_options = json.loads(data_source.custom_options.configuration)

        return DorisSQLSource(
            name=doris_options["name"],
            query=doris_options["query"],
            table=doris_options["table"],
            field_mapping=dict(data_source.field_mapping),
            timestamp_field=data_source.timestamp_field,
            created_timestamp_column=data_source.created_timestamp_column,
            description=data_source.description,
            tags=dict(data_source.tags),
            owner=data_source.owner,
        )

    def to_proto(self) -> DataSourceProto:
        data_source_proto = DataSourceProto(
            name=self.name,
            type=DataSourceProto.CUSTOM_SOURCE,
            # Change this to doris path
            data_source_class_type="feast.infra.offline_stores.contrib.doris_offline_store.doris_source.DorisSQLSource",
            field_mapping=self.field_mapping,
            custom_options=self._doris_options.to_proto(),
            description=self.description,
            tags=self.tags,
            owner=self.owner,
        )

        data_source_proto.timestamp_field = self.timestamp_field
        data_source_proto.created_timestamp_column = self.created_timestamp_column

        return data_source_proto

    def validate(self, config: RepoConfig):
        pass

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        return doris_type_to_feast_value_type

    def get_table_column_names_and_types(
        self, config: RepoConfig
    ) -> Iterable[Tuple[str, str]]:
        with _get_conn(config.offline_store) as conn, conn.cursor() as cur:
            sql_query = f"SELECT * FROM {self.get_table_query_string()} AS sub LIMIT 0"
            df = pd.read_sql(
                sql_query,
                conn,
            )
            columns = list(df.columns)
            column_list = "', '".join(columns)
            query = f"""
            SELECT
                COLUMN_NAME,
                DATA_TYPE
            FROM
                information_schema.COLUMNS
            WHERE
                TABLE_SCHEMA = '{config.offline_store.database}'
                AND COLUMN_NAME IN ('{column_list}');
            """
            cur.execute(query)
            data = cur.fetchall()
            return data

    def get_table_query_string(self) -> str:
        if self._doris_options._table:
            return f"{self._doris_options._table}"
        else:
            return f"({self._doris_options._query})"


class DorisSQLOptions:
    def __init__(
        self,
        name: Optional[str],
        query: Optional[str],
        table: Optional[str],
    ):
        self._name = name or ""
        self._query = query or ""
        self._table = table or ""

    @classmethod
    def from_proto(cls, doris_options_proto: DataSourceProto.CustomSourceOptions):
        config = json.loads(doris_options_proto.configuration.decode("utf8"))
        doris_options = cls(
            name=config["name"], query=config["query"], table=config["table"]
        )

        return doris_options

    def to_proto(self) -> DataSourceProto.CustomSourceOptions:
        doris_options_proto = DataSourceProto.CustomSourceOptions(
            configuration=json.dumps(
                {"name": self._name, "query": self._query, "table": self._table}
            ).encode()
        )
        return doris_options_proto


class SavedDatasetDorisSQLStorage(SavedDatasetStorage):
    _proto_attr_name = "custom_storage"

    doris_options: DorisSQLOptions

    def __init__(self, table_ref: str):
        self.doris_options = DorisSQLOptions(table=table_ref, name=None, query=None)

    @staticmethod
    def from_proto(storage_proto: SavedDatasetStorageProto) -> SavedDatasetStorage:
        return SavedDatasetDorisSQLStorage(
            table_ref=DorisSQLOptions.from_proto(storage_proto.custom_storage)._table
        )

    def to_proto(self) -> SavedDatasetStorageProto:
        return SavedDatasetStorageProto(custom_storage=self.doris_options.to_proto())

    def to_data_source(self) -> DataSource:
        return DorisSQLSource(table=self.doris_options._table)
