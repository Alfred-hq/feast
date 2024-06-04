from datetime import datetime
from typing import Dict

import mysql.connector as sql
import numpy as np
import pandas as pd
import pyarrow as pa
from pymysql import NULL

from .doris_config import DorisSQLConfig, arrow_to_doris_type


def _get_conn(config: DorisSQLConfig):
    conn = sql.connect(
        database=config.database,
        host=config.host,
        port=int(config.port),
        user=config.user,
        password=config.password,
    )
    return conn


def _df_to_create_table_sql(entity_df, table_name) -> str:
    pa_table = pa.Table.from_pandas(entity_df)
    columns = [
        f"`{f.name}` {arrow_to_doris_type(str(f.type))}" for f in pa_table.schema
    ]
    return f"""
        CREATE TABLE IF NOT EXISTS`{table_name}` (
            {", ".join(columns)}
        )
        DISTRIBUTED BY HASH(`event_timestamp`) BUCKETS 1
        PROPERTIES ('replication_num' = '1');
    """


def df_to_doris_table(
    config: DorisSQLConfig, df: pd.DataFrame, table_name: str
) -> Dict[str, np.dtype]:
    with _get_conn(config) as conn, conn.cursor() as cur:
        table_creation_query = _df_to_create_table_sql(df, table_name)
        cur.execute(table_creation_query)
        conn.commit()
        rows = df.replace({np.NaN: NULL}).values.tolist()

        rows = [
            [
                (
                    elem.strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(elem, datetime)
                    else elem
                )
                for elem in row
            ]
            for row in rows
        ]
        values = ", ".join([f"({', '.join(map(repr, row))})" for row in rows])
        insert_sql = f"""INSERT INTO `{table_name}` VALUES {values};"""
        cur.execute(insert_sql)
        conn.commit()
        return dict(zip(df.columns, df.dtypes))


def get_query_schema(config: DorisSQLConfig, sql_query: str) -> Dict[str, np.dtype]:
    with _get_conn(config) as conn:
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
            TABLE_SCHEMA = '{config.database}'
            AND COLUMN_NAME IN ('{column_list}');
        """
        dtype_df = pd.read_sql(query, conn)
        dtype_mapping = {
            "NULL": np.dtype(object),
            "BOOLEAN": np.dtype(np.bool_),
            "TINYINT": np.dtype(np.int8),
            "SMALLINT": np.dtype(np.int16),
            "INT": np.dtype(np.int32),
            "BIGINT": np.dtype(np.int64),
            "ARRAY<INT>": np.dtype(object),
            "ARRAY<BIGINT>": np.dtype(object),
            "ARRAY<BOOLEAN>": np.dtype(object),
            "ARRAY<DOUBLE>": np.dtype(object),
            "ARRAY<DATETIME>": np.dtype(object),
            "FLOAT": np.dtype(np.float32),
            "DOUBLE": np.dtype(np.float64),
            "BINARY": np.dtype(object),
            "STRING": np.dtype(object),
            "DATETIME": np.dtype("datetime64[ns]"),
            "TIMESTAMP": np.dtype("datetime64[ns]"),
            "TIMESTAMP WITH TIME ZONE": np.dtype("datetime64[ns]"),
            "TIMESTAMP WITHOUT TIME ZONE": np.dtype("datetime64[ns]"),
        }
        dtype_df["DATA_TYPE"] = dtype_df["DATA_TYPE"].apply(
            lambda x: dtype_mapping.get(x.upper(), np.dtype(object))
        )

        return dict(zip(dtype_df["COLUMN_NAME"], dtype_df["DATA_TYPE"]))
