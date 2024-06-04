from typing import Dict

from feast.repo_config import FeastConfigBaseModel
from feast.type_map import feast_value_type_to_pa
from feast.value_type import ValueType
from pydantic import Field, StrictStr


class DorisSQLConfig(FeastConfigBaseModel):
    host: StrictStr = Field(default="127.0.0.1")
    port: int = 9030
    database: StrictStr = Field(default="demo")
    user: StrictStr = Field(default="root")
    password: StrictStr = Field(default="root")


def doris_type_code_to_arrow(code: int) -> str:
    return feast_value_type_to_pa(
        doris_type_to_feast_value_type(doris_type_code_to_doris_type(code))
    )


def arrow_to_doris_type(t_str: str) -> str:
    try:
        if t_str.startswith("timestamp") or t_str.startswith("datetime"):
            return "DATETIME" if "tz=" in t_str else "DATETIME"
        return {
            "null": "NULL",
            "bool": "BOOLEAN",
            "int8": "TINYINT",
            "int16": "SMALLINT",
            "int32": "INT",
            "int64": "BIGINT",
            "list<item: int32>": "ARRAY<INT>",
            "list<item: int64>": "ARRAY<BIGINT>",
            "list<item: bool>": "ARRAY<BOOLEAN>",
            "list<item: double>": "ARRAY<DOUBLE>",
            "list<item: timestamp[us]>": "ARRAY<DATETIME>",
            "uint8": "SMALLINT",
            "uint16": "INT",
            "uint32": "BIGINT",
            "uint64": "BIGINT",
            "float": "FLOAT",
            "double": "DOUBLE",
            "binary": "BINARY",
            "string": "STRING",
        }[t_str]
    except KeyError:
        raise ValueError(f"Unsupported type: {t_str}")


def doris_type_code_to_doris_type(code: int) -> str:
    DORIS_TYPE_MAP = {
        1: "boolean",
        2: "smallint",
        3: "int",
        8: "bigint",
        254: "struct",
        4: "float",
        5: "double",
        246: "decimal",
        10: "date",
        12: "datetime",
        144: "map",
        252: "string",
    }

    return DORIS_TYPE_MAP.get(code, "string")


def doris_type_to_feast_value_type(type_str: str) -> ValueType:
    type_map: Dict[str, ValueType] = {
        "boolean": ValueType.BOOL,
        "smallint": ValueType.INT32,
        "int": ValueType.INT32,
        "bigint": ValueType.INT64,
        "float": ValueType.FLOAT,
        "double": ValueType.DOUBLE,
        "decimal": ValueType.DOUBLE,
        "date": ValueType.UNIX_TIMESTAMP,
        "datetime": ValueType.UNIX_TIMESTAMP,
        "string": ValueType.STRING,
        "struct": ValueType.STRING,
        "map": ValueType.STRING,
    }
    value = (
        type_map[type_str.lower()]
        if type_str.lower() in type_map
        else ValueType.UNKNOWN
    )
    if value == ValueType.UNKNOWN:
        print("unknown type:", type_str)
    return value
