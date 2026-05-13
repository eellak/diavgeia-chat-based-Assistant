import logging
from dataclasses import dataclass

import pandas as pd


class InfoDataFrame(pd.DataFrame):
    column_info = None
    column_names = None

    def __init__(self, *args, **kw):
        super(InfoDataFrame, self).__init__(*args, **kw)
        if "logger" in kw:
            self.logger = kw["logger"]
        else:
            self.logger = logging.getLogger(__name__)
        self._check_missing_columns(self.column_names, self.columns)
        for col, col_type in self.column_info.__annotations__.items():
            if col not in self.columns:
                print(f"Column {col} not in {self.columns} Creating it as None")
                self[col] = None
            self._fillna(col, col_type)
            if col_type in {int, float, str, bool}:
                self[col] = self[col].astype(col_type)

    def _fillna(self, k, v):

        if v == int or v == float:
            self[k] = self[k].fillna(-1)
        elif v == str:
            self[k] = self[k].fillna("")
        elif v == bool:
            self[k] = self[k].fillna(False)

    def _check_missing_columns(self, expected_columns, actual_columns):
        columns_exist = set(expected_columns).issubset(set(actual_columns))
        if not columns_exist:
            self.logger.warn(
                f"Expected columns {set(expected_columns) - set(actual_columns)} columns are missing. Casting them as None."
            )


@dataclass
class ConversationUtterance:
    user: str
    utterance: str
    time: str


class Conversation(InfoDataFrame):
    column_info = ConversationUtterance
    column_names = list(column_info.__annotations__.keys())
