import logging
import os
import uuid
import zlib
from datetime import datetime as dt

import pandas as pd
from pandas.util import hash_pandas_object

from diaygeia.data_io.redis.engine import create_redis_engine
from diaygeia.domain import Conversation

DATETIME_FORMAT_STR = "%d/%m/%Y %H:%M:%S"

# Chat history expires automatically (data minimisation); each new turn renews it.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 24 * 3600))

SESSION_KEY_PREFIX = "session:"


def _session_key(session_id):
    """Redis key for a chat session: ``session:<uuid>``.

    Only a canonical UUID (the lowercase, hyphenated form ``str(uuid.uuid4())``
    produces) is accepted, so a session id can never address any other Redis key.
    Session ids are generated server-side; this closes the pattern regardless.
    """
    try:
        canonical = str(uuid.UUID(str(session_id)))
    except (ValueError, TypeError, AttributeError):
        canonical = None
    if canonical is None or canonical != session_id:
        raise ValueError("invalid session id")
    return SESSION_KEY_PREFIX + canonical


class SessionManager:
    def __init__(self, logger=None):
        if not logger:
            self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.engine = create_redis_engine(self.logger)

    def get_user_session(self, session_id):
        try:
            key = _session_key(session_id)
            if self.engine.get(key):
                result = Conversation(
                    pd.read_json(zlib.decompress(self.engine.get(key)).decode("utf-8"))
                )
            else:
                result = Conversation(pd.DataFrame(), columns=Conversation.column_names)
        except Exception as e:
            self.logger.warning(f"Could not retrieve user session due to {e}")
            result = Conversation(pd.DataFrame(), columns=Conversation.column_names)

        return result

    def reset_user_session(self, session_id):
        try:
            key = _session_key(session_id)
            self.engine.delete(key)
        except Exception as e:
            self.logger.warning(f"Could not reset user session due to {e}")

    def update_user_session(
        self, session_id, conversation: Conversation, question: str, response: str
    ):
        try:
            user_conversation = Conversation(
                pd.concat(
                    [
                        conversation,
                        pd.DataFrame(
                            [
                                {
                                    "user": session_id,
                                    "utterance": question,
                                    "time": dt.now().strftime(DATETIME_FORMAT_STR),
                                }
                            ]
                        ),
                        pd.DataFrame(
                            [
                                {
                                    "user": "BOT",
                                    "utterance": response,
                                    "time": dt.now().strftime(DATETIME_FORMAT_STR),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            )
            key = _session_key(session_id)
            json_s = zlib.compress(user_conversation.to_json().encode("utf-8"))
            self.engine.set(key, json_s, ex=SESSION_TTL_SECONDS)
        except Exception as e:
            self.logger.warning(f"Could not update user session due to {e}")


def _handle_unhashable_types(x):
    if type(x) == list:
        return tuple(x)
    elif type(x) == dict:
        return frozenset(x.items())
    else:
        return x


def _hash_known_args(args):
    key_hash = []
    for arg in args:
        if issubclass(type(arg), pd.DataFrame):
            key_hash.append(
                str(hash_pandas_object(arg.applymap(_handle_unhashable_types)).sum())
            )
        elif issubclass(type(arg), pd.Series):
            key_hash.append(
                str(hash_pandas_object(arg.map(_handle_unhashable_types)).sum())
            )
        elif issubclass(type(arg), str):
            key_hash.append(arg)
        elif issubclass(type(arg), int) or issubclass(type(arg), float):
            key_hash.append(str(arg))
        else:
            continue
    return key_hash
