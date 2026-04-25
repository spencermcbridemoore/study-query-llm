"""
Database Connection Management (V1 -- deprecated).

.. deprecated::
    Use :class:`~study_query_llm.db.connection_v2.DatabaseConnectionV2` and the
    v2 schema for all new work.  This module is retained only for migration
    tests and backward compatibility.
"""

import warnings

from .write_intent import WriteIntent
from .models import Base
from ._base_connection import BaseDatabaseConnection


class DatabaseConnection(BaseDatabaseConnection):
    """V1 database connection.

    .. deprecated::
        Use ``DatabaseConnectionV2`` instead.
    """

    def __init__(
        self,
        connection_string: str,
        echo: bool = False,
        write_intent: WriteIntent | str | None = None,
        quiet: bool = False,
    ):
        warnings.warn(
            "DatabaseConnection (V1) is deprecated; use DatabaseConnectionV2",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            connection_string,
            echo=echo,
            write_intent=write_intent,
            quiet=quiet,
        )

    def _get_metadata(self):
        return Base.metadata
