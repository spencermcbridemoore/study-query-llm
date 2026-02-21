"""
Database Connection Management (V1 -- deprecated).

.. deprecated::
    Use :class:`~study_query_llm.db.connection_v2.DatabaseConnectionV2` and the
    v2 schema for all new work.  This module is retained only for migration
    tests and backward compatibility.
"""

import warnings

from .models import Base
from ._base_connection import BaseDatabaseConnection


class DatabaseConnection(BaseDatabaseConnection):
    """V1 database connection.

    .. deprecated::
        Use ``DatabaseConnectionV2`` instead.
    """

    def __init__(self, connection_string: str, echo: bool = False):
        warnings.warn(
            "DatabaseConnection (V1) is deprecated; use DatabaseConnectionV2",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(connection_string, echo=echo)

    def _get_metadata(self):
        return Base.metadata
