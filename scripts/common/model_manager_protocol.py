"""
ModelManager Protocol -- shared interface for all resource lifecycle managers.

Formalises the duck-typed contract already satisfied by ``ACITEIManager``,
``LocalDockerTEIManager``, and the new ``OllamaModelManager``.  Using a
``typing.Protocol`` means the concrete classes do **not** need to inherit
from anything; a type checker (pyright / mypy) verifies conformance
structurally at lint time, and ``@runtime_checkable`` allows ``isinstance``
checks in tests.

Minimum contract
~~~~~~~~~~~~~~~~
* ``model_id``       -- identifier of the model being managed
* ``endpoint_url``   -- set after ``start()``, ``None`` before / after ``stop()``
* ``provider_label`` -- short human-readable label (e.g. ``"aci_tei"``)
* ``start()``        -- provision / load the model; return the endpoint URL
* ``stop()``         -- tear down / unload; free resources (VRAM, containers …)
* ``ping()``         -- reset the idle-shutdown timer
* context manager    -- ``__enter__`` calls ``start()``, ``__exit__`` calls ``stop()``
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ModelManager(Protocol):
    """Structural interface for model lifecycle managers."""

    model_id: str
    endpoint_url: Optional[str]
    provider_label: str

    def start(self) -> str: ...
    def stop(self) -> None: ...
    def ping(self) -> None: ...
    def __enter__(self) -> ModelManager: ...
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...
