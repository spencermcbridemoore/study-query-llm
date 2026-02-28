"""Protocol conformance tests for all ModelManager implementations.

Uses the ``@runtime_checkable`` flag on the ``ModelManager`` Protocol to
verify that each concrete class satisfies the structural interface at
runtime via ``isinstance()``.
"""

from study_query_llm.providers.managers import (
    ModelManager,
    OllamaModelManager,
    LocalDockerTEIManager,
    ACITEIManager,
)


def test_ollama_model_manager_satisfies_protocol():
    mgr = OllamaModelManager(model_id="llama3.1:8b")
    assert isinstance(mgr, ModelManager)


def test_local_docker_tei_manager_satisfies_protocol():
    mgr = LocalDockerTEIManager(model_id="BAAI/bge-m3")
    assert isinstance(mgr, ModelManager)


def test_aci_tei_manager_satisfies_protocol():
    mgr = ACITEIManager(
        subscription_id="fake-sub",
        resource_group="fake-rg",
        container_name="fake-container",
        model_id="BAAI/bge-m3",
    )
    assert isinstance(mgr, ModelManager)
