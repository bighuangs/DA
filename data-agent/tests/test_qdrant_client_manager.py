from unittest.mock import patch

from app.infrastructure.clients.qdrant_client_manager import QdrantClientManager
from app.config.app_config import QdrantConfig


def test_qdrant_service_proxy_policy_is_explicitly_forwarded():
    manager = QdrantClientManager(QdrantConfig(
        host="127.0.0.1",
        port=6333,
        embedding_size=1024,
        trust_env=True,
    ))

    with patch(
        "app.infrastructure.clients.qdrant_client_manager.AsyncQdrantClient"
    ) as client_type:
        manager.init()

    client_type.assert_called_once_with(
        url="http://127.0.0.1:6333",
        trust_env=True,
        check_compatibility=False,
    )
