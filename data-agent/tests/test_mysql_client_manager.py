import unittest

from app.infrastructure.clients.mysql_client_manager import MysqlClientManager
from app.config.app_config import DBConfig


class MysqlClientManagerTest(unittest.TestCase):
    def test_url_create_preserves_special_characters_in_credentials(self):
        manager = MysqlClientManager(DBConfig(
            host="127.0.0.1",
            port=3306,
            user="user@local",
            password="p@ss:/?#word",
            database="meta2",
        ))
        url = manager._get_url()
        self.assertEqual(url.username, "user@local")
        self.assertEqual(url.password, "p@ss:/?#word")
        self.assertEqual(url.database, "meta2")


if __name__ == "__main__":
    unittest.main()
