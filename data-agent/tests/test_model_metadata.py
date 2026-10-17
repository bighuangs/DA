import unittest

from sqlalchemy.sql.ddl import sort_tables_and_constraints
from sqlalchemy import Text

from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import Base


class ModelMetadataTest(unittest.TestCase):
    def test_all_foreign_keys_resolve_before_startup_create_all(self):
        """复现 create_all 的依赖排序，避免错误表名让应用启动失败。"""
        sorted_items = sort_tables_and_constraints(Base.metadata.tables.values())
        table_names = {table.name for table, _ in sorted_items if table is not None}
        self.assertIn("app_user", table_names)
        self.assertIn("access_policy", table_names)
        for table in Base.metadata.tables.values():
            for foreign_key in table.foreign_keys:
                self.assertIsNotNone(foreign_key.column.table)

    def test_text_columns_do_not_use_incompatible_server_defaults(self):
        for table in Base.metadata.tables.values():
            for column in table.columns:
                if isinstance(column.type, Text):
                    self.assertIsNone(
                        column.server_default,
                        f"{table.name}.{column.name} 不应声明 TEXT DEFAULT",
                    )


if __name__ == "__main__":
    unittest.main()
