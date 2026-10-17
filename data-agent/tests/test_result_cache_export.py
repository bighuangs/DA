import unittest
from zipfile import ZipFile
from io import BytesIO
from decimal import Decimal

from app.infrastructure.cache.result_cache import ResultCache
from app.application.services.export_service import build_xlsx, normalize_result_value


class ResultCacheTest(unittest.TestCase):
    def test_cache_returns_copy_and_tracks_hit(self):
        cache = ResultCache(max_entries=2)
        value = [{"amount": "12345678901234567890"}]
        cache.set("sql-key", value, ttl_seconds=30)
        first = cache.get("sql-key", ttl_seconds=30)
        first[0]["amount"] = "changed"
        second = cache.get("sql-key", ttl_seconds=30)

        self.assertEqual(second[0]["amount"], "12345678901234567890")
        self.assertEqual(cache.stats()["hits"], 2)


class ResultExportTest(unittest.TestCase):
    def test_large_values_are_normalized_without_float_conversion(self):
        self.assertEqual(normalize_result_value(Decimal("12345678901234567890.12")), "12345678901234567890.12")
        self.assertEqual(normalize_result_value(12345678901234567890), "12345678901234567890")
        self.assertEqual(normalize_result_value(123), 123)

    def test_build_xlsx_contains_sheet_and_exact_large_value(self):
        content = build_xlsx(
            ["id", "amount"],
            [{"id": 1, "amount": Decimal("12345678901234567890.12")}],
        )
        with ZipFile(BytesIO(content)) as archive:
            self.assertIn("xl/worksheets/sheet1.xml", archive.namelist())
            sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
        self.assertIn("12345678901234567890.12", sheet)


if __name__ == "__main__":
    unittest.main()
