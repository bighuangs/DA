import unittest

from app.application.services.export_service import build_xlsx
from app.application.services.adhoc_dataset_service import (
    _infer_type,
    _normalise_rows,
    _parse_csv,
    _parse_xlsx,
)


class AdhocParserTest(unittest.TestCase):
    def test_csv_supports_quoted_values_and_duplicate_headers(self):
        content = '地区,地区,金额\n"华东,一组",华东,1200\n'.encode()
        headers, rows = _normalise_rows(_parse_csv(content))
        self.assertEqual(headers, ["地区", "地区_2", "金额"])
        self.assertEqual(rows[0], ["华东,一组", "华东", "1200"])

    def test_xlsx_supports_inline_strings_and_numbers(self):
        content = build_xlsx(["地区", "金额"], [{"地区": "华东", "金额": 1200}])
        self.assertEqual(
            _parse_xlsx(content),
            [["地区", "金额"], ["华东", "1200"]],
        )

    def test_type_inference_prefers_safe_fallback(self):
        self.assertEqual(_infer_type(["1", "2", "3"]), "BIGINT")
        self.assertEqual(_infer_type(["1.20", "3.40"]), "DECIMAL(30,6)")
        self.assertEqual(_infer_type(["2026-07-01", "2026-07-02"]), "DATE")
        self.assertEqual(_infer_type(["1", "unknown"]), "TEXT")

if __name__ == "__main__":
    unittest.main()
