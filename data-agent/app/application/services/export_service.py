"""API 查询结果导出与 XLSX 文件构造。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

from app.domain.query.result_normalization import normalize_result_value, normalize_rows


class ExportLimitError(ValueError):
    """Export snapshot exceeds a configured resource boundary."""


@dataclass(frozen=True, slots=True)
class ExportLimits:
    """一次查询结果导出的资源边界。"""

    max_rows: int = 100_000
    max_columns: int = 200
    max_cells: int = 2_000_000
    max_cell_chars: int = 32_767

    def __post_init__(self) -> None:
        if min(
            self.max_rows,
            self.max_columns,
            self.max_cells,
            self.max_cell_chars,
        ) < 1:
            raise ValueError("导出限制必须全部大于 0")


_DEFAULT_EXPORT_LIMITS = ExportLimits()


def validate_export_shape(
    columns: list[str],
    rows: list[dict],
    *,
    limits: ExportLimits = _DEFAULT_EXPORT_LIMITS,
) -> None:
    if len(rows) > limits.max_rows:
        raise ExportLimitError(f"导出行数超过上限 {limits.max_rows}")
    if len(columns) > limits.max_columns:
        raise ExportLimitError("导出列数超过上限")
    if len(columns) * (len(rows) + 1) > limits.max_cells:
        raise ExportLimitError("导出单元格数量超过上限")
    if any(
        len(str(value)) > limits.max_cell_chars
        for row in rows for value in row.values() if value is not None
    ):
        raise ExportLimitError("导出单元格文本长度超过上限")


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_text(value) -> str:
    return escape(str(value), {"\"": "&quot;", "'": "&apos;"})


def _cell(reference: str, value) -> str:
    value = normalize_result_value(value)
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f'<c r="{reference}" t="n"><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{_xml_text(value if value is not None else "")}</t></is></c>'


def build_xlsx(columns: list[str], rows: list[dict]) -> bytes:
    """构造单 Sheet XLSX；大数和 Decimal 以文本单元格写入，保证精确。"""
    columns = [str(column) for column in columns]
    normalized = normalize_rows(rows)
    xml_rows = []
    header_cells = "".join(_cell(f"{_column_name(index)}1", column) for index, column in enumerate(columns, 1))
    xml_rows.append(f'<row r="1">{header_cells}</row>')
    for row_index, row in enumerate(normalized, 2):
        cells = "".join(
            _cell(f"{_column_name(column_index)}{row_index}", row.get(column))
            for column_index, column in enumerate(columns, 1)
        )
        xml_rows.append(f'<row r="{row_index}">{cells}</row>')

    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="查询结果" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return output.getvalue()


def safe_filename(filename: str) -> str:
    name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", filename or "data-agent-result").strip("-")
    return (name or "data-agent-result")[:80]
