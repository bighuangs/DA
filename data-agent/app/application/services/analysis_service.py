"""
API 按需数据分析服务：数据预测 + 数据异常检测。

一句话总述：独立于 LangGraph 主流水线的轻量 LLM 服务，接收前端按需请求
            （query/sql/chart/result），调用 LLM 产出预测数据行或异常检测报告，
            以异步生成器逐条产出 SSE chunk，供路由层包装为 StreamingResponse。

设计说明：
    - 数据预测由 LLM 基于历史趋势外推≥2周期，返回与原数据同结构的
      JSON 数组（仅预测段）；无法预测则返回原因文本。
    - 数据异常检测基于结果与聚合统计发现极值、突变、缺失、分布或逻辑异常，
      返回结构化中文报告。
    - 复用 generate_analysis 的 _compute_stats 计算聚合统计，避免重复实现。
"""
import json
import logging
import re
import statistics
from datetime import datetime, timedelta
from typing import AsyncGenerator

from app.application.ports.analysis_model import AnalysisModelPort


logger = logging.getLogger(__name__)
# 兼容旧测试 patch 点；生产调用通过 AnalysisModelPort 注入，不读取该变量。
invoke_optional_llm = None
# 预测时喂给 LLM 的最大历史行数（趋势外推无需全量，控制 token）
PREDICT_MAX_ROWS = 30
PREDICT_PERIODS = 2

_TIME_FIELD_HINTS = (
    "date", "day", "week", "month", "quarter", "year", "period",
    "日期", "日", "周", "月", "季度", "年", "周期",
)
_MEASURE_FIELD_HINTS = (
    "amount", "sales", "revenue", "gmv", "quantity", "count", "total",
    "rate", "ratio", "金额", "销售", "收入", "数量", "总计", "占比", "比率",
)


def _extract_json_array(text: str) -> list[dict]:
    """
    从 LLM 输出中容错提取首个 JSON 对象数组（预测数据行）。

    业务逻辑：
        1. 去除 Markdown 代码块围栏
        2. 正则定位首个 [...] 平衡片段
        3. json.loads 解析，仅保留 dict 元素；失败返回空列表
    """
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    match = re.search(r"\[.*?\]", candidate, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
        return []
    except Exception:
        return []


def _number(value) -> float | None:
    """把常见数值单元格转成 float；布尔值和脱敏文本不参与统计。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.endswith("%"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _round_like(value: float, samples: list) -> int | float:
    """沿用原序列的小数精度，避免预测结果出现无意义的浮点尾数。"""
    precision = 0
    for sample in samples:
        text = str(sample)
        if "." in text:
            precision = max(precision, min(6, len(text.rstrip("0").split(".", 1)[1])))
    rounded = round(value, precision)
    return int(rounded) if precision == 0 else rounded


def _ordered_numeric(values: list) -> bool:
    numeric = [_number(value) for value in values]
    return (
        len(numeric) >= 3
        and all(value is not None for value in numeric)
        and all(numeric[index] > numeric[index - 1] for index in range(1, len(numeric)))
    )


def _series_columns(result: list[dict]) -> tuple[str, str] | None:
    """选择一个时间/顺序字段和一个数值指标字段。"""
    if len(result) < 3:
        return None
    columns = list(result[0].keys())
    if len(columns) < 2:
        return None

    def time_score(column: str) -> tuple[int, int]:
        key = column.lower()
        values = [row.get(column) for row in result]
        hinted = any(hint in key for hint in _TIME_FIELD_HINTS)
        ordered = _ordered_numeric(values) or all(
            re.match(r"^\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2})?", str(value or ""))
            for value in values
        )
        return (2 if hinted else 0) + (1 if ordered else 0), -columns.index(column)

    x_column = max(columns, key=time_score)
    if time_score(x_column)[0] <= 0:
        return None

    measure_candidates = []
    for column in columns:
        if column == x_column:
            continue
        numeric_count = sum(_number(row.get(column)) is not None for row in result)
        if numeric_count < max(3, int(len(result) * 0.8)):
            continue
        key = column.lower()
        hinted = any(hint in key for hint in _MEASURE_FIELD_HINTS)
        measure_candidates.append((2 if hinted else 1, numeric_count, -columns.index(column), column))
    if not measure_candidates:
        return None
    return x_column, max(measure_candidates)[-1]


def _next_periods(column: str, values: list, query: str, count: int) -> list:
    """按字段类型推导后续期次；无法可靠推导时返回空列表。"""
    last = values[-1]
    key = column.lower()
    text = str(last).strip()

    month_match = re.match(r"^(\d{4})[-/年](\d{1,2})(?:月)?$", text)
    if month_match:
        year, month = map(int, month_match.groups())
        periods = []
        for _ in range(count):
            month += 1
            if month > 12:
                year += 1
                month = 1
            periods.append(f"{year:04d}-{month:02d}")
        return periods

    date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
    if date_match:
        parsed = [datetime.strptime(str(value), "%Y-%m-%d").date() for value in values]
        gaps = [(parsed[index] - parsed[index - 1]).days for index in range(1, len(parsed))]
        step = max(1, int(statistics.median(gaps)))
        return [(parsed[-1] + timedelta(days=step * index)).isoformat() for index in range(1, count + 1)]

    numeric = [_number(value) for value in values]
    if all(value is not None for value in numeric):
        integer_month = "month" in key or "月" in column
        if integer_month and int(numeric[-1]) == 12:
            year_match = re.search(r"(20\d{2})\s*年?", query or "")
            if year_match:
                year = int(year_match.group(1)) + 1
                return [f"{year:04d}-{index:02d}" for index in range(1, count + 1)]
        gaps = [numeric[index] - numeric[index - 1] for index in range(1, len(numeric))]
        if not gaps or statistics.median(gaps) <= 0:
            return []
        step = statistics.median(gaps)
        return [
            _round_like(numeric[-1] + step * index, values)
            for index in range(1, count + 1)
        ]
    return []


def _deterministic_prediction(
    query: str, chart: dict, result: list[dict], periods: int = PREDICT_PERIODS
) -> list[dict]:
    """为常见单指标时序生成快速、可重复的线性趋势预测。"""
    columns = _series_columns(result)
    if not columns:
        return []
    x_column, y_column = columns
    x_values = [row.get(x_column) for row in result]
    y_values = [_number(row.get(y_column)) for row in result]
    if any(value is None for value in y_values):
        return []
    next_x = _next_periods(x_column, x_values, query, periods)
    if len(next_x) != periods:
        return []

    window = y_values[-min(6, len(y_values)):]
    deltas = [window[index] - window[index - 1] for index in range(1, len(window))]
    slope = statistics.mean(deltas) if deltas else 0.0
    non_negative = all(value >= 0 for value in y_values)
    predicted = []
    for index, period in enumerate(next_x, start=1):
        row = dict(result[-1])
        row[x_column] = period
        value = y_values[-1] + slope * index
        if non_negative:
            value = max(0.0, value)
        row[y_column] = _round_like(value, [item.get(y_column) for item in result])
        predicted.append(row)
    return predicted


def _quartile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _deterministic_anomaly_report(result: list[dict]) -> str:
    """以 IQR、Z-score 和相邻变化率生成不依赖模型的异常说明。"""
    if not result:
        return "确定性异常检测：结果为空，当前没有可检测的数据点。"
    series_columns = _series_columns(result)
    dimension_column = series_columns[0] if series_columns else None
    numeric_columns = []
    for column in result[0]:
        # 月份、年份、期次等顺序字段虽然是数字，但属于坐标维度，不应被
        # 当成业务指标计算均值、突变或离群点。
        if column == dimension_column:
            continue
        values = [_number(row.get(column)) for row in result]
        present = [value for value in values if value is not None]
        if len(present) >= max(3, int(len(result) * 0.8)):
            numeric_columns.append((column, values, present))
    if not numeric_columns:
        return "确定性异常检测：结果中没有足够的连续数值字段，无法执行统计异常检测。"

    findings: list[str] = []
    summaries: list[str] = []
    for column, values, present in numeric_columns:
        mean = statistics.mean(present)
        std = statistics.pstdev(present) if len(present) > 1 else 0.0
        q1, q3 = _quartile(present, 0.25), _quartile(present, 0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = []
        for index, value in enumerate(values):
            if value is None:
                continue
            z_outlier = std > 0 and abs(value - mean) / std >= 3
            iqr_outlier = iqr > 0 and (value < lower or value > upper)
            if z_outlier or iqr_outlier:
                outliers.append(index + 1)
        changes = []
        for index in range(1, len(values)):
            previous, current = values[index - 1], values[index]
            if previous not in (None, 0) and current is not None:
                changes.append((index + 1, (current - previous) / abs(previous)))
        sharp = [(index, ratio) for index, ratio in changes if abs(ratio) >= 0.5]
        missing = len(values) - len(present)
        summaries.append(
            f"{column}：均值 {mean:,.2f}，范围 {min(present):,.2f}～{max(present):,.2f}"
            + (f"，缺失 {missing} 个" if missing else "")
        )
        if outliers:
            findings.append(f"{column} 的第 {', '.join(map(str, outliers))} 个数据点超出 IQR/Z-score 阈值")
        if sharp:
            details = "、".join(f"第{index}点 {ratio:+.1%}" for index, ratio in sharp[:5])
            findings.append(f"{column} 存在相邻突变：{details}")

    conclusion = (
        "；".join(findings)
        if findings
        else "未发现超过 IQR/Z-score 或 50% 相邻变化率阈值的明显异常"
    )
    return (
        f"确定性异常检测（{len(result)} 行）："
        + "；".join(summaries)
        + f"。结论：{conclusion}。该结果是统计筛查，不替代业务口径复核。"
    )


async def predict(
    query: str,
    sql: str,
    chart: dict,
    result: list[dict],
    *,
    model: AnalysisModelPort | None = None,
) -> AsyncGenerator[dict, None]:
    """
    数据预测：基于历史结果趋势外推未来数据行。

    业务逻辑：
        1. 推送阶段"数据预测"
        2. 由 result 取字段列、行数、截取前 PREDICT_MAX_ROWS 行喂 LLM
        3. 调用 generate_predict prompt → llm → StrOutputParser
        4. 容错解析 JSON 数组：成功推送 {"predict": rows}；失败推送 {"predict_failed": 原因文本}
        5. 异常时推送 {"predict_failed": 错误信息}，不抛出

    Yields:
        dict: SSE chunk —— {"stage":...} / {"predict":[...]} / {"predict_failed":"..."}
    """
    yield {"stage": "数据预测"}
    try:
        deterministic = _deterministic_prediction(query, chart, result)
        if deterministic:
            logger.info(f"确定性数据预测完成，预测 {len(deterministic)} 行")
            yield {
                "predict": deterministic,
                "method": "deterministic_linear_trend_v1",
            }
            return

        if model is None:
            yield {
                "predict_failed": (
                    "该数据无法进行确定性预测，且当前未装配模型增强能力。"
                )
            }
            return
        raw = await model.predict(
            query=query,
            sql=sql,
            chart=chart,
            result=result[:PREDICT_MAX_ROWS],
        )

        predicted = _extract_json_array(raw)
        if predicted:
            logger.info(f"数据预测完成，预测 {len(predicted)} 行")
            yield {"predict": predicted}
        else:
            # LLM 返回非 JSON（无法预测的原因文本）
            reason = raw.strip().strip("`").strip()
            logger.info(f"数据预测无法执行：{reason[:80]}")
            yield {"predict_failed": reason or "该数据无法进行预测（非趋势/时序数据或样本过少）。"}
    except Exception as e:
        logger.error(f"数据预测失败：{e}")
        yield {"predict_failed": f"数据预测服务异常：{e}"}


async def anomaly(
    query: str,
    sql: str,
    chart: dict,
    result: list[dict],
    *,
    model: AnalysisModelPort | None = None,
) -> AsyncGenerator[dict, None]:
    """
    数据异常检测：基于结果与聚合统计发现异常模式，产出中文报告。

    业务逻辑：
        1. 推送阶段"数据异常检测"
        2. 由 result 派生字段列、行数、聚合统计、前 10 行样例
        3. 调用 generate_anomaly prompt → llm → StrOutputParser
        4. 推送 {"anomaly": text}；异常时推送空串

    Yields:
        dict: SSE chunk —— {"stage":...} / {"anomaly":"..."}
    """
    yield {"stage": "数据异常检测"}
    try:
        deterministic = _deterministic_anomaly_report(result)
        if deterministic:
            logger.info(f"确定性数据异常检测完成（{len(deterministic)}字）")
            yield {
                "anomaly": deterministic,
                "method": "deterministic_iqr_zscore_v1",
            }
            return

        if model is None:
            yield {"anomaly": "当前没有可执行的确定性或模型异常检测能力。"}
            return
        text = await model.anomaly(
            query=query,
            sql=sql,
            chart=chart,
            result=result,
        )

        logger.info(f"数据异常检测完成（{len(text)}字）")
        yield {"anomaly": text.strip()}
    except Exception as e:
        logger.error(f"数据异常检测失败：{e}")
        yield {"anomaly": f"数据异常检测服务异常：{e}"}


class AnalysisService:
    """组合确定性分析与可选模型增强端口。"""

    def __init__(self, model: AnalysisModelPort | None = None) -> None:
        self._model = model

    def predict(
        self,
        query: str,
        sql: str,
        chart: dict,
        result: list[dict],
    ) -> AsyncGenerator[dict, None]:
        return predict(query, sql, chart, result, model=self._model)

    def anomaly(
        self,
        query: str,
        sql: str,
        chart: dict,
        result: list[dict],
    ) -> AsyncGenerator[dict, None]:
        return anomaly(query, sql, chart, result, model=self._model)
