"""
NL2SQL 评测历史对比 CLI（TODO-23）。

一句话总述：加载 var/evals/legacy/runs/ 下所有结构化评测结果，按时间排序，生成跨 run 的指标对比
            与逐用例通过/失败矩阵，定位回归与改善点。

用法：
    python -m evals.legacy.compare_runs --suite easy_sql_v1
    python -m evals.legacy.compare_runs --suite easy_sql_v1 --last 3
    python -m evals.legacy.compare_runs --suite easy_sql_v1 --runs 20260711_171436,20260711_180000
"""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

from evals.legacy.governance import (
    compare_fingerprints,
    experiment_report,
    resolve_baseline_artifact,
)
from evals.paths import REPORTS_ROOT, RUNS_ROOT

RUNS_DIR = RUNS_ROOT
COMPARE_DIR = REPORTS_ROOT / "nl2sql"


def run_payload_is_valid(payload: dict) -> bool:
    """Accept only artifacts that explicitly satisfy the current run contract."""
    return payload.get("valid") is True


def _file_slug(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} 不能为空")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", normalized).strip(".-")
    if not slug:
        raise ValueError(f"{label} 不包含可用的文件名字符")
    return slug


def load_runs(
    dataset_suite: str,
    last: int | None = None,
    run_ids: list[str] | None = None,
) -> list[dict]:
    """加载同一 suite 的历史 run，按 run_id（时间戳）升序。"""
    dataset_suite = str(dataset_suite or "").strip()
    if not dataset_suite:
        raise ValueError("dataset_suite 不能为空")
    if not RUNS_DIR.exists():
        return []
    files = sorted(RUNS_DIR.glob("run_*.json"), key=lambda p: p.name)
    runs = []
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            if (
                run_payload_is_valid(payload)
                and str(payload.get("dataset_suite") or "") == dataset_suite
            ):
                runs.append(payload)
        except Exception:
            continue
    if run_ids:
        id_set = set(run_ids)
        runs = [r for r in runs if r.get("run_id") in id_set]
    if last:
        runs = runs[-last:]
    return runs


def load_experiment_runs(
    experiment_id: str,
    dataset_suite: str,
) -> list[dict]:
    """Load one suite's experiment, including invalid infrastructure runs."""
    dataset_suite = str(dataset_suite or "").strip()
    if not dataset_suite:
        raise ValueError("dataset_suite 不能为空")
    runs: list[dict] = []
    if not RUNS_DIR.exists():
        return runs
    for path in sorted(RUNS_DIR.glob("run_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            (payload.get("experiment") or {}).get("experiment_id")
            == experiment_id
            and str(payload.get("dataset_suite") or "") == dataset_suite
        ):
            payload["_artifact"] = path.name
            runs.append(payload)
    return runs


def render_experiment(report: dict) -> str:
    lines = [
        f"# NL2SQL 实验组 {report.get('experiment_id') or '未找到'}",
        "",
        f"> 状态：`{report.get('status')}` ｜ 完整：{report.get('complete')} ｜ 可批准：{report.get('qualified')}",
        "",
    ]
    if not report.get("available"):
        lines.append(str(report.get("message") or "没有实验数据。"))
        return "\n".join(lines)
    lines.extend([
        "## 最差一次门禁",
        "",
        "| 表 exact | 结果一致 | TTSQL P95 | TTResult P95 | TTComplete P95 | 单题 Token 最大值 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {report['worst']['tables_correct']} | {report['worst']['result_equivalent']} | "
            f"{report['worst']['tt_sql_p95']:.1f} ms | {report['worst']['tt_result_p95']:.1f} ms | "
            f"{report['worst']['tt_complete_p95']:.1f} ms | {report['worst'].get('tokens_max', 0):.0f} |"
        ),
        "",
        "## 三次运行",
        "",
        "| Repeat | Run | 缓存 | 表 exact | 结果一致 | TTSQL | TTResult | TTComplete | Token max / case | 模型失败 | 基础设施错误 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ])
    for item in report.get("runs") or []:
        lines.append(
            f"| {item['repeat_index']} | {item['run_id']} | {item['cache_state']} | "
            f"{item['tables_correct']}/{item['total']} | {item['result_equivalent']}/{item['total']} | "
            f"{item['tt_sql_p95']:.1f} | {item['tt_result_p95']:.1f} | "
            f"{item['tt_complete_p95']:.1f} | {item.get('tokens_max', 0):.0f} / "
            f"{item.get('tokens_max_case_id') or 'N/A'} | {item.get('failed_llm_calls', 0)} | "
            f"{item['infrastructure_errors']} |"
        )
    lines.extend(["", "## 准入清单", "", "| 检查 | 结果 |", "| --- | --- |"])
    for key, passed in (report.get("checks") or {}).items():
        lines.append(f"| {key} | {'✅' if passed else '❌'} |")
    lines.extend([
        "",
        f"- 回归用例：{', '.join(report.get('regressed_case_ids') or []) or '无'}",
        f"- 指纹差异：{', '.join(report.get('fingerprint_differences') or []) or '无'}",
        "",
    ])
    return "\n".join(lines)


def _pct(n: int, d: int) -> str:
    return f"{(n / d * 100):.1f}%" if d else "N/A"


def render_compare(runs: list[dict]) -> str:
    """渲染跨 run 对比 Markdown：整体指标趋势 + 逐用例通过矩阵 + 回归/改善清单。"""
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# NL2SQL 评测历史对比\n", f"> 生成时间：{ts} ｜ 对比 run 数：{len(runs)}\n"]
    if not runs:
        lines.append("无历史 run，请先运行 `python -m evals.legacy.run_eval`。")
        return "\n".join(lines)

    # ── 整体指标趋势：每行一个指标，每列一个 run ──
    lines.append("## 整体指标趋势\n")
    header = "| 指标 | " + " | ".join(r["run_id"] for r in runs) + " |"
    sep = "|---|" + "|".join(["---"] * len(runs)) + "|"
    lines.append(header)
    lines.append(sep)
    metric_rows = [
        ("用例数", lambda a: str(a["n"])),
        ("语法正确率", lambda a: _pct(a["syntax_valid"], a["n"])),
        ("执行成功率", lambda a: _pct(a["exec_success"], a["n"])),
        ("SQL 等价率", lambda a: _pct(a["sql_equivalent"], a["n"])),
        ("表选择准确率", lambda a: _pct(a["tables_correct"], a["n"])),
        ("结果一致率", lambda a: _pct(a["result_equivalent"], a["n"])),
        ("纠错触发率", lambda a: _pct(a["retry_triggered"], a["n"])),
        ("纠错成功率", lambda a: _pct(a["retry_success"], a["retry_triggered"])),
        ("平均延迟(ms)", lambda a: str(a["latency_mean"])),
        ("P95延迟(ms)", lambda a: str(a["latency_p95"])),
    ]
    for name, fn in metric_rows:
        cells = " | ".join(fn(r["aggregate"]) for r in runs)
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    # ── 相邻 run 关键指标 delta ──
    if len(runs) >= 2:
        lines.append("## 版本指纹可比性\n")
        lines.append("| 对比 | 是否可直接归因 | 差异 |")
        lines.append("|---|---|---|")
        for i in range(1, len(runs)):
            comparison = compare_fingerprints(runs[i - 1].get("fingerprint"), runs[i].get("fingerprint"))
            differences = "、".join(comparison["differences"]) if comparison["differences"] else "无"
            lines.append(
                f"| {runs[i-1]['run_id']}→{runs[i]['run_id']} | "
                f"{'是' if comparison['comparable'] else '否'} | {differences} |"
            )
        lines.append("")

        lines.append("## 相邻 run 关键指标变化（Δ）\n")
        lines.append("| 对比 | 结果一致率 | SQL等价率 | 表选择 | 平均延迟 |")
        lines.append("|---|---|---|---|---|")
        for i in range(1, len(runs)):
            prev, cur = runs[i - 1]["aggregate"], runs[i]["aggregate"]
            def _delta(key):
                pv, cv = prev[key], cur[key]
                d = cv - pv
                sign = "+" if d > 0 else ""
                return f"{_pct(pv, prev['n'])}→{_pct(cv, cur['n'])}({sign}{d/prev['n']*100:.1f}%)" if prev["n"] else "N/A"
            def _delta_raw(key):
                pv, cv = prev[key], cur[key]
                d = cv - pv
                sign = "+" if d > 0 else ""
                return f"{pv}→{cv}({sign}{d})"
            lines.append(
                f"| {runs[i-1]['run_id']}→{runs[i]['run_id']} | "
                f"{_delta('result_equivalent')} | {_delta('sql_equivalent')} | "
                f"{_delta('tables_correct')} | {_delta_raw('latency_mean')} |"
            )
        lines.append("")

    # ── 逐用例通过/失败矩阵：定位回归与改善 ──
    all_case_ids = []
    seen = set()
    for r in runs:
        for c in r["cases"]:
            cid = c["id"]
            if cid not in seen:
                seen.add(cid)
                all_case_ids.append(cid)
    lines.append("## 逐用例结果一致率矩阵（✅通过/❌不一致/⚠️执行失败）\n")
    header = "| 用例ID | 问题 | " + " | ".join(r["run_id"] for r in runs) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(runs)) + "|"
    lines.append(header)
    lines.append(sep)
    # 按 run_id 建 id->case 索引
    case_map = {r["run_id"]: {c["id"]: c for c in r["cases"]} for r in runs}
    for cid in all_case_ids:
        # 取问题文案（任一 run 中找）
        q = next((case_map[r["run_id"]][cid]["question"] for r in runs if cid in case_map[r["run_id"]]), "")
        cells = []
        for r in runs:
            c = case_map[r["run_id"]].get(cid)
            if not c:
                cells.append("—")
            elif not c["exec_success"]:
                cells.append("⚠️")
            elif c["result_equivalent"]:
                cells.append("✅")
            else:
                cells.append("❌")
        lines.append(f"| {cid} | {q} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── 回归/改善清单（仅相邻最近两次）──
    if len(runs) >= 2:
        prev, cur = runs[-2], runs[-1]
        pm, cm = case_map[prev["run_id"]], case_map[cur["run_id"]]
        regressed, improved = [], []
        for cid in all_case_ids:
            pc, cc = pm.get(cid), cm.get(cid)
            if not pc or not cc:
                continue
            p_ok = pc["exec_success"] and pc["result_equivalent"]
            c_ok = cc["exec_success"] and cc["result_equivalent"]
            if p_ok and not c_ok:
                regressed.append(cid)
            elif not p_ok and c_ok:
                improved.append(cid)
        lines.append(f"## 最近两次对比（{prev['run_id']} → {cur['run_id']}）\n")
        lines.append(f"- 📈 改善用例：{', '.join(improved) if improved else '无'}")
        lines.append(f"- 📉 回归用例：{', '.join(regressed) if regressed else '无'}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", required=True,
        help="精确选择 dataset_suite；可传当前或历史 suite，不允许跨 suite 聚合",
    )
    parser.add_argument("--last", type=int, default=None, help="只对比最近 N 次 run")
    parser.add_argument("--runs", default=None, help="指定 run_id 列表，逗号分隔")
    parser.add_argument("--experiment-id", default=None, help="按 experiment 聚合三次 run 并使用最差值准入")
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=None,
        help="可选的当前基线制品；不再读取旧文件 registry 或历史快照",
    )
    args = parser.parse_args()
    dataset_suite = str(args.suite).strip()
    suite_slug = _file_slug(dataset_suite, "dataset_suite")
    if args.experiment_id:
        experiment_slug = _file_slug(args.experiment_id, "experiment_id")
        runs = load_experiment_runs(args.experiment_id, dataset_suite)
        workspace_id = str((runs[0] if runs else {}).get("workspace_id") or "")
        suites = {str(run.get("dataset_suite") or "") for run in runs}
        expected_suite = next(iter(suites)) if len(suites) == 1 else None
        baseline = (
            resolve_baseline_artifact(
                explicit=args.baseline_run,
                workspace_id=workspace_id or None,
                expected_dataset_suite=expected_suite,
            ).get("payload")
            if args.baseline_run
            else None
        )
        result = experiment_report(
            runs, baseline_run=baseline, experiment_id=args.experiment_id,
        )
        report = render_experiment(result)
        COMPARE_DIR.mkdir(parents=True, exist_ok=True)
        out = COMPARE_DIR / (
            f"experiment_{suite_slug}_{experiment_slug}.md"
        )
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"\n实验报告已写入：{out}")
        raise SystemExit(0 if result.get("qualified") else 2)
    run_ids = [x.strip() for x in args.runs.split(",")] if args.runs else None
    runs = load_runs(
        dataset_suite, last=args.last, run_ids=run_ids,
    )
    report = render_compare(runs)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = COMPARE_DIR / f"compare_{suite_slug}_{ts}.md"
    out.write_text(report, encoding="utf-8")
    (COMPARE_DIR / f"compare_{suite_slug}_latest.md").write_text(
        report, encoding="utf-8",
    )
    print(report)
    print(f"\n对比报告已写入：{out}")


if __name__ == "__main__":
    main()
