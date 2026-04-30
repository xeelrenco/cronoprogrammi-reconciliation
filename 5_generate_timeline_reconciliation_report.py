import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from timeline_reconciliation_common import connect_motherduck, parse_config_txt


DEFAULT_OUTPUT = "timeline_reconciliation_report.xlsx"

HEADERS_LINKS = [
    "#",
    "TimelineName",
    "TaskRowId",
    "TaskCode",
    "TaskName",
    "WbsName",
    "TaskClass",
    "TaskClassConfidence",
    "TaskClassReason",
    "LinkRank",
    "LinkReason",
    "MdrDocumentTitle",
    "DocumentRaciTitle",
    "TaskStartDate",
    "TaskFinishDate",
    "TaskActualStartDate",
    "TaskActualFinishDate",
]

COL_WIDTHS_LINKS = [6, 28, 10, 16, 44, 30, 12, 14, 42, 9, 52, 42, 38, 18, 18, 18, 18]
HEADERS_TASKS = [
    "#",
    "TimelineName",
    "TaskRowId",
    "TaskCode",
    "TaskName",
    "WbsName",
    "TaskClass",
    "TaskClassConfidence",
    "TaskClassReason",
    "TaskStartDate",
    "TaskFinishDate",
    "TaskActualStartDate",
    "TaskActualFinishDate",
    "ResolverLinkCount",
]
COL_WIDTHS_TASKS = [6, 30, 10, 16, 54, 36, 12, 14, 52, 18, 18, 18, 18, 12]

NAVY = "0D1B2A"
WHITE = "FFFFFF"
GRID = "CBD5E1"

CLASS_BG = {"ENG_DOC": "DBEAFE", "OTHER": "F3F4F6"}
CLASS_FG = {"ENG_DOC": "1E3A8A", "OTHER": "374151"}
SECTION_COLORS = {
    "task": ("1E40AF", "EFF6FF"),
    "resolver": ("166534", "ECFDF5"),
    "raci": ("6D28D9", "F5F3FF"),
    "dates": ("475569", "F8FAFC"),
}

thin = Side(style="thin", color=GRID)


def _fill(color):
    return PatternFill("solid", start_color=color)


def _align(center=False):
    return Alignment(horizontal=("center" if center else "left"), vertical="top", wrap_text=True)


def _border():
    return Border(left=thin, right=thin, top=thin, bottom=thin)


def _fmt_ts(value):
    if value is None:
        return ""
    text = str(value)
    return text.replace("T", " ")


def _safe_text(value):
    if value is None:
        return ""
    return str(value)[:32767]


def load_links_rows(conn, db_name, timeline_name=None):
    timeline_filter = ""
    params = []
    if timeline_name:
        timeline_filter = "WHERE c.TimelineName = ?"
        params.append(timeline_name)

    sql = f"""
    WITH classified_latest AS (
        SELECT *
        FROM (
            SELECT
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.TimelineName, c.TaskRowId
                    ORDER BY c.UpdatedAt DESC, c.CreatedAt DESC
                ) AS rn
            FROM {db_name}.timeline_reconciliation.TimelineTasksClassified c
        ) x
        WHERE x.rn = 1
    ),
    links_latest AS (
        SELECT *
        FROM (
            SELECT
                l.*,
                ROW_NUMBER() OVER (
                    PARTITION BY l.TimelineName, l.TaskRowId, l.MdrTitleKey
                    ORDER BY l.CreatedAt DESC
                ) AS rn
            FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrLinks l
        ) y
        WHERE y.rn = 1
    ),
    link_counts AS (
        SELECT
            TimelineName,
            TaskRowId,
            COUNT(*) AS ResolverLinkCount
        FROM links_latest
        GROUP BY TimelineName, TaskRowId
    )
    SELECT
        c.TimelineName,
        c.ProjectCode,
        c.TaskRowId,
        c.TaskCode,
        c.TaskName,
        c.WbsName,
        c.TaskClass,
        c.TaskClassConfidence,
        c.TaskClassReason,
        c.TaskStartDate,
        c.TaskFinishDate,
        c.TaskActualStartDate,
        c.TaskActualFinishDate,
        c.TaskDateFieldsJson,
        COALESCE(lc.ResolverLinkCount, 0) AS ResolverLinkCount,
        l.MdrDocumentTitle AS LinkMdrDocumentTitle,
        l.ConsolidatedRaciTitle AS DocumentRaciTitle,
        l.LinkReason,
        l.LinkRank
    FROM classified_latest c
    LEFT JOIN link_counts lc
      ON lc.TimelineName = c.TimelineName
     AND lc.TaskRowId = c.TaskRowId
    LEFT JOIN links_latest l
      ON l.TimelineName = c.TimelineName
     AND l.TaskRowId = c.TaskRowId
    {timeline_filter}
    ORDER BY c.TimelineName, c.TaskRowId, l.LinkRank
    """
    df = conn.execute(sql, params).fetchdf()

    rows = []
    for _, row in df.iterrows():
        out = {
            "TimelineName": _safe_text(row.get("TimelineName")),
            "ProjectCode": _safe_text(row.get("ProjectCode")),
            "TaskRowId": int(row.get("TaskRowId")),
            "TaskCode": _safe_text(row.get("TaskCode")),
            "TaskName": _safe_text(row.get("TaskName")),
            "WbsName": _safe_text(row.get("WbsName")),
            "TaskClass": _safe_text(row.get("TaskClass")),
            "TaskClassConfidence": _safe_text(row.get("TaskClassConfidence")),
            "TaskClassReason": _safe_text(row.get("TaskClassReason")),
            "TaskStartDate": _fmt_ts(row.get("TaskStartDate")),
            "TaskFinishDate": _fmt_ts(row.get("TaskFinishDate")),
            "TaskActualStartDate": _fmt_ts(row.get("TaskActualStartDate")),
            "TaskActualFinishDate": _fmt_ts(row.get("TaskActualFinishDate")),
            "ResolverLinkCount": int(row.get("ResolverLinkCount") or 0),
            "LinkRank": "" if pd.isna(row.get("LinkRank")) else int(row.get("LinkRank")),
            "LinkReason": _safe_text(row.get("LinkReason")),
            "MdrDocumentTitle": _safe_text(row.get("LinkMdrDocumentTitle")),
            "DocumentRaciTitle": _safe_text(row.get("DocumentRaciTitle")),
        }
        rows.append(out)
    return rows


def build_task_summary_rows(link_rows):
    grouped = {}
    for r in link_rows:
        key = (r["TimelineName"], r["TaskRowId"])
        if key not in grouped:
            grouped[key] = {
                "TimelineName": r["TimelineName"],
                "TaskRowId": r["TaskRowId"],
                "TaskCode": r["TaskCode"],
                "TaskName": r["TaskName"],
                "WbsName": r["WbsName"],
                "TaskClass": r["TaskClass"],
                "TaskClassConfidence": r["TaskClassConfidence"],
                "TaskClassReason": r["TaskClassReason"],
                "TaskStartDate": r["TaskStartDate"],
                "TaskFinishDate": r["TaskFinishDate"],
                "TaskActualStartDate": r["TaskActualStartDate"],
                "TaskActualFinishDate": r["TaskActualFinishDate"],
                "ResolverLinkCount": r["ResolverLinkCount"],
            }
    return sorted(grouped.values(), key=lambda x: (x["TimelineName"], x["TaskRowId"]))


def _build_links_sheet(ws, title, rows):
    ws.merge_cells(f"A1:{get_column_letter(len(HEADERS_LINKS))}1")
    ws["A1"] = f"Timeline Reconciliation Report - {title} | rows: {len(rows)}"
    ws["A1"].font = Font(name="Arial", bold=True, size=11, color=WHITE)
    ws["A1"].fill = _fill(NAVY)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
    ws.row_dimensions[1].height = 24

    sections = [
        ("Task + Classify", 2, 9, "task"),
        ("Resolver Final Link", 10, 12, "resolver"),
        ("MDR to RACI Context", 13, 13, "raci"),
        ("Task Dates", 14, 17, "dates"),
    ]
    ws.cell(row=2, column=1, value="#")
    ws.merge_cells(start_row=2, start_column=1, end_row=3, end_column=1)
    cell = ws.cell(row=2, column=1)
    cell.font = Font(name="Arial", bold=True, size=9, color=WHITE)
    cell.fill = _fill("1A2E42")
    cell.alignment = _align(center=True)
    cell.border = _border()

    for label, start, end, key in sections:
        header_color, _ = SECTION_COLORS[key]
        ws.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)
        sc = ws.cell(row=2, column=start, value=label)
        sc.font = Font(name="Arial", bold=True, size=9, color=WHITE)
        sc.fill = _fill(header_color)
        sc.alignment = _align(center=True)
        for col in range(start, end + 1):
            ws.cell(row=2, column=col).border = _border()

    for idx, (header, width) in enumerate(zip(HEADERS_LINKS, COL_WIDTHS_LINKS), 1):
        if idx == 1:
            ws.column_dimensions[get_column_letter(idx)].width = width
            continue
        c = ws.cell(row=3, column=idx, value=header)
        section_key = "task" if idx <= 9 else "resolver" if idx <= 12 else "raci" if idx == 13 else "dates"
        header_color, _ = SECTION_COLORS[section_key]
        c.font = Font(name="Arial", bold=True, size=9, color=WHITE)
        c.fill = _fill(header_color)
        c.alignment = _align(center=True)
        c.border = _border()
        ws.column_dimensions[get_column_letter(idx)].width = width

    for i, row in enumerate(rows, 1):
        excel_row = i + 3
        task_class = row.get("TaskClass", "")
        class_bg = CLASS_BG.get(task_class, "F8FAFC")
        fg = CLASS_FG.get(task_class, "1A1A2E")

        values = [
            i,
            row["TimelineName"],
            row["TaskRowId"],
            row["TaskCode"],
            row["TaskName"],
            row["WbsName"],
            row["TaskClass"],
            row["TaskClassConfidence"],
            row["TaskClassReason"],
            row["LinkRank"],
            row["LinkReason"],
            row["MdrDocumentTitle"],
            row["DocumentRaciTitle"],
            row["TaskStartDate"],
            row["TaskFinishDate"],
            row["TaskActualStartDate"],
            row["TaskActualFinishDate"],
        ]
        for col_idx, value in enumerate(values, 1):
            c = ws.cell(row=excel_row, column=col_idx, value=_safe_text(value))
            c.border = _border()
            c.alignment = _align(center=col_idx in (1, 3, 7, 8, 10, 13, 14))
            if col_idx in (7, 8):
                c.fill = _fill(class_bg)
                c.font = Font(name="Arial", bold=True, size=9, color=fg)
            else:
                if col_idx <= 9:
                    _, cell_color = SECTION_COLORS["task"]
                elif col_idx <= 12:
                    _, cell_color = SECTION_COLORS["resolver"]
                elif col_idx == 13:
                    _, cell_color = SECTION_COLORS["raci"]
                else:
                    _, cell_color = SECTION_COLORS["dates"]
                c.fill = _fill(cell_color)
                c.font = Font(name="Arial", size=9, color="1A1A2E")

        ws.row_dimensions[excel_row].height = 72

    ws.auto_filter.ref = f"A3:{get_column_letter(len(HEADERS_LINKS))}{len(rows) + 3}"
    ws.freeze_panes = "A4"


def _build_tasks_sheet(ws, title, rows):
    ws.merge_cells(f"A1:{get_column_letter(len(HEADERS_TASKS))}1")
    ws["A1"] = f"Timeline Task Summary - {title} | tasks: {len(rows)}"
    ws["A1"].font = Font(name="Arial", bold=True, size=11, color=WHITE)
    ws["A1"].fill = _fill(NAVY)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
    ws.row_dimensions[1].height = 24

    for idx, (header, width) in enumerate(zip(HEADERS_TASKS, COL_WIDTHS_TASKS), 1):
        c = ws.cell(row=2, column=idx, value=header)
        c.font = Font(name="Arial", bold=True, size=9, color=WHITE)
        c.fill = _fill("1A2E42")
        c.alignment = _align(center=True)
        c.border = _border()
        ws.column_dimensions[get_column_letter(idx)].width = width

    for i, row in enumerate(rows, 1):
        excel_row = i + 2
        task_class = row.get("TaskClass", "")
        bg = CLASS_BG.get(task_class, "F8FAFC")
        fg = CLASS_FG.get(task_class, "1A1A2E")
        values = [i] + [row[h] for h in HEADERS_TASKS[1:]]
        for col_idx, value in enumerate(values, 1):
            c = ws.cell(row=excel_row, column=col_idx, value=_safe_text(value))
            c.border = _border()
            c.alignment = _align(center=col_idx in (1, 3, 7, 8, 14))
            if col_idx in (7, 8):
                c.fill = _fill(bg)
                c.font = Font(name="Arial", bold=True, size=9, color=fg)
            else:
                c.fill = _fill("FFFFFF")
                c.font = Font(name="Arial", size=9, color="1A1A2E")
        ws.row_dimensions[excel_row].height = 60

    ws.auto_filter.ref = f"A2:{get_column_letter(len(HEADERS_TASKS))}{len(rows) + 2}"
    ws.freeze_panes = "A3"


def write_report(link_rows, output_path):
    task_rows = build_task_summary_rows(link_rows)
    wb = Workbook()
    ws_links = wb.active
    ws_links.title = "Task-MDR Links"
    _build_links_sheet(ws_links, "Task-MDR Links", link_rows)

    eng_link_rows = [r for r in link_rows if r.get("TaskClass") == "ENG_DOC"]
    linked_rows = [r for r in link_rows if r.get("LinkRank", "") != ""]
    unlinked_rows = [r for r in link_rows if r.get("ResolverLinkCount", 0) == 0]

    _build_links_sheet(wb.create_sheet("ENG_DOC Links"), "ENG_DOC Links", eng_link_rows)
    _build_links_sheet(wb.create_sheet("Resolved Links"), "Resolved Links", linked_rows)
    _build_links_sheet(wb.create_sheet("Unlinked Tasks"), "Unlinked Tasks", unlinked_rows)
    _build_tasks_sheet(wb.create_sheet("Task Summary"), "Task Summary", task_rows)

    wb.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate timeline reconciliation Excel report (classify + resolver).")
    parser.add_argument("--timeline", default="", help="Optional TimelineName filter.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output xlsx path.")
    args = parser.parse_args()

    cfg = parse_config_txt()
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    timeline_name = args.timeline.strip() or None
    output_path = Path(args.output).resolve()

    conn = connect_motherduck(cfg)
    try:
        rows = load_links_rows(conn, db_name, timeline_name=timeline_name)
    finally:
        conn.close()

    write_report(rows, output_path)
    print(f"[OK] Report generated: {output_path}")
    task_rows = build_task_summary_rows(rows)
    print(f"Total link-view rows: {len(rows)}")
    print(f"Total tasks: {len(task_rows)}")
    print(f"ENG_DOC tasks: {sum(1 for r in task_rows if r.get('TaskClass') == 'ENG_DOC')}")
    print(f"OTHER tasks: {sum(1 for r in task_rows if r.get('TaskClass') == 'OTHER')}")
    print(f"Tasks with links: {sum(1 for r in task_rows if r.get('ResolverLinkCount', 0) > 0)}")


if __name__ == "__main__":
    main()
