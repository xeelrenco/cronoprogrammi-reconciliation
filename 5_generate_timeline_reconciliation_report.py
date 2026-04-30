import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from timeline_reconciliation_common import connect_motherduck, parse_config_txt


DEFAULT_OUTPUT = "timeline_reconciliation_report.xlsx"

HEADERS = [
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
    "Resolver Link 1",
    "Resolver Link 2",
    "Resolver Link 3",
    "Retrieval Top1",
    "Retrieval Top2",
    "Retrieval Top3",
]

COL_WIDTHS = [6, 34, 10, 16, 58, 44, 12, 14, 52, 18, 18, 18, 18, 12, 70, 70, 70, 56, 56, 56]

NAVY = "0D1B2A"
WHITE = "FFFFFF"
GRID = "CBD5E1"

CLASS_BG = {"ENG_DOC": "DBEAFE", "OTHER": "F3F4F6"}
CLASS_FG = {"ENG_DOC": "1E3A8A", "OTHER": "374151"}

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


def _link_text(row):
    return (
        f"MDR: {_safe_text(row['MdrDocumentTitle'])}\n"
        f"RACI: {_safe_text(row['ConsolidatedRaciTitle'])}\n"
        f"Score: {row['LinkScore'] if row['LinkScore'] is not None else '—'}\n"
        f"Reason: {_safe_text(row['LinkReason']) or '—'}"
    )


def _retrieval_text(row):
    return (
        f"MDR: {_safe_text(row['MdrDocumentTitle'])}\n"
        f"RACI: {_safe_text(row['ConsolidatedRaciTitle'])}\n"
        f"Similarity: {row['Similarity'] if row['Similarity'] is not None else '—'}"
    )


def load_report_rows(conn, db_name, timeline_name=None):
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
    candidates_latest AS (
        SELECT *
        FROM (
            SELECT
                k.*,
                ROW_NUMBER() OVER (
                    PARTITION BY k.TimelineName, k.TaskRowId, k.MdrTitleKey, k.Rank
                    ORDER BY k.CreatedAt DESC
                ) AS rn
            FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrCandidates k
            WHERE k.Rank <= 3
        ) z
        WHERE z.rn = 1
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
        l.MdrDocumentTitle AS LinkMdrDocumentTitle,
        l.ConsolidatedRaciTitle AS LinkConsolidatedRaciTitle,
        l.LinkScore,
        l.LinkReason,
        l.LinkRank,
        k.MdrDocumentTitle AS CandMdrDocumentTitle,
        k.ConsolidatedRaciTitle AS CandConsolidatedRaciTitle,
        k.Similarity,
        k.Rank AS CandRank
    FROM classified_latest c
    LEFT JOIN links_latest l
      ON l.TimelineName = c.TimelineName
     AND l.TaskRowId = c.TaskRowId
    LEFT JOIN candidates_latest k
      ON k.TimelineName = c.TimelineName
     AND k.TaskRowId = c.TaskRowId
    {timeline_filter}
    ORDER BY c.TimelineName, c.TaskRowId, l.LinkRank, k.Rank
    """
    df = conn.execute(sql, params).fetchdf()

    grouped = {}
    for _, row in df.iterrows():
        key = (str(row["TimelineName"]), int(row["TaskRowId"]))
        if key not in grouped:
            grouped[key] = {
                "TimelineName": str(row["TimelineName"]),
                "ProjectCode": str(row.get("ProjectCode", "") or ""),
                "TaskRowId": int(row["TaskRowId"]),
                "TaskCode": _safe_text(row.get("TaskCode", "")),
                "TaskName": _safe_text(row.get("TaskName", "")),
                "WbsName": _safe_text(row.get("WbsName", "")),
                "TaskClass": _safe_text(row.get("TaskClass", "")),
                "TaskClassConfidence": _safe_text(row.get("TaskClassConfidence", "")),
                "TaskClassReason": _safe_text(row.get("TaskClassReason", "")),
                "TaskStartDate": _fmt_ts(row.get("TaskStartDate")),
                "TaskFinishDate": _fmt_ts(row.get("TaskFinishDate")),
                "TaskActualStartDate": _fmt_ts(row.get("TaskActualStartDate")),
                "TaskActualFinishDate": _fmt_ts(row.get("TaskActualFinishDate")),
                "links": {},
                "retrievals": {},
            }

        link_rank = row.get("LinkRank")
        if link_rank is not None and not pd.isna(link_rank):
            rank = int(link_rank)
            if rank not in grouped[key]["links"] and rank <= 3:
                grouped[key]["links"][rank] = {
                    "MdrDocumentTitle": row.get("LinkMdrDocumentTitle"),
                    "ConsolidatedRaciTitle": row.get("LinkConsolidatedRaciTitle"),
                    "LinkScore": row.get("LinkScore"),
                    "LinkReason": row.get("LinkReason"),
                }

        cand_rank = row.get("CandRank")
        if cand_rank is not None and not pd.isna(cand_rank):
            rank = int(cand_rank)
            if rank not in grouped[key]["retrievals"] and rank <= 3:
                grouped[key]["retrievals"][rank] = {
                    "MdrDocumentTitle": row.get("CandMdrDocumentTitle"),
                    "ConsolidatedRaciTitle": row.get("CandConsolidatedRaciTitle"),
                    "Similarity": row.get("Similarity"),
                }

    rows = []
    for _, rec in sorted(grouped.items(), key=lambda x: (x[1]["TimelineName"], x[1]["TaskRowId"])):
        out = dict(rec)
        out["ResolverLinkCount"] = len(rec["links"])
        for rank in (1, 2, 3):
            out[f"Resolver Link {rank}"] = _link_text(rec["links"][rank]) if rank in rec["links"] else "—"
            out[f"Retrieval Top{rank}"] = _retrieval_text(rec["retrievals"][rank]) if rank in rec["retrievals"] else "—"
        rows.append(out)
    return rows


def _build_sheet(ws, title, rows):
    ws.merge_cells(f"A1:{get_column_letter(len(HEADERS))}1")
    ws["A1"] = f"Timeline Reconciliation Report - {title} | rows: {len(rows)}"
    ws["A1"].font = Font(name="Arial", bold=True, size=11, color=WHITE)
    ws["A1"].fill = _fill(NAVY)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
    ws.row_dimensions[1].height = 24

    for idx, (header, width) in enumerate(zip(HEADERS, COL_WIDTHS), 1):
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
            row["TaskStartDate"],
            row["TaskFinishDate"],
            row["TaskActualStartDate"],
            row["TaskActualFinishDate"],
            row["ResolverLinkCount"],
            row["Resolver Link 1"],
            row["Resolver Link 2"],
            row["Resolver Link 3"],
            row["Retrieval Top1"],
            row["Retrieval Top2"],
            row["Retrieval Top3"],
        ]
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

        ws.row_dimensions[excel_row].height = 88

    ws.auto_filter.ref = f"A2:{get_column_letter(len(HEADERS))}{len(rows) + 2}"
    ws.freeze_panes = "A3"


def write_report(rows, output_path):
    wb = Workbook()
    ws_all = wb.active
    ws_all.title = "All Tasks"
    _build_sheet(ws_all, "All Tasks", rows)

    eng_rows = [r for r in rows if r.get("TaskClass") == "ENG_DOC"]
    other_rows = [r for r in rows if r.get("TaskClass") == "OTHER"]
    linked_rows = [r for r in rows if r.get("ResolverLinkCount", 0) > 0]
    unlinked_eng = [r for r in eng_rows if r.get("ResolverLinkCount", 0) == 0]

    _build_sheet(wb.create_sheet("ENG_DOC"), "ENG_DOC", eng_rows)
    _build_sheet(wb.create_sheet("OTHER"), "OTHER", other_rows)
    _build_sheet(wb.create_sheet("Linked"), "Linked", linked_rows)
    _build_sheet(wb.create_sheet("ENG_DOC Unlinked"), "ENG_DOC Unlinked", unlinked_eng)

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
        rows = load_report_rows(conn, db_name, timeline_name=timeline_name)
    finally:
        conn.close()

    write_report(rows, output_path)
    print(f"[OK] Report generated: {output_path}")
    print(f"Total rows: {len(rows)}")
    print(f"ENG_DOC: {sum(1 for r in rows if r.get('TaskClass') == 'ENG_DOC')}")
    print(f"OTHER: {sum(1 for r in rows if r.get('TaskClass') == 'OTHER')}")
    print(f"Linked tasks: {sum(1 for r in rows if r.get('ResolverLinkCount', 0) > 0)}")


if __name__ == "__main__":
    main()
