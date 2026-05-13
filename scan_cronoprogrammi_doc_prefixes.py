"""
Scansiona tutti i .xlsx in cronoprogrammi/ e conta prefissi tipo IFC/IFD/IDC nei task_name.
Utile per definire regole in 1_classify_timeline_tasks.py.

Dopo l'esecuzione scrive un Excel compatto in output/: Riepilogo, Prefissi_rilevati, Altri_segnali.

Uso (dalla cartella del progetto):
  python scan_cronoprogrammi_doc_prefixes.py
  python scan_cronoprogrammi_doc_prefixes.py --top-tokens 120
  python scan_cronoprogrammi_doc_prefixes.py --output output/mio_report.xlsx
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from timeline_reconciliation_common import CRONOPROGRAMMI_DIR, OUTPUT_DIR, load_task_with_wbs

# Match all'inizio del nome task (dopo strip), case-insensitive
DOC_PREFIX_RE = re.compile(
    r"^(IFI\+|IFT\+|IFI|IFC|IFD|IFR|IFO|IFA|IFF|IFT|IDC|TRN|ASB)\b",
    re.I,
)


def clean_task_name(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1].strip()
    return s


def first_token(name: str) -> str:
    s = clean_task_name(name)
    if not s:
        return ""
    if " - " in s:
        head = s.split(" - ", 1)[0].strip()
        parts = head.split()
        return (parts[0] if parts else "").upper()[:24]
    parts = s.split()
    return (parts[0] if parts else "").upper()[:24]


def normalized_prefix(match: re.Match) -> str:
    return match.group(1).upper()


# Giudizio sui codici rilevati dallo scan (inizio nome task)
PREFIX_ENG_DOC: dict[str, tuple[str, str]] = {
    "IFI": ("Sì", "Issue for Information — emissione / revisione documenti."),
    "IFI+": ("Sì", "Variante IFI — stesso ambito documentale."),
    "IFC": ("Sì", "Issue for Construction — documenti per costruzione."),
    "IFD": ("Sì", "Issue for Design / acquisto-progettazione — specifiche e pacchetti doc."),
    "IFR": ("Sì", "Issue for Review — 1ª issue, revisione documenti."),
    "IFO": ("Sì", "Issue for Owner — revisione / approvazione lato cliente."),
    "IFA": ("Sì", "Issue for Approval — approvazione documenti."),
    "IFF": ("Sì", "Issue for Field (uso tipico EPC) — documentazione di cantiere / as-built in avvio."),
    "IFT": ("Sì", "Issue for Tender — documentazione per gara / commessa."),
    "IFT+": ("Sì", "Variante IFT — stesso ambito."),
    "IDC": ("Sì", "Internal / Inter-discipline check — controllo e coordinamento documenti."),
    "TRN": ("Sì", "Transmittal — trasmissione / consegna documentazione (DAP, DAT, ecc.)."),
    "ASB": ("Sì", "As-Built — documentazione di rilievo esecutivo."),
}

# Primi token frequenti che NON sono nel regex ma possono indicare attività su documenti (sempre da incrociare con il testo)
OPTIONAL_DOC_HINTS: dict[str, tuple[str, str]] = {
    "AB": ("Valuta", "Spesso As-Built o studi/verifiche; può essere altro (es. abbreviazioni)."),
    "APP": ("Valuta", "Spesso pacchetti di approvazione / applicazioni su documenti."),
    "RFC": ("Valuta", "Richiesta commenti su documentazione tecnica."),
    "TE": ("Valuta", "Technical evaluation — spesso valutazione su specifiche / offerta tecnica."),
    "TBE": ("Valuta", "Technical bid evaluation — legato a valutazione offerte (non solo doc MDR)."),
    "EPA": ("Valuta", "Sessioni EPA su studi di processo / documentazione."),
    "BCO": ("Valuta", "Spesso documenti / baseline (dipende da standard progetto)."),
    "INT": ("Valuta", "Interference / integrazione documenti (se usato così nel progetto)."),
}


def write_excel_report_compact(
    out_path: Path,
    *,
    cronoprogrammi_dir: Path,
    n_files: int,
    n_skipped: int,
    total_tasks: int,
    matched: int,
    prefix_hits: Counter[str],
    first_token_counts: Counter[str],
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pct = round(100.0 * matched / total_tasks, 2) if total_tasks else 0.0
    df_summary = pd.DataFrame(
        [
            ("Cartella cronoprogrammi", str(cronoprogrammi_dir.resolve())),
            ("Task con nome analizzati", str(total_tasks)),
            ("Task con prefisso doc in testa al nome", f"{matched} ({pct}%)"),
            ("File cronogrammi letti", str(n_files)),
            ("File non letti", str(n_skipped)),
            ("Generato", time.strftime("%Y-%m-%d %H:%M:%S")),
        ],
        columns=["Voce", "Valore"],
    )

    rows_pref = []
    for code, cnt in prefix_hits.most_common():
        eng, nota = PREFIX_ENG_DOC.get(code, ("—", "Codice non mappato nelle note standard."))
        rows_pref.append(
            {
                "Prefisso_rilevato": code,
                "N_task": int(cnt),
                "Riconducibile_doc_ingegneria": eng,
                "Nota": nota,
            }
        )
    df_pref = pd.DataFrame(rows_pref)

    rows_opt = []
    for tok, (eng, nota) in OPTIONAL_DOC_HINTS.items():
        c = int(first_token_counts.get(tok, 0))
        if c <= 0:
            continue
        rows_opt.append(
            {
                "Token_iniziale": tok,
                "N_task": c,
                "Riconducibile_doc_ingegneria": eng,
                "Nota": nota,
            }
        )
    rows_opt.sort(key=lambda x: -x["N_task"])
    df_opt = pd.DataFrame(rows_opt)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Riepilogo", index=False)
        df_pref.to_excel(writer, sheet_name="Prefissi_rilevati", index=False)
        if not df_opt.empty:
            df_opt.to_excel(writer, sheet_name="Altri_segnali", index=False)
        else:
            pd.DataFrame(
                [{"Messaggio": "Nessuno dei token opzionali monitorati è presente nei dati."}]
            ).to_excel(writer, sheet_name="Altri_segnali", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--top-tokens",
        type=int,
        default=80,
        help="Solo console: quanti primi token stampare in discovery (default 80).",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Percorso .xlsx (default: output/scan_cronoprogrammi_doc_prefixes_<timestamp>.xlsx).",
    )
    ap.add_argument("--no-excel", action="store_true", help="Non scrivere il file Excel.")
    args = ap.parse_args()

    files = sorted(CRONOPROGRAMMI_DIR.glob("*.xlsx"))
    if not files:
        print(f"Nessun file .xlsx trovato in: {CRONOPROGRAMMI_DIR.resolve()}")
        print("Verifica che la cartella esista e contenga i cronogrammi.")
        return 1

    prefix_hits: Counter[str] = Counter()
    first_token_counts: Counter[str] = Counter()
    per_file_counts: dict[str, tuple[int, int]] = {}
    total_tasks = 0
    matched = 0
    files_processed: list[str] = []
    skipped: list[tuple[str, str]] = []

    for path in files:
        try:
            df = load_task_with_wbs(path)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            skipped.append((path.name, str(exc)))
            continue
        files_processed.append(path.name)
        n_local = 0
        m_local = 0
        for _, row in df.iterrows():
            name = clean_task_name(str(row.get("task_name", "") or ""))
            if not name:
                continue
            total_tasks += 1
            n_local += 1
            m = DOC_PREFIX_RE.match(name)
            if m:
                matched += 1
                m_local += 1
                code = normalized_prefix(m)
                prefix_hits[code] += 1
            ft = first_token(name)
            if ft:
                first_token_counts[ft] += 1
        per_file_counts[path.name] = (n_local, m_local)

    print(f"Cartella: {CRONOPROGRAMMI_DIR.resolve()}")
    print(f"File .xlsx: {len(files)}")
    print(f"Task totali (con task_name non vuoto): {total_tasks}")
    print(f"Task con prefisso doc noto in testa al nome: {matched}")
    print()
    print("--- Per file (task | con prefisso) ---")
    for fname, (n, m) in sorted(per_file_counts.items()):
        print(f"  {fname}: {n} | {m}")
    print()
    print("--- Conteggio per codice (inizio stringa, case-insensitive) ---")
    for k, v in prefix_hits.most_common():
        print(f"  {k:6}  {v}")
    print()
    print(f"--- Top {args.top_tokens} primi token (discovery) ---")
    for k, v in first_token_counts.most_common(args.top_tokens):
        print(f"  {k:24}  {v}")

    if not args.no_excel:
        ts = time.strftime("%Y%m%d_%H%M%S")
        default_name = OUTPUT_DIR / f"scan_cronoprogrammi_doc_prefixes_{ts}.xlsx"
        out_path = Path(args.output.strip()) if args.output.strip() else default_name
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        write_excel_report_compact(
            out_path,
            cronoprogrammi_dir=CRONOPROGRAMMI_DIR,
            n_files=len(files_processed),
            n_skipped=len(skipped),
            total_tasks=total_tasks,
            matched=matched,
            prefix_hits=prefix_hits,
            first_token_counts=first_token_counts,
        )
        print()
        print(f"[OK] Report Excel: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
