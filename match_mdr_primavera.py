import pandas as pd
import re
from pathlib import Path

# ===== CONFIG =====
# Cartella con i file Excel dei cronoprogrammi (uno per progetto)
CRONOPROGRAMMI_DIR = Path(__file__).resolve().parent / "cronoprogrammi"
# File unico con tutte le righe MDR: A = MDR Ref, B = Doc. Number, C = Doc. Title
MDR_MAPPATURA_FILE = Path(__file__).resolve().parent / "Mappatura_MDR_passati.xlsx"
# Cartella in cui scrivere i file di output (uno per cronoprogramma + file statistiche aggregate)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

MDR_REF_COL = "MDR Ref"
DOC_NUMBER_COL = "Doc. Number"   # colonna B
DOC_TITLE_COL = "Doc. Title"    # colonna C
TASK_SHEET = "TASK"

# ===== FUZZY =====
try:
    from rapidfuzz import fuzz
    def sim(a, b):
        return fuzz.token_set_ratio(a, b)
except:
    from difflib import SequenceMatcher
    def sim(a, b):
        return int(100 * SequenceMatcher(None, a, b).ratio())

# ===== CLEAN =====
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_prefix(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    parts = text.split("-", 1)
    if len(parts) == 2 and len(parts[0]) <= 4:
        return parts[1].strip()
    return text

def _get(s, key):
    """Da una Series (riga DataFrame) restituisce il valore per key, o None se mancante/NaN."""
    if s is None:
        return None
    val = s.get(key)
    return None if pd.isna(val) else val

# ===== LOAD MDR MAPPATURA (una volta) =====
if not MDR_MAPPATURA_FILE.exists():
    raise FileNotFoundError(f"File MDR non trovato: {MDR_MAPPATURA_FILE}")
mdr_full = pd.read_excel(MDR_MAPPATURA_FILE, sheet_name=0)
mdr_full.columns = mdr_full.columns.str.strip()
if MDR_REF_COL not in mdr_full.columns:
    raise ValueError(f"Nel file MDR manca la colonna '{MDR_REF_COL}'. Colonne: {list(mdr_full.columns)}")
if DOC_TITLE_COL not in mdr_full.columns:
    raise ValueError(f"Nel file MDR manca la colonna '{DOC_TITLE_COL}'. Colonne: {list(mdr_full.columns)}")
# Colonna B = Doc. Number: se non esiste con questo nome, usa la seconda colonna (indice 1)
if DOC_NUMBER_COL not in mdr_full.columns and len(mdr_full.columns) >= 2:
    mdr_full[DOC_NUMBER_COL] = mdr_full.iloc[:, 1]
elif DOC_NUMBER_COL not in mdr_full.columns:
    mdr_full[DOC_NUMBER_COL] = ""
mdr_full[MDR_REF_COL] = mdr_full[MDR_REF_COL].fillna("").astype(str).str.strip()

# ===== ELENCO CRONOPROGRAMMI =====
if not CRONOPROGRAMMI_DIR.exists():
    raise FileNotFoundError(f"Cartella cronoprogrammi non trovata: {CRONOPROGRAMMI_DIR}")
cronoprogramma_files = sorted(CRONOPROGRAMMI_DIR.glob("*.xlsx"))
if not cronoprogramma_files:
    raise FileNotFoundError(f"Nessun file .xlsx in {CRONOPROGRAMMI_DIR}")


def run_matching(task: pd.DataFrame, mdr: pd.DataFrame) -> pd.DataFrame:
    """Esegue il matching tra task Primavera e righe MDR; restituisce il DataFrame risultati."""
    task = task.copy()
    task["name_clean"] = task["task_name"].apply(remove_prefix)
    task["name_norm"] = task["name_clean"].apply(normalize)
    mdr = mdr.copy()
    mdr["title_norm"] = mdr[DOC_TITLE_COL].fillna("").astype(str).apply(normalize)

    results = []
    for _, row in mdr.iterrows():
        title = row["title_norm"]
        doc_title = row[DOC_TITLE_COL] if pd.notna(row[DOC_TITLE_COL]) else ""
        doc_number = row.get(DOC_NUMBER_COL, "")
        if pd.isna(doc_number):
            doc_number = ""
        doc_number = str(doc_number).strip()

        if not title:
            results.append({
                "Doc. Number": doc_number,
                "Document Title": doc_title,
                "match_N": "0",
                "best_task_code": None,
                "best_task_name": None,
                "best_task_name_clean": None,
                "best_score": None,
                "category": "NO_TITLE"
            })
            continue

        scored = [(sim(title, t), i) for i, t in enumerate(task["name_norm"]) if t]
        if not scored:
            best_score = 0
            best_indices = []
        else:
            best_score = max(s[0] for s in scored)
            best_indices = [i for s, i in scored if s == best_score]

        if best_score >= 95:
            cat = "PERFECT"
        elif best_score >= 85:
            cat = "GOOD"
        elif best_score >= 70:
            cat = "MEDIUM"
        else:
            cat = "LOW"

        for rank, idx in enumerate(best_indices, 1):
            best_task = task.iloc[idx]
            results.append({
                "Doc. Number": doc_number,
                "Document Title": doc_title,
                "match_N": f"{rank} of {len(best_indices)}",
                "best_task_code": _get(best_task, "task_code"),
                "best_task_name": _get(best_task, "task_name"),
                "best_task_name_clean": _get(best_task, "name_clean"),
                "best_score": best_score,
                "category": cat
            })
        if not best_indices:
            results.append({
                "Doc. Number": doc_number,
                "Document Title": doc_title,
                "match_N": "0",
                "best_task_code": None,
                "best_task_name": None,
                "best_task_name_clean": None,
                "best_score": None,
                "category": cat
            })

    return pd.DataFrame(results)

def score_band(s):
    if pd.isna(s):
        return "NO_TITLE"
    s = int(s)
    if s >= 95:
        return "95-100 (PERFECT)"
    if s >= 85:
        return "85-94 (GOOD)"
    if s >= 70:
        return "70-84 (MEDIUM)"
    return "0-69 (LOW)"

BAND_ORDER = ["95-100 (PERFECT)", "85-94 (GOOD)", "70-84 (MEDIUM)", "0-69 (LOW)", "NO_TITLE"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
all_doc_dfs = []  # per statistiche aggregate a fine elaborazione

# ===== PER OGNI CRONOPROGRAMMA: MATCH + STATISTICHE + OUTPUT =====
for prim_file in cronoprogramma_files:
    ref_name = prim_file.name
    ref_stem = prim_file.stem
    # Filtra righe MDR che appartengono a questo cronoprogramma (MDR Ref = nome file o stem)
    mdr_subset = mdr_full[
        (mdr_full[MDR_REF_COL] == ref_name) | (mdr_full[MDR_REF_COL] == ref_stem)
    ].copy()
    if mdr_subset.empty:
        print(f"  Skip {ref_name}: nessuna riga MDR con MDR Ref = '{ref_name}' o '{ref_stem}'")
        continue

    try:
        task = pd.read_excel(prim_file, sheet_name=TASK_SHEET)
    except Exception as e:
        print(f"  Errore lettura {ref_name}: {e}")
        continue
    # Normalizza nomi colonne comuni (es. "Task Name" -> task_name)
    col_map = {}
    for c in task.columns:
        c2 = str(c).strip().lower().replace(" ", "_")
        if c2 in ("task_name", "task_code") and c != c2:
            col_map[c] = c2
    if col_map:
        task = task.rename(columns=col_map)
    if "task_name" not in task.columns:
        print(f"  Skip {ref_name}: foglio TASK senza colonna 'task_name'. Colonne: {list(task.columns)[:5]}...")
        continue

    df = run_matching(task, mdr_subset)
    df["score_band"] = df["best_score"].apply(score_band)
    df_doc = df.drop_duplicates(subset=["Document Title"], keep="first")
    df_doc = df_doc.copy()
    df_doc["cronoprogramma"] = ref_stem
    all_doc_dfs.append(df_doc)
    total = len(df_doc)

    stat_category = df_doc["category"].value_counts().rename_axis("category").reset_index(name="count")
    stat_category["pct"] = (stat_category["count"] / total * 100).round(2)

    scores = df_doc["best_score"].dropna()
    if len(scores) > 0:
        stat_score = pd.DataFrame([{"metric": "count", "value": len(scores)},
            {"metric": "mean", "value": round(scores.mean(), 2)},
            {"metric": "median", "value": round(scores.median(), 2)},
            {"metric": "min", "value": int(scores.min())},
            {"metric": "max", "value": int(scores.max())},
            {"metric": "std", "value": round(scores.std(), 2) if len(scores) > 1 else 0}])
    else:
        stat_score = pd.DataFrame(columns=["metric", "value"])

    stat_bands = df_doc["score_band"].value_counts().rename_axis("score_band").reset_index(name="count")
    stat_bands["pct"] = (stat_bands["count"] / total * 100).round(2)
    stat_bands["_order"] = stat_bands["score_band"].map(lambda b: BAND_ORDER.index(b) if b in BAND_ORDER else 99)
    stat_bands = stat_bands.sort_values("_order").drop(columns=["_order"])

    out_path = OUTPUT_DIR / f"matching_{ref_stem}.xlsx"
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.drop(columns=["score_band"], errors="ignore").to_excel(writer, sheet_name="Results", index=False)
            stat_category.to_excel(writer, sheet_name="Stat_category", index=False)
            stat_score.to_excel(writer, sheet_name="Stat_score", index=False)
            stat_bands.to_excel(writer, sheet_name="Stat_score_bands", index=False)
        print(f"Creato: {out_path}")
    except PermissionError:
        print(f"ERRORE: Impossibile scrivere '{out_path}'. Chiudi il file se aperto in Excel.")

# ===== STATISTICHE AGGREGATE (tutti i cronoprogrammi) =====
if all_doc_dfs:
    df_doc_all = pd.concat(all_doc_dfs, ignore_index=True)
    total_agg = len(df_doc_all)
    stat_cat_agg = df_doc_all["category"].value_counts().rename_axis("category").reset_index(name="count")
    stat_cat_agg["pct"] = (stat_cat_agg["count"] / total_agg * 100).round(2)
    scores_agg = df_doc_all["best_score"].dropna()
    if len(scores_agg) > 0:
        stat_score_agg = pd.DataFrame([
            {"metric": "count", "value": len(scores_agg)},
            {"metric": "mean", "value": round(scores_agg.mean(), 2)},
            {"metric": "median", "value": round(scores_agg.median(), 2)},
            {"metric": "min", "value": int(scores_agg.min())},
            {"metric": "max", "value": int(scores_agg.max())},
            {"metric": "std", "value": round(scores_agg.std(), 2) if len(scores_agg) > 1 else 0}
        ])
    else:
        stat_score_agg = pd.DataFrame(columns=["metric", "value"])
    stat_bands_agg = df_doc_all["score_band"].value_counts().rename_axis("score_band").reset_index(name="count")
    stat_bands_agg["pct"] = (stat_bands_agg["count"] / total_agg * 100).round(2)
    stat_bands_agg["_order"] = stat_bands_agg["score_band"].map(lambda b: BAND_ORDER.index(b) if b in BAND_ORDER else 99)
    stat_bands_agg = stat_bands_agg.sort_values("_order").drop(columns=["_order"])
    out_agg = OUTPUT_DIR / "statistiche_aggregate.xlsx"
    try:
        with pd.ExcelWriter(out_agg, engine="openpyxl") as writer:
            stat_cat_agg.to_excel(writer, sheet_name="Stat_category", index=False)
            stat_score_agg.to_excel(writer, sheet_name="Stat_score", index=False)
            stat_bands_agg.to_excel(writer, sheet_name="Stat_score_bands", index=False)
        print(f"Creato: {out_agg}")
    except PermissionError:
        print(f"ERRORE: Impossibile scrivere '{out_agg}'. Chiudi il file se aperto in Excel.")

print("Fatto.")