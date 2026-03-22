"""
Scraper: SiAM Tiburones Colombia
Fuente: iNaturalist proyecto "tiburones-rayas-y-quimeras-de-colombia"
API: https://api.inaturalist.org/v1/

Genera dos archivos en OUTPUT_DIR:
  - siam_avistamientos.txt   : observaciones ciudadanas (lugar, fecha, especie)
  - siam_especies.txt        : fichas de taxonomía y conservación por especie
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict

PROJECT_ID = "tiburones-rayas-y-quimeras-de-colombia"
OUTPUT_DIR = Path(__file__).parent.parent / "docs"
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "RAG-Buconos/1.0 (research; contact@example.com)",
    "Accept": "application/json",
}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=20)
    return json.loads(resp.read())


def fetch_observations() -> list[dict]:
    """Descarga todas las observaciones del proyecto (paginado)."""
    all_obs = []
    page = 1
    per_page = 200
    while True:
        url = (
            f"https://api.inaturalist.org/v1/observations"
            f"?project_id={PROJECT_ID}"
            f"&per_page={per_page}&page={page}"
            f"&order=created_at&order_by=desc"
        )
        data = get(url)
        batch = data.get("results", [])
        all_obs.extend(batch)
        print(f"  Página {page}: {len(batch)} obs (total {len(all_obs)}/{data['total_results']})")
        if len(all_obs) >= data["total_results"] or not batch:
            break
        page += 1
        time.sleep(0.5)
    return all_obs


def fetch_taxon_info(taxon_id: int) -> dict:
    """Obtiene descripción y estado de conservación de un taxón."""
    try:
        data = get(f"https://api.inaturalist.org/v1/taxa/{taxon_id}")
        return data["results"][0] if data.get("results") else {}
    except Exception:
        return {}


def conservation_status(taxon: dict) -> str:
    cs = taxon.get("conservation_status") or {}
    return cs.get("status_name") or cs.get("status") or "No evaluado"


def format_observation(obs: dict) -> str:
    taxon    = obs.get("taxon") or {}
    name_es  = (taxon.get("preferred_common_name") or taxon.get("name") or obs.get("species_guess") or "Especie desconocida")
    name_sci = taxon.get("name") or ""
    place    = obs.get("place_guess") or "Lugar no especificado"
    date     = obs.get("observed_on") or obs.get("created_at", "")[:10]
    quality  = obs.get("quality_grade", "casual")
    desc     = obs.get("description") or ""
    coords   = ""
    if obs.get("location"):
        lat, lon = obs["location"].split(",")
        coords = f"Coordenadas: {float(lat):.4f}, {float(lon):.4f}"

    lines = [
        f"Avistamiento #{obs['id']}",
        f"Especie: {name_es}" + (f" ({name_sci})" if name_sci else ""),
        f"Lugar: {place}",
        f"Fecha: {date}",
        f"Calidad: {quality}",
    ]
    if coords:
        lines.append(coords)
    if desc.strip():
        lines.append(f"Descripción: {desc.strip()[:400]}")
    return "\n".join(lines)


def build_species_sheet(taxon: dict, obs_list: list[dict]) -> str:
    name_es  = taxon.get("preferred_common_name") or taxon.get("name", "")
    name_sci = taxon.get("name", "")
    rank     = taxon.get("rank", "")
    anc      = " > ".join(
        a.get("name", "") for a in (taxon.get("ancestors") or [])
        if a.get("rank") in ("class", "order", "family")
    )
    conseq   = conservation_status(taxon)
    wiki     = taxon.get("wikipedia_summary") or ""
    places   = sorted({o.get("place_guess", "") for o in obs_list if o.get("place_guess")})
    dates    = sorted(o.get("observed_on", "") for o in obs_list if o.get("observed_on"))

    lines = [
        f"=== {name_es} ({name_sci}) ===",
        f"Rango taxonómico: {rank}",
    ]
    if anc:
        lines.append(f"Clasificación: {anc}")
    lines.append(f"Estado de conservación: {conseq}")
    lines.append(f"Observaciones en Colombia: {len(obs_list)}")
    if places:
        lines.append(f"Lugares registrados: {', '.join(places[:10])}")
    if dates:
        lines.append(f"Período de avistamientos: {dates[0]} a {dates[-1]}")
    if wiki:
        lines.append(f"\nDescripción:\n{wiki[:1000]}")
    return "\n".join(lines)


def main():
    print("▶ Descargando observaciones de iNaturalist...")
    observations = fetch_observations()
    print(f"  Total: {len(observations)} observaciones\n")

    # Agrupar por taxón
    by_taxon: dict[int, list] = defaultdict(list)
    taxon_obj: dict[int, dict] = {}
    for obs in observations:
        t = obs.get("taxon") or {}
        tid = t.get("id")
        if tid:
            by_taxon[tid].append(obs)
            if tid not in taxon_obj:
                taxon_obj[tid] = t

    # ── Archivo 1: avistamientos ────────────────────────────────────────────
    print("▶ Generando siam_avistamientos.txt...")
    lines_obs = [
        "AVISTAMIENTOS DE TIBURONES, RAYAS Y QUIMERAS EN COLOMBIA",
        f"Fuente: iNaturalist · proyecto {PROJECT_ID}",
        f"Total observaciones: {len(observations)}",
        "=" * 60, "",
    ]
    for obs in observations:
        lines_obs.append(format_observation(obs))
        lines_obs.append("")

    obs_path = OUTPUT_DIR / "siam_avistamientos.txt"
    obs_path.write_text("\n".join(lines_obs), encoding="utf-8")
    print(f"  Guardado: {obs_path} ({obs_path.stat().st_size // 1024} KB)")

    # ── Archivo 2: fichas de especie ────────────────────────────────────────
    print(f"\n▶ Generando siam_especies.txt ({len(by_taxon)} especies)...")
    lines_sp = [
        "FICHAS DE ESPECIES — TIBURONES, RAYAS Y QUIMERAS DE COLOMBIA",
        "Fuente: iNaturalist API · datos de observaciones en Colombia",
        "=" * 60, "",
    ]
    for i, (tid, obs_list) in enumerate(sorted(by_taxon.items(), key=lambda x: -len(x[1]))):
        t = taxon_obj[tid]
        # Enriquecer con datos completos del taxón (wiki, conservación)
        if i < 40:  # primeras 40 especies más observadas
            print(f"  [{i+1}/{min(40,len(by_taxon))}] {t.get('name','?')} ({len(obs_list)} obs)")
            full = fetch_taxon_info(tid)
            if full:
                t = {**t, **full}
            time.sleep(0.3)
        lines_sp.append(build_species_sheet(t, obs_list))
        lines_sp.append("")

    sp_path = OUTPUT_DIR / "siam_especies.txt"
    sp_path.write_text("\n".join(lines_sp), encoding="utf-8")
    print(f"  Guardado: {sp_path} ({sp_path.stat().st_size // 1024} KB)")

    print("\n✅ Listo. Archivos listos para indexar en el RAG.")
    print(f"   {obs_path}")
    print(f"   {sp_path}")


if __name__ == "__main__":
    main()
