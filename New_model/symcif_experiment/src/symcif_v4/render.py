from __future__ import annotations

from typing import Any

from .orbit_engine import OrbitEngine


def render_record_cif(record: dict[str, Any], engine: OrbitEngine, data_name: str | None = None) -> str:
    free_params = {idx: dict(row.get("free_params") or {}) for idx, row in enumerate(record["wa_table"])}
    return engine.render_cif_from_wa_table(
        record["wa_table"],
        lattice=record["lattice"],
        free_params_by_row=free_params,
        formula_counts=record["formula_counts"],
        sg=int(record["sg"]),
        sg_symbol=str(record.get("sg_symbol") or ""),
        data_name=data_name or str(record.get("id") or record.get("sample_id") or "symcif_v4"),
    )

