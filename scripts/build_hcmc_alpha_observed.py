"""Build an observed HCMC traffic-factor sidecar from speed observations.

The source observations use OSM ``way_id`` values.  This script deliberately
does not guess a mapping to the synthetic HCMC graph's E00001-style edges.
That mapping needs geometry/map matching first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "timestamp", "way_id", "currentSpeed", "freeFlowSpeed",
    "currentTravelTime", "freeFlowTravelTime", "confidence", "roadClosure",
}


def build(source: Path, output: Path) -> None:
    frame = pd.read_parquet(source)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    numeric = [
        "way_id", "currentSpeed", "freeFlowSpeed", "currentTravelTime",
        "freeFlowTravelTime", "confidence",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        frame["way_id"].notna()
        & frame["timestamp"].notna()
        & (frame["currentSpeed"] > 0)
        & (frame["freeFlowSpeed"] > 0)
        & (frame["currentTravelTime"] > 0)
        & (frame["freeFlowTravelTime"] > 0)
    )
    frame = frame.loc[valid].copy()
    frame["alpha_speed"] = (
        frame["freeFlowSpeed"] / frame["currentSpeed"]
    ).clip(lower=1.0)
    frame["alpha_time"] = (
        frame["currentTravelTime"] / frame["freeFlowTravelTime"]
    ).clip(lower=1.0)
    frame["usable"] = (~frame["roadClosure"].astype(bool)) & (frame["confidence"] >= 0.8)
    frame["way_id"] = frame["way_id"].astype("int64")
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    columns = [
        "timestamp", "way_id", "name", "lat", "lon", "frc",
        "currentSpeed", "freeFlowSpeed", "currentTravelTime",
        "freeFlowTravelTime", "alpha_speed", "alpha_time", "confidence",
        "roadClosure", "usable", "version", "zoom", "snap_dist_m",
    ]
    frame = frame[[column for column in columns if column in frame.columns]]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)

    metadata = {
        "format": "hcmc-observed-traffic-factor",
        "format_version": 1,
        "source": str(source),
        "output": str(output),
        "rows": int(len(frame)),
        "way_count": int(frame["way_id"].nunique()),
        "alpha_definition": "max(1, freeFlowSpeed/currentSpeed)",
        "alpha_time_definition": "max(1, currentTravelTime/freeFlowTravelTime)",
        "usable_definition": "not roadClosure and confidence >= 0.8",
        "mapping_status": "OSM way_id not mapped to synthetic E00001-style graph edges",
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/vantoctb/vietnam-road-traffic-observations.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hcmc/sidecars/hcmc.alpha-observed.parquet"),
    )
    args = parser.parse_args()
    build(args.source, args.output)
