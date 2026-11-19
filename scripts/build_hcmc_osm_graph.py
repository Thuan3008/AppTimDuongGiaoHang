"""Build the project's local road-graph JSON from an OpenStreetMap PBF/XML file.

The converter performs two passes.  Pass one finds intersections and way
endpoints.  Pass two collapses intermediate geometry-only OSM nodes while
preserving accumulated road distance.  Finally it keeps the largest connected
component (strong by default) so the TDVRP router does not retain isolated road
fragments.

Example:

    .venv\\Scripts\\python.exe scripts\\build_hcmc_osm_graph.py \
        --input data\\hcmc\\osm\\vietnam-latest.osm.pbf \
        --output data\\hcmc\\sidecars\\hcmc.road-osm.json

For an exact administrative boundary, pass a Polygon/MultiPolygon GeoJSON via
``--boundary-geojson``.  Without one, the default HCMC study bounding box is
used (west,south,east,north = 106.35,10.35,107.05,11.20).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

try:
    import osmium
except ImportError as error:  # pragma: no cover - exercised only without the optional tool dependency
    raise SystemExit(
        "Missing dependency 'osmium'. Install it with: "
        ".venv\\Scripts\\python.exe -m pip install osmium"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "hcmc" / "osm" / "vietnam-latest.osm.pbf"
DEFAULT_OUTPUT = ROOT / "data" / "hcmc" / "sidecars" / "hcmc.road-osm.json"
DEFAULT_BBOX = (106.35, 10.35, 107.05, 11.20)

DRIVABLE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
}
BLOCKED_ACCESS = {"no", "private"}
ONEWAY_TRUE = {"yes", "true", "1"}
ONEWAY_FALSE = {"no", "false", "0"}
DEFAULT_SPEED_KMH = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 60.0,
    "trunk_link": 45.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 40.0,
    "secondary_link": 35.0,
    "tertiary": 35.0,
    "tertiary_link": 30.0,
    "unclassified": 30.0,
    "residential": 25.0,
    "living_street": 15.0,
    "service": 15.0,
    "road": 25.0,
}


@dataclass(frozen=True)
class Point:
    node_id: int
    lat: float
    lon: float


@dataclass(frozen=True)
class WayProfile:
    way_id: int
    road_name: str
    road_class: str
    speed_kmh: float
    direction: int  # 0=bidirectional, 1=OSM order, -1=reverse OSM order


class Region:
    """Fast bounding-box filter with optional exact GeoJSON polygons."""

    def __init__(self, bbox: tuple[float, float, float, float], polygons=None, source="bbox"):
        west, south, east, north = bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox must be west,south,east,north with valid coordinates")
        self.bbox = bbox
        self.polygons = polygons or []
        self.source = source

    def contains(self, lat: float, lon: float) -> bool:
        west, south, east, north = self.bbox
        if not (west <= lon <= east and south <= lat <= north):
            return False
        if not self.polygons:
            return True
        return any(_point_in_ring(lon, lat, rings[0])
                   and not any(_point_in_ring(lon, lat, hole) for hole in rings[1:])
                   for rings in self.polygons)

    @classmethod
    def from_geojson(cls, path: Path) -> "Region":
        payload = json.loads(path.read_text(encoding="utf-8"))
        geometries = []
        if payload.get("type") == "FeatureCollection":
            geometries = [feature.get("geometry", {}) for feature in payload.get("features", [])]
        elif payload.get("type") == "Feature":
            geometries = [payload.get("geometry", {})]
        else:
            geometries = [payload]
        polygons = []
        for geometry in geometries:
            if geometry.get("type") == "Polygon":
                polygons.append(geometry.get("coordinates", []))
            elif geometry.get("type") == "MultiPolygon":
                polygons.extend(geometry.get("coordinates", []))
        polygons = [rings for rings in polygons if rings and len(rings[0]) >= 4]
        if not polygons:
            raise ValueError("boundary GeoJSON must contain a Polygon or MultiPolygon")
        coordinates = [coordinate for rings in polygons for ring in rings for coordinate in ring]
        bbox = (min(point[0] for point in coordinates), min(point[1] for point in coordinates),
                max(point[0] for point in coordinates), max(point[1] for point in coordinates))
        return cls(bbox, polygons, source=str(path))


def _point_in_ring(x: float, y: float, ring) -> bool:
    inside = False
    previous_x, previous_y = ring[-1][:2]
    for coordinate in ring:
        current_x, current_y = coordinate[:2]
        if ((current_y > y) != (previous_y > y)):
            crossing_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox must contain four numbers") from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    try:
        Region(values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return values


def haversine_m(first: Point, second: Point) -> float:
    radius = 6_371_008.8
    lat1, lat2 = math.radians(first.lat), math.radians(second.lat)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(second.lon - first.lon)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def parse_speed(value: str | None) -> float | None:
    if not value:
        return None
    speeds = []
    for part in value.split(";"):
        match = re.search(r"\d+(?:[.,]\d+)?", part)
        if not match:
            continue
        speed = float(match.group(0).replace(",", "."))
        if "mph" in part.casefold():
            speed *= 1.609344
        if math.isfinite(speed) and speed > 0:
            speeds.append(speed)
    return min(speeds) if speeds else None


def way_profile(way) -> WayProfile | None:
    tags = way.tags
    road_class = tags.get("highway")
    if road_class not in DRIVABLE_HIGHWAYS:
        return None
    if any(tags.get(key, "").casefold() in BLOCKED_ACCESS for key in ("access", "vehicle", "motor_vehicle")):
        return None
    oneway = tags.get("oneway", "").casefold()
    if oneway == "-1" or oneway == "reverse":
        direction = -1
    elif oneway in ONEWAY_TRUE or (oneway not in ONEWAY_FALSE and road_class == "motorway") \
            or (oneway not in ONEWAY_FALSE and tags.get("junction", "").casefold() == "roundabout"):
        direction = 1
    else:
        direction = 0
    directional_key = "maxspeed:backward" if direction == -1 else "maxspeed:forward"
    speed_candidates = [parse_speed(tags.get(directional_key)), parse_speed(tags.get("maxspeed"))]
    if direction == 0:
        speed_candidates.extend((parse_speed(tags.get("maxspeed:forward")),
                                 parse_speed(tags.get("maxspeed:backward"))))
    valid_speeds = [speed for speed in speed_candidates if speed is not None]
    speed = min(valid_speeds) if valid_speeds else DEFAULT_SPEED_KMH[road_class]
    speed = min(130.0, max(5.0, speed))
    name = tags.get("name") or tags.get("name:vi") or tags.get("ref") or f"OSM way {way.id}"
    return WayProfile(int(way.id), str(name), road_class, speed, direction)


def way_runs(way, region: Region) -> Iterator[list[Point]]:
    current = []
    for node in way.nodes:
        try:
            if not node.location.valid():
                inside = False
            else:
                point = Point(int(node.ref), float(node.lat), float(node.lon))
                inside = region.contains(point.lat, point.lon)
        except (RuntimeError, ValueError):
            inside = False
        if inside:
            current.append(point)
        else:
            if len(current) >= 2:
                yield current
            current = []
    if len(current) >= 2:
        yield current


class TopologyPass(osmium.SimpleHandler):
    def __init__(self, region: Region, progress_every: int):
        super().__init__()
        self.region = region
        self.progress_every = progress_every
        self.node_way_count: dict[int, int] = defaultdict(int)
        self.anchors: set[int] = set()
        self.drivable_ways = 0
        self.total_ways = 0

    def way(self, way):
        self.total_ways += 1
        profile = way_profile(way)
        if profile is None:
            return
        found = False
        for run in way_runs(way, self.region):
            found = True
            for node_id in {point.node_id for point in run}:
                self.node_way_count[node_id] += 1
            self.anchors.add(run[0].node_id)
            self.anchors.add(run[-1].node_id)
            if run[0].node_id == run[-1].node_id and len(run) > 3:
                self.anchors.add(run[len(run) // 2].node_id)
        if found:
            self.drivable_ways += 1
            if self.progress_every and self.drivable_ways % self.progress_every == 0:
                print(f"Pass 1: {self.drivable_ways:,} drivable ways in region", flush=True)

    def final_anchors(self) -> set[int]:
        return self.anchors | {node_id for node_id, count in self.node_way_count.items() if count > 1}


class EdgePass(osmium.SimpleHandler):
    def __init__(self, region: Region, anchors: set[int], progress_every: int):
        super().__init__()
        self.region = region
        self.anchors = anchors
        self.progress_every = progress_every
        self.nodes: dict[int, dict] = {}
        self.edges: list[dict] = []
        self.drivable_ways = 0

    def way(self, way):
        profile = way_profile(way)
        if profile is None:
            return
        found = False
        segment_number = 0
        for run in way_runs(way, self.region):
            found = True
            start = run[0]
            distance = 0.0
            for previous, current in zip(run, run[1:]):
                distance += haversine_m(previous, current)
                if current.node_id not in self.anchors and current.node_id != run[-1].node_id:
                    continue
                if start.node_id != current.node_id and distance > 0.01:
                    source, target = (current, start) if profile.direction == -1 else (start, current)
                    self.nodes[source.node_id] = {"id": source.node_id, "lat": source.lat, "lon": source.lon}
                    self.nodes[target.node_id] = {"id": target.node_id, "lat": target.lat, "lon": target.lon}
                    self.edges.append({
                        "id": f"W{profile.way_id}:{segment_number}",
                        "from": source.node_id,
                        "to": target.node_id,
                        "road_name": profile.road_name,
                        "distance_m": round(distance, 3),
                        "free_flow_speed_kmh": round(profile.speed_kmh, 3),
                        "bidirectional": profile.direction == 0,
                        "road_class": profile.road_class,
                        "osm_way_ids": [profile.way_id],
                    })
                    segment_number += 1
                start = current
                distance = 0.0
        if found:
            self.drivable_ways += 1
            if self.progress_every and self.drivable_ways % self.progress_every == 0:
                print(f"Pass 2: {self.drivable_ways:,} ways, {len(self.edges):,} simplified edges", flush=True)


def largest_component(nodes: dict[int, dict], edges: list[dict], mode: str) -> set[int]:
    if not edges:
        raise ValueError("No road edges found inside the selected region")
    adjacency: dict[int, list[int]] = defaultdict(list)
    reverse: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        source, target = edge["from"], edge["to"]
        adjacency[source].append(target)
        reverse[target].append(source)
        if edge["bidirectional"] or mode == "weak":
            adjacency[target].append(source)
            reverse[source].append(target)
    if mode == "weak":
        unseen = set(nodes)
        largest: set[int] = set()
        while unseen:
            root = next(iter(unseen))
            component = {root}
            stack = [root]
            unseen.remove(root)
            while stack:
                current = stack.pop()
                for other in adjacency[current]:
                    if other in unseen:
                        unseen.remove(other)
                        component.add(other)
                        stack.append(other)
            if len(component) > len(largest):
                largest = component
        return largest

    seen: set[int] = set()
    order: list[int] = []
    for root in nodes:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                order.append(current)
                continue
            stack.append((current, True))
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append((other, False))
    assigned: set[int] = set()
    largest = set()
    for root in reversed(order):
        if root in assigned:
            continue
        component = {root}
        stack = [root]
        assigned.add(root)
        while stack:
            current = stack.pop()
            for other in reverse[current]:
                if other not in assigned:
                    assigned.add(other)
                    component.add(other)
                    stack.append(other)
        if len(component) > len(largest):
            largest = component
    return largest


def build_graph(input_path: Path, output_path: Path, region: Region, *, component_mode="strong",
                progress_every=10_000, location_index="flex_mem") -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"OSM input not found: {input_path}")
    started = time.perf_counter()
    topology = TopologyPass(region, progress_every)
    topology.apply_file(str(input_path), locations=True, idx=location_index)
    anchors = topology.final_anchors()
    if not anchors:
        raise ValueError("No drivable roads found inside the selected region")
    print(f"Pass 1 complete: {topology.drivable_ways:,} ways, {len(anchors):,} routing anchors", flush=True)

    materializer = EdgePass(region, anchors, progress_every)
    materializer.apply_file(str(input_path), locations=True, idx=location_index)
    print(f"Pass 2 complete: {len(materializer.nodes):,} nodes, {len(materializer.edges):,} edges", flush=True)

    component = largest_component(materializer.nodes, materializer.edges, component_mode)
    nodes = [materializer.nodes[node_id] for node_id in sorted(component)]
    edges = [edge for edge in materializer.edges
             if edge["from"] in component and edge["to"] in component]
    if not nodes or not edges:
        raise ValueError("Largest connected component is empty")
    removed_nodes = len(materializer.nodes) - len(nodes)
    removed_edges = len(materializer.edges) - len(edges)
    payload = {
        "schema_version": "1.0",
        "region": "hcmc-osm-road-network",
        "source": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary": {"source": region.source, "bbox": list(region.bbox)},
        "component_policy": f"largest-{component_mode}ly-connected",
        "simplification": "OSM way endpoints and shared junctions; geometry-only nodes collapsed",
        "stats": {
            "source_drivable_ways": materializer.drivable_ways,
            "nodes": len(nodes),
            "edges": len(edges),
            "removed_component_nodes": removed_nodes,
            "removed_component_edges": removed_edges,
            "build_seconds": round(time.perf_counter() - started, 3),
        },
        "nodes": nodes,
        "edges": edges,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    try:
        temporary.replace(output_path)
    except PermissionError:
        # Some Windows sandbox/antivirus setups deny os.replace even when both
        # paths share a directory.  Keep a portable fallback for the CLI.
        shutil.copyfile(temporary, output_path)
        temporary.unlink()
    print(f"Written {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
    print(f"Kept {len(nodes):,} nodes and {len(edges):,} edges; removed {removed_nodes:,}/{removed_edges:,}", flush=True)
    return payload


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Convert OSM PBF/XML roads to the RAPO HCMC road-graph JSON")
    command.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source .osm.pbf or .osm XML")
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination road-graph JSON")
    command.add_argument("--bbox", type=parse_bbox, default=DEFAULT_BBOX,
                         help="west,south,east,north (default: HCMC study bounds)")
    command.add_argument("--boundary-geojson", type=Path,
                         help="Optional exact Polygon/MultiPolygon GeoJSON; overrides bbox")
    command.add_argument("--component", choices=("strong", "weak"), default="strong",
                         help="Connected component policy (strong guarantees directed reachability)")
    command.add_argument("--progress-every", type=int, default=10_000,
                         help="Print progress after this many retained ways; 0 disables")
    command.add_argument("--location-index", default="flex_mem",
                         help="PyOsmium location index, e.g. flex_mem")
    return command


def main(argv: Iterable[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    try:
        region = (Region.from_geojson(arguments.boundary_geojson)
                  if arguments.boundary_geojson else Region(arguments.bbox))
        build_graph(arguments.input, arguments.output, region,
                    component_mode=arguments.component,
                    progress_every=arguments.progress_every,
                    location_index=arguments.location_index)
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
