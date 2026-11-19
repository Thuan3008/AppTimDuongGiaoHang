"""Backend bản đồ: chạy RAPO, giữ metric dataset và lấy đường vẽ từ OSRM."""

from __future__ import annotations

import json
import logging
import math
import random
import re
import statistics
import sys
import time
import unicodedata
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from threading import Lock, Timer
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, abort, g, jsonify, request, send_file, session

from algorithms.poryos_baseline_adapter import run_baseline
from algorithms.rapo_2709 import RAPO2709Config, optimize
from algorithms.hcmc_td_erapo import HCMCTDERAPOConfig, optimize as run_hcmc_td_erapo
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.load_poryos import load_poryos
from core.data.models import Customer, ProceduralTrafficOverlay
from core.routing.fastest_path import prepare_adjacency
from web_map.auth import (
    authenticate,
    complete_trip_stop,
    create_trip,
    create_user,
    delete_driver,
    driver_exists,
    get_db,
    init_auth,
    list_drivers,
    list_job_stops,
    list_trips_for_user,
    list_trip_stops,
    login_required,
    reassign_trip,
    role_required,
    trip_by_id,
    trip_payload,
    update_driver,
    update_trip_status,
    user_payload,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.errorhandler(404)
def handle_404(e):
    return jsonify(error="Không tìm thấy tài nguyên yêu cầu (404)"), 404


@app.errorhandler(403)
def handle_403(e):
    return jsonify(error="Bạn không có quyền thực hiện thao tác này (403)"), 403


@app.errorhandler(405)
def handle_405(e):
    return jsonify(error="Phương thức HTTP không được hỗ trợ (405)"), 405


@app.errorhandler(500)
def handle_500(e):
    return jsonify(error="Lỗi máy chủ nội bộ (500)"), 500


@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
# GA dùng adapter có sẵn, cùng quần thể ban đầu và pipeline vẽ tuyến OSRM.
ALGORITHMS = ('RAPO-2709', 'RAPO-Pufferfish', 'GA')
ALGORITHM_LABELS = {'RAPO-2709': 'RAPO', 'HCMC-TD-ERAPO': 'HCMC-TD-ERAPO'}


def algorithm_label(name: str) -> str:
    """Tên hiển thị; khóa thuật toán nội bộ được giữ nguyên để tương thích dữ liệu."""
    return ALGORITHM_LABELS.get(name, name)


DATASETS = {
    f'hk{n}': ROOT / f'data/TDVRP/hong_kong/n={n}/poryos-hong_kong-n{n}-hyb/bpr-heavy/poryos-hong_kong-n{n}-hyb-bpr-heavy.vrp.json'
    for n in (25, 50, 100)
}
DATASET_ALGORITHMS = {key: ALGORITHMS for key in DATASETS}
for n in (10, 100, 500, 1000):
    DATASETS[f'hcmc{n}'] = ROOT / f'data/hcmc/instances/hcmc-n{n}-heavy.vrp.json'
    DATASET_ALGORITHMS[f'hcmc{n}'] = ('HCMC-TD-ERAPO',)
OUTPUT = ROOT / 'results' / 'web_map'
init_auth(app, OUTPUT / 'delivery-auth.sqlite3')
EXECUTOR = ThreadPoolExecutor(max_workers=1)
JOBS: dict[str, dict] = {}
JOB_LOCK = Lock()
OSRM_LOCK = Lock()
OSRM_LAST_REQUEST = 0.0
GEOCODE_LOCK = Lock()
GEOCODE_LAST_REQUEST = 0.0
LOG = logging.getLogger('rapo-map')

HCMC_OSM_GRAPH = ROOT / 'data/hcmc/sidecars/hcmc.road-osm.json'
HCMC_OBSERVED_GRAPH = ROOT / 'data/hcmc/sidecars/hcmc.road-observed.json'
HCMC_OBSERVED_TRAFFIC = ROOT / 'data/hcmc/sidecars/hcmc.traffic-observed.json'
HCMC_HOURLY_ALPHA = (
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.2, 2.4, 2.0, 1.3, 1.3, 1.3,
    1.6, 1.3, 1.3, 1.3, 1.4, 2.6,
    2.3, 1.8, 1.5, 1.2, 1.1, 1.1,
)
HCMC_ROAD_CLASS_WEIGHTS = {
    'motorway': 0.55, 'motorway_link': 0.65,
    'trunk': 0.65, 'trunk_link': 0.75,
    'primary': 0.80, 'primary_link': 0.90,
    'secondary': 0.95, 'secondary_link': 1.00,
    'tertiary': 1.05, 'tertiary_link': 1.05,
    'unclassified': 1.00, 'residential': 0.90,
    'living_street': 0.85, 'service': 0.80, 'road': 1.00,
}
SNAP_AUTO_MAX_M = 100.0
SNAP_CONFIRM_MAX_M = 300.0
NODE_INDEXES = {}


def sidecar_status(path):
    """Kiểm tra đủ cả hai sidecar trước khi cho chọn dataset."""
    if not path.is_file():
        return [str(path)]
    payload = json.loads(path.read_text(encoding='utf-8'))
    missing = []
    for key in ('graph', 'traffic'):
        relative = Path(payload['td'][key]['path'])
        resolved = next((ancestor / relative for ancestor in path.parents
                         if (ancestor / relative).is_file()), None)
        if resolved is None:
            missing.append(str(path.parent.parent / relative))
    return missing


def _observed_way_speeds():
    """Collect hourly observations keyed by OSM way id."""
    if not (HCMC_OBSERVED_GRAPH.is_file() and HCMC_OBSERVED_TRAFFIC.is_file()):
        return {}
    _, observed_edges = load_road_graph(HCMC_OBSERVED_GRAPH)
    observed_traffic = load_traffic(HCMC_OBSERVED_TRAFFIC, observed_edges)
    samples = {}
    for edge_id, edge in observed_edges.items():
        profile = observed_traffic.edges[edge_id].hourly_speed_kmh
        for way_id in edge.osm_way_ids:
            bucket = samples.setdefault(way_id, {f'{hour:02d}': [] for hour in range(24)})
            for hour, speed in profile.items():
                bucket[hour].append(float(speed))
    return {
        way_id: {hour: statistics.median(values) for hour, values in hours.items()}
        for way_id, hours in samples.items()
    }


@lru_cache(maxsize=1)
def _load_hcmc_network():
    """Load one shared HCMC graph so each dataset does not duplicate it in RAM."""
    if HCMC_OSM_GRAPH.is_file():
        graph = load_road_graph(HCMC_OSM_GRAPH)
        prepare_adjacency(graph)
        traffic = ProceduralTrafficOverlay(
            observed_way_speeds=_observed_way_speeds(),
            hourly_alpha=HCMC_HOURLY_ALPHA,
            road_class_weights=HCMC_ROAD_CLASS_WEIGHTS,
        )
        return graph, traffic, HCMC_OSM_GRAPH
    graph_path = HCMC_OBSERVED_GRAPH
    graph = load_road_graph(graph_path)
    prepare_adjacency(graph)
    traffic = load_traffic(HCMC_OBSERVED_TRAFFIC, graph[1])
    return graph, traffic, graph_path


def _snap_instance_to_graph(instance, graph, graph_path):
    """Replace legacy synthetic node ids with nearest nodes in the OSM graph."""
    nodes, _ = graph
    if all(node_id in nodes for node_id in instance.road_node_ids):
        return instance
    node_index = _road_node_index(nodes)
    road_node_ids = [node_index.nearest(lat, lon).id for lat, lon in instance.coordinates]
    customers = [
        Customer(index, road_node_ids[index], instance.demands[index],
                 instance.service_times[index], instance.coordinates[index])
        for index in range(1, instance.num_customers + 1)
    ]
    return replace(
        instance,
        road_node_ids=road_node_ids,
        customers=customers,
        graph_path=str(graph_path),
        traffic_path='procedural:hcmc-osm-heavy',
        metadata={**instance.metadata, 'road_graph': 'hcmc-osm',
                  'legacy_nodes_resnapped': True},
    )


@lru_cache(maxsize=6)
def load_data(key, exact=False):
    """Preview không dựng ATF; khi tối ưu mới dựng và xác minh ATF."""
    if key.startswith('hcmc'):
        instance = load_instance(DATASETS[key], validate_sidecars=True)
        graph, traffic, graph_path = _load_hcmc_network()
        instance = _snap_instance_to_graph(instance, graph, graph_path)
        return instance, graph, traffic
    return load_poryos(DATASETS[key], verify_checksums=True,
                       allow_proxy=False, materialize_atfs=exact)


def map_points(instance, graph):
    """Lấy lon/lat của node thật, không dùng tọa độ cục bộ của instance."""
    nodes, _ = graph
    custom_points = instance.metadata.get('custom_points', [])
    points = []
    for customer_id, node_id in enumerate(instance.road_node_ids):
        node = nodes[node_id]
        if not (math.isfinite(node.lat) and math.isfinite(node.lon)
                and -90 <= node.lat <= 90 and -180 <= node.lon <= 180):
            raise ValueError(f'Tọa độ địa lý không hợp lệ tại khách hàng {customer_id}')
        custom = custom_points[customer_id] if customer_id < len(custom_points) else None
        points.append(dict(
            id=customer_id,
            name=(custom or {}).get('name', 'Kho' if customer_id == instance.depot else f'Khách {customer_id}'),
            address=(custom or {}).get('address'),
            lat=(custom or {}).get('lat', node.lat), lon=(custom or {}).get('lon', node.lon),
            road_lat=node.lat, road_lon=node.lon,
            snap_distance_m=(custom or {}).get('snap_distance_m', 0.0),
            demand=instance.demands[customer_id], depot=customer_id == instance.depot,
        ))
    return points


def _finite_number(value, label, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} phải là số hợp lệ')
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f'{label} phải lớn hơn 0')
    if minimum is not None and number < minimum or maximum is not None and number > maximum:
        raise ValueError(f'{label} nằm ngoài phạm vi cho phép')
    return number


def normalize_custom_points(value):
    """Validate and normalize a manually entered depot and delivery list."""
    if not isinstance(value, dict):
        raise ValueError('Dữ liệu điểm giao tự nhập không hợp lệ')
    depot = value.get('depot')
    customers = value.get('customers')
    if not isinstance(depot, dict):
        raise ValueError('Cần nhập tọa độ kho xuất phát')
    if not isinstance(customers, list) or not customers:
        raise ValueError('Cần ít nhất một điểm giao')
    if len(customers) > 100:
        raise ValueError('Mỗi chuyến chỉ được nhập tối đa 100 điểm giao')
    vehicles = value.get('num_vehicles')
    if type(vehicles) is not int or not 1 <= vehicles <= 50:
        raise ValueError('Số xe phải là số nguyên từ 1 đến 50')
    capacity = _finite_number(value.get('vehicle_capacity'), 'Tải trọng xe', positive=True, maximum=1_000_000)

    def location(item, label):
        lat = _finite_number(item.get('lat'), f'Vĩ độ {label}', minimum=-90, maximum=90)
        lon = _finite_number(item.get('lon'), f'Kinh độ {label}', minimum=-180, maximum=180)
        return lat, lon

    depot_lat, depot_lon = location(depot, 'kho')
    depot_address = str(depot.get('address') or '').strip()
    if len(depot_address) > 300:
        raise ValueError('Địa chỉ kho dài quá 300 ký tự')
    normalized_customers = []
    for index, item in enumerate(customers, 1):
        if not isinstance(item, dict):
            raise ValueError(f'Điểm giao {index} không hợp lệ')
        lat, lon = location(item, f'điểm giao {index}')
        demand = _finite_number(item.get('demand'), f'Nhu cầu điểm giao {index}', positive=True, maximum=capacity)
        service_time = _finite_number(item.get('service_time', 0), f'Thời gian phục vụ điểm giao {index}', minimum=0, maximum=86400)
        name = str(item.get('name') or f'Điểm giao {index}').strip()
        if len(name) > 100:
            raise ValueError(f'Tên điểm giao {index} dài quá 100 ký tự')
        address = str(item.get('address') or '').strip()
        if len(address) > 300:
            raise ValueError(f'Địa chỉ điểm giao {index} dài quá 300 ký tự')
        window = item.get('time_window')
        if window is not None:
            if not isinstance(window, list) or len(window) != 2:
                raise ValueError(f'Khung giờ điểm giao {index} phải có giờ bắt đầu và kết thúc')
            start = _finite_number(window[0], f'Giờ bắt đầu điểm giao {index}', minimum=0, maximum=172799)
            end = _finite_number(window[1], f'Giờ kết thúc điểm giao {index}', minimum=1, maximum=172800)
            if start >= end:
                raise ValueError(f'Khung giờ điểm giao {index} phải kết thúc sau khi bắt đầu')
            window = [start, end]
        normalized_customers.append(dict(
            name=name, address=address, lat=lat, lon=lon, demand=demand,
            service_time=service_time, time_window=window,
            snap_confirmed=item.get('snap_confirmed') is True,
        ))
    if sum(item['demand'] for item in normalized_customers) > vehicles * capacity + 1e-9:
        raise ValueError('Tổng nhu cầu vượt quá tổng tải trọng của các xe')
    return dict(
        depot=dict(name=str(depot.get('name') or 'Kho').strip()[:100], address=depot_address,
                   lat=depot_lat, lon=depot_lon,
                   snap_confirmed=depot.get('snap_confirmed') is True),
        num_vehicles=vehicles, vehicle_capacity=capacity, customers=normalized_customers,
    )


def _distance_m(lat1, lon1, lat2, lon2):
    """Fast local-distance approximation used only to snap UI coordinates."""
    mean_lat = math.radians((lat1 + lat2) / 2)
    return math.hypot((lat1 - lat2) * 111_320, (lon1 - lon2) * 111_320 * math.cos(mean_lat))


def _snap_policy(distance_m):
    if distance_m <= SNAP_AUTO_MAX_M:
        return 'auto'
    if distance_m <= SNAP_CONFIRM_MAX_M:
        return 'confirm'
    return 'rejected'


class RoadNodeIndex:
    """Small dependency-free spatial grid for nearest-road-node queries."""

    cell_size = 0.01

    def __init__(self, nodes):
        self.nodes = nodes
        self.cells = {}
        for node in nodes.values():
            key = self._cell(node.lat, node.lon)
            self.cells.setdefault(key, []).append(node)

    @classmethod
    def _cell(cls, lat, lon):
        return math.floor(lat / cls.cell_size), math.floor(lon / cls.cell_size)

    def nearest(self, lat, lon):
        center_y, center_x = self._cell(lat, lon)
        best_node = None
        best_distance = math.inf
        for radius in range(101):
            for cell_y in range(center_y - radius, center_y + radius + 1):
                for cell_x in range(center_x - radius, center_x + radius + 1):
                    if radius and (abs(cell_y - center_y) != radius
                                   and abs(cell_x - center_x) != radius):
                        continue
                    for node in self.cells.get((cell_y, cell_x), ()):
                        distance = _distance_m(lat, lon, node.lat, node.lon)
                        if distance < best_distance:
                            best_node, best_distance = node, distance
            if best_node is not None:
                south = (center_y - radius) * self.cell_size
                north = (center_y + radius + 1) * self.cell_size
                west = (center_x - radius) * self.cell_size
                east = (center_x + radius + 1) * self.cell_size
                lon_scale = 111_320 * max(0.01, math.cos(math.radians(lat)))
                outside_lower_bound = min(
                    (lat - south) * 111_320,
                    (north - lat) * 111_320,
                    (lon - west) * lon_scale,
                    (east - lon) * lon_scale,
                )
                if best_distance <= outside_lower_bound:
                    return best_node
        if best_node is None:
            raise ValueError('Road graph không có node gần tọa độ đã chọn')
        return best_node


def _road_node_index(nodes):
    """Reuse the index for the shared city graph without copying its nodes."""
    key = id(nodes)
    cached = NODE_INDEXES.get(key)
    if cached is None or cached[0] is not nodes:
        cached = (nodes, RoadNodeIndex(nodes))
        NODE_INDEXES[key] = cached
    return cached[1]


@lru_cache(maxsize=512)
def geocode_address(address):
    """Resolve one explicitly submitted address, with two OSM providers."""
    global GEOCODE_LAST_REQUEST
    query = address.strip()
    if not query:
        raise ValueError('Địa chỉ không được để trống')
    if len(query) > 300:
        raise ValueError('Địa chỉ dài quá 300 ký tự')
    lowered = query.casefold()
    if 'hồ chí minh' not in lowered and 'ho chi minh' not in lowered and 'tp.hcm' not in lowered:
        query += ', Thành phố Hồ Chí Minh, Việt Nam'
    request_headers = {'User-Agent': 'TDVRP-RAPO-Research/1.0 (local academic delivery-routing app)'}
    provider_requests = [
        ('Photon', 'https://photon.komoot.io/api/?' + urlencode({
            'q': query, 'limit': 5, 'lat': 10.79, 'lon': 106.65,
        })),
        ('Nominatim', 'https://nominatim.openstreetmap.org/search?' + urlencode({
            'q': query, 'format': 'jsonv2', 'limit': 5, 'countrycodes': 'vn',
            'addressdetails': 1, 'accept-language': 'vi',
        })),
    ]
    failures = []
    combined = []
    for provider_order, (provider, url) in enumerate(provider_requests):
        try:
            with GEOCODE_LOCK:
                time.sleep(max(0, 1.05 - (time.monotonic() - GEOCODE_LAST_REQUEST)))
                GEOCODE_LAST_REQUEST = time.monotonic()
                with urlopen(Request(url, headers=request_headers), timeout=15) as response:
                    payload = json.load(response)
        except Exception as error:
            failures.append(f'{provider}: {error}')
            continue
        results = []
        if provider == 'Photon':
            for source_rank, item in enumerate(payload.get('features', []) if isinstance(payload, dict) else []):
                coordinates = item.get('geometry', {}).get('coordinates', [])
                if len(coordinates) < 2:
                    continue
                properties = item.get('properties', {})
                address_parts = [properties.get(key) for key in (
                    'name', 'housenumber', 'street', 'district', 'city', 'country')]
                display_name = ', '.join(dict.fromkeys(str(part) for part in address_parts if part))
                try:
                    lon, lat = map(float, coordinates[:2])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(lat) and math.isfinite(lon):
                    results.append(dict(
                        lat=lat, lon=lon, display_name=display_name or query,
                        provider=provider, provider_order=provider_order, source_rank=source_rank,
                        house_number=str(properties.get('housenumber') or '').strip(),
                    ))
        else:
            for source_rank, item in enumerate(payload if isinstance(payload, list) else []):
                try:
                    lat, lon = float(item['lat']), float(item['lon'])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(lat) and math.isfinite(lon):
                    address_fields = item.get('address', {}) if isinstance(item.get('address'), dict) else {}
                    results.append(dict(
                        lat=lat, lon=lon,
                        display_name=str(item.get('display_name') or query), provider=provider,
                        provider_order=provider_order, source_rank=source_rank,
                        house_number=str(address_fields.get('house_number') or '').strip(),
                    ))
        if results:
            combined.extend(results)
        else:
            failures.append(f'{provider}: không có kết quả')
    if combined:
        unique = []
        seen = set()
        for item in combined:
            key = (round(item['lat'], 6), round(item['lon'], 6),
                   _normalize_address_text(item['display_name']))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
    raise ValueError('Không tìm được địa chỉ qua Photon/Nominatim. ' + ' | '.join(failures))


ADDRESS_STOP_WORDS = {
    'duong', 'phuong', 'quan', 'huyen', 'xa', 'thi', 'tran', 'thanh', 'pho',
    'ho', 'chi', 'minh', 'viet', 'nam', 'tp', 'hcm',
}


def _normalize_address_text(value):
    text = unicodedata.normalize('NFD', str(value).casefold().replace('đ', 'd'))
    text = ''.join(character for character in text if unicodedata.category(character) != 'Mn')
    return re.sub(r'[^a-z0-9/]+', ' ', text).strip()


def _query_house_number(address):
    normalized = _normalize_address_text(address)
    match = re.match(r'^(\d+(?:\s*/\s*\d+)*)\b', normalized)
    return re.sub(r'\s+', '', match.group(1)) if match else ''


def _address_tokens(value):
    return {
        token for token in _normalize_address_text(value).split()
        if len(token) > 1 and not token.isdigit() and token not in ADDRESS_STOP_WORDS
    }


def _score_geocode_candidate(query, candidate):
    """Prefer address agreement; graph distance is only a final tie-breaker."""
    query_house = _query_house_number(query)
    candidate_house = re.sub(r'\s+', '', _normalize_address_text(candidate.get('house_number', '')))
    query_tokens = _address_tokens(query)
    candidate_tokens = _address_tokens(candidate['display_name'])
    overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
    score = overlap * 60.0
    if query_tokens and query_tokens <= candidate_tokens:
        score += 20.0
    if query_house:
        if candidate_house == query_house:
            score += 140.0
            house_match = 'exact'
        elif candidate_house:
            score -= 60.0
            house_match = 'wrong'
        else:
            score -= 45.0
            house_match = 'missing'
    else:
        house_match = 'not_requested'
    score -= candidate.get('provider_order', 0) * 0.25
    score -= candidate.get('source_rank', 0) * 0.05
    return round(score, 3), house_match


def build_custom_instance(base_instance, graph, custom_points, trip_name):
    """Create an in-memory instance by snapping entered coordinates to the selected road graph."""
    custom = normalize_custom_points(custom_points)
    nodes, _ = graph
    node_index = _road_node_index(nodes)
    entered = [custom['depot'], *custom['customers']]
    road_node_ids, details = [], []
    for index, point in enumerate(entered):
        node = node_index.nearest(point['lat'], point['lon'])
        snap_distance = _distance_m(point['lat'], point['lon'], node.lat, node.lon)
        snap_status = _snap_policy(snap_distance)
        label = 'Kho' if index == 0 else f'Điểm giao {index}'
        if snap_status == 'rejected':
            raise ValueError(
                f'{label} cách đường gần nhất {snap_distance:.1f} m, vượt giới hạn 300 m; '
                'hãy chọn lại vị trí gần đường xe chạy hơn'
            )
        if snap_status == 'confirm' and not point['snap_confirmed']:
            raise ValueError(
                f'{label} cách đường gần nhất {snap_distance:.1f} m; '
                'cần xác nhận vị trí trước khi tạo chuyến'
            )
        road_node_ids.append(node.id)
        details.append({**point, 'snap_distance_m': round(snap_distance, 1),
                        'snap_status': snap_status})
    demands = [0.0, *[item['demand'] for item in custom['customers']]]
    service_times = [0.0, *[item['service_time'] for item in custom['customers']]]
    coordinates = [(item['lat'], item['lon']) for item in entered]
    horizon = tuple(map(float, base_instance.horizon))
    time_windows = [horizon, *[
        tuple(item['time_window']) if item['time_window'] is not None else horizon
        for item in custom['customers']
    ]]
    customers = [
        Customer(index, road_node_ids[index], demands[index], service_times[index], coordinates[index])
        for index in range(1, len(entered))
    ]
    return replace(
        base_instance,
        instance_name=trip_name,
        num_customers=len(custom['customers']),
        num_vehicles=custom['num_vehicles'],
        vehicle_capacity=custom['vehicle_capacity'],
        coordinates=coordinates,
        demands=demands,
        service_times=service_times,
        road_node_ids=road_node_ids,
        customers=customers,
        time_windows=time_windows,
        arrival_time_functions=None,
        metadata={**base_instance.metadata, 'custom_points': details, 'source': 'admin-ui'},
    )


@lru_cache(maxsize=512)
def osrm_segment(points):
    """Lấy GeoJSON theo thứ tự ghé thăm; cache và giới hạn một request/giây."""
    global OSRM_LAST_REQUEST
    coordinates = ';'.join(f'{lon:.7f},{lat:.7f}' for lon, lat in points)
    url = ('https://router.project-osrm.org/route/v1/driving/' + coordinates
           + '?overview=full&geometries=geojson&steps=false&alternatives=false')
    with OSRM_LOCK:
        time.sleep(max(0, 1.1 - (time.monotonic() - OSRM_LAST_REQUEST)))
        OSRM_LAST_REQUEST = time.monotonic()
        with urlopen(Request(url, headers={'User-Agent': 'TDVRP-RAPO-Research-Map/1.0'}), timeout=15) as response:
            payload = json.load(response)
    if payload.get('code') != 'Ok' or not payload.get('routes'):
        raise ValueError('OSRM: ' + str(payload.get('code', 'phản hồi không hợp lệ')))
    geometry = payload['routes'][0]['geometry']
    coords = geometry.get('coordinates', [])
    if geometry.get('type') != 'LineString' or len(coords) < 2:
        raise ValueError('OSRM thiếu LineString')
    for point in coords:
        if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            raise ValueError('OSRM trả tọa độ không hợp lệ')
        if not (-180 <= point[0] <= 180 and -90 <= point[1] <= 90):
            raise ValueError('OSRM trả tọa độ ngoài giới hạn')
    return coords


def osrm_geometry(points):
    """Chia tuyến dài thành các đoạn chồng một điểm, giữ nguyên thứ tự khách."""
    coords = []
    for start in range(0, len(points) - 1, 49):
        segment = osrm_segment(tuple(points[start:start + 50]))
        coords.extend(segment if not coords else segment[1:])
    return {'type': 'LineString', 'coordinates': coords}


def route_payload(route, instance, graph, osrm_available=True):
    """Giữ nguyên thời gian dataset; hình học OSRM chỉ dùng để hiển thị."""
    nodes, _ = graph
    visits = [instance.depot] + [c for c in route.customer_ids if c != instance.depot] + [instance.depot]
    waypoints = [(nodes[instance.road_node_ids[c]].lon, nodes[instance.road_node_ids[c]].lat) for c in visits]
    warning = None
    source = 'osrm'
    try:
        if not osrm_available:
            raise ValueError('OSRM tạm không khả dụng trong lần chạy này')
        geometry = osrm_geometry(waypoints)
    except Exception as error:
        # Dự phòng bằng đường của dataset và ghi nhãn rõ, không giả đường OSRM.
        warning = str(error)
        trace = [(nodes[n].lon, nodes[n].lat) for n in route.node_path if n in nodes]
        source = 'dataset-road-graph' if len(trace) >= 2 else 'straight-line'
        geometry = {'type': 'LineString', 'coordinates': trace if len(trace) >= 2 else waypoints}
    return dict(vehicle=route.vehicle_index + 1, visits=visits, load=route.load,
                departure_time=route.start_time, end_time=route.end_time,
                travel_time=route.travel_time, service_time=route.service_time,
                waiting_time=route.waiting_time, duration=route.duration,
                objective=route.travel_time + route.service_time,
                node_path=list(route.node_path), edge_path=list(route.edge_path),
                dataset_distance_m=route.distance, geometry=geometry,
                geometry_source=source, warning=warning,
                legs=[asdict(leg) for leg in route.legs],
                stops=[asdict(stop) for stop in route.stops])


def update_job(job_id, **values):
    with JOB_LOCK:
        JOBS[job_id].update(values)
    if 'stage' in values:
        LOG.info('%s | %s', job_id[:8], values['stage'])


def run_job(job_id, config):
    """Chạy tuần tự trên worker để HTTP vẫn phản hồi tiến trình và bản đồ."""
    started = time.perf_counter()
    results, errors = {}, {}
    runs = {algorithm: [] for algorithm in config['algorithms']}
    total_runs = config.get('num_seeds', 1) * len(config['algorithms'])
    completed_runs = 0
    try:
        update_job(job_id, status='running', stage='Đang nạp dữ liệu mạng lưới')
        instance, graph, traffic = load_data(config['dataset'], True)
        if config.get('custom_points') is not None:
            instance = build_custom_instance(instance, graph, config['custom_points'], config['trip_name'])
        customers = list(range(1, instance.num_customers + 1))
        osrm_available = True
        # Mỗi seed có P0 riêng; các thuật toán ở cùng seed dùng đúng P0 đó.
        schedule = [(seed, algorithm)
                    for seed in range(config['seed'], config['seed'] + config.get('num_seeds', 1))
                    for algorithm in config['algorithms']]
        for seed, algorithm in schedule:
            rng = random.Random(seed)
            population = [tuple(rng.sample(customers, len(customers))) for _ in range(config['population_size'])]
            label = f'[{completed_runs + 1}/{total_runs}] {algorithm_label(algorithm)} · seed {seed}'
            try:
                update_job(job_id, stage=f'{label}: đang tối ưu, ngân sách {config["budget"]} NFE',
                           completed_runs=completed_runs, total_runs=total_runs)
                algorithm_start = time.perf_counter()
                if algorithm == 'RAPO-2709':
                    result = optimize(instance, graph, traffic, config=RAPO2709Config(
                        population_size=config['population_size'], max_function_evaluations=config['budget'],
                        seed=seed, earliest_departure=config['earliest_departure'],
                        latest_departure=config.get('latest_departure')),
                        initial_population=population)
                elif algorithm == 'HCMC-TD-ERAPO':
                    def on_hcmc_progress(evals, budget, current_fitness):
                        update_job(job_id, stage=f'{label}: NFE {evals}/{budget} (fitness: {current_fitness:.1f})')

                    result = run_hcmc_td_erapo(
                        instance,
                        graph,
                        traffic,
                        config=HCMCTDERAPOConfig(
                            population_size=config['population_size'],
                            max_function_evaluations=config['budget'],
                            earliest_departure=config['earliest_departure'],
                            latest_departure=config.get('latest_departure'),
                            seed=seed,
                        ),
                        seed=seed,
                        initial_population=population,
                        progress_callback=on_hcmc_progress,
                    )
                else:
                    result = run_baseline(algorithm, instance, graph, traffic,
                        instance_path=DATASETS[config['dataset']], population_size=config['population_size'],
                        nfe_budget=config['budget'], seed=seed, initial_population=population,
                        start_time=config['earliest_departure'],
                        latest_time=config.get('latest_departure'))
                evaluation_seconds = time.perf_counter() - algorithm_start
                evaluation = result.best_evaluation
                if not evaluation.feasible or not math.isfinite(evaluation.fitness):
                    raise ValueError('Nghiệm không khả thi: ' + '; '.join(evaluation.errors))
                routes = []
                for index, route in enumerate(evaluation.routes):
                    update_job(job_id, stage=f'{label}: lấy đường vẽ {index + 1}/{len(evaluation.routes)}')
                    item = route_payload(route, instance, graph, osrm_available)
                    if item['warning']:
                        osrm_available = False
                    routes.append(item)
                item = dict(seed=seed, fitness=evaluation.fitness, feasible=evaluation.feasible,
                    travel_time=evaluation.total_travel_time, service_time=evaluation.total_service_time,
                    total_route_duration=evaluation.total_route_duration,
                    nfe=result.evaluations, runtime_seconds=result.runtime_seconds,
                    optimization_and_trace_seconds=evaluation_seconds,
                    departure_time_mode=evaluation.departure_time_mode,
                    convergence=list(result.convergence), routes=routes)
                runs[algorithm].append(item)
                # Giữ nghiệm tốt nhất để tương thích với kết quả một seed trước đây.
                if algorithm not in results or item['fitness'] < results[algorithm]['fitness']:
                    results[algorithm] = item
            except Exception as error:
                LOG.exception('%s bị lỗi', label)
                errors[f'{algorithm} · seed {seed}'] = str(error)
            completed_runs += 1
            update_job(job_id, completed_runs=completed_runs)
        # Chỉ tổng hợp các lượt thành công; ghi rõ số lỗi, SD mẫu cần ít nhất 2 seed.
        summary = {}
        for algorithm, items in runs.items():
            fitnesses = [item['fitness'] for item in items]
            summary[algorithm] = dict(
                completed=len(items), failed=config.get('num_seeds', 1) - len(items),
                best=min(fitnesses) if items else None,
                mean=statistics.mean(fitnesses) if items else None,
                median=statistics.median(fitnesses) if items else None,
                sd=statistics.stdev(fitnesses) if len(items) > 1 else None,
                runtime_mean_seconds=statistics.mean(item['runtime_seconds'] for item in items) if items else None,
                best_seed=results[algorithm]['seed'] if items else None)
        payload = dict(config=config, instance=instance.instance_name, points=map_points(instance, graph),
                       capacity=instance.vehicle_capacity,
                       results=results, runs=runs, summary=summary, errors=errors,
                       elapsed_seconds=time.perf_counter() - started,
                       metric_source=(
                           'HCMC: vận tốc quan sát theo giờ + fallback mô phỏng cho cung chưa ghép'
                           if config['dataset'].startswith('hcmc')
                           else 'Poryos dataset; OSRM chỉ cung cấp hình học'),
                       initialization_policy='common_random_population')
        # Lưu kết quả đã hoàn tất để tải lại trang hoặc restart server vẫn xem được.
        OUTPUT.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT / f'{job_id}.json'
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(destination)
        update_job(job_id, status='completed' if results else 'failed',
                   stage=f'Hoàn tất {completed_runs}/{total_runs} lượt' if results else 'Các thuật toán đều lỗi', payload=payload)
    except Exception as error:
        LOG.exception('Lỗi job %s', job_id)
        update_job(job_id, status='failed', stage=str(error))


def load_saved_job(job_id):
    """Return a completed job payload from memory or its persisted JSON file."""
    with JOB_LOCK:
        active = JOBS.get(job_id)
        if active is not None and active.get('payload') is not None:
            return active['payload']
    path = OUTPUT / f'{job_id}.json'
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def payload_for_trip(row):
    """Filter an optimization result so a Driver sees only the assigned route."""
    payload = load_saved_job(row['job_id'])
    if payload is None:
        return None
    result = payload.get('results', {}).get(row['algorithm'])
    route_index = int(row['route_index'])
    if result is None or not 0 <= route_index < len(result.get('routes', [])):
        return None
    assigned_route = result['routes'][route_index]
    visible_customer_ids = {int(customer_id) for customer_id in assigned_route.get('visits', [])}
    filtered_result = {**result, 'routes': [assigned_route]}
    return {
        'config': payload.get('config', {}),
        'instance': payload.get('instance'),
        'points': [point for point in payload.get('points', []) if int(point.get('id', -1)) in visible_customer_ids],
        'results': {row['algorithm']: filtered_result},
        'errors': {},
        'metric_source': payload.get('metric_source'),
    }


@app.get('/')
def index():
    template_path = Path(__file__).resolve().parent / 'templates' / 'index.html'
    if template_path.exists():
        return send_file(template_path)
    return send_file(Path(__file__).with_name('index.html'))


@app.post('/api/auth/login')
def login():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get('username'), str) or not isinstance(body.get('password'), str):
        return jsonify(error='Cần username và password'), 400
    user = authenticate(body['username'], body['password'])
    if user is None:
        return jsonify(error='Sai tài khoản hoặc mật khẩu'), 401
    session.clear()
    session['user_id'] = int(user['id'])
    session.permanent = True
    return jsonify(user=user_payload(user))


@app.post('/api/auth/logout')
@login_required
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get('/api/auth/me')
@login_required
def me():
    return jsonify(user=user_payload())


@app.get('/api/drivers')
@role_required('admin')
def drivers():
    include_inactive = request.args.get('include_inactive', '').lower() in ('1', 'true', 'yes') or request.args.get('all', '').lower() in ('1', 'true', 'yes')
    return jsonify(drivers=list_drivers(include_inactive=include_inactive))


@app.post('/api/drivers')
@role_required('admin')
def add_driver():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get('username'), str) or not isinstance(body.get('password'), str):
        return jsonify(error='Cần username và password cho tài xế'), 400
    full_name = body.get('full_name', '')
    if full_name is None:
        full_name = ''
    if not isinstance(full_name, str):
        return jsonify(error='Tên tài xế không hợp lệ'), 400
    try:
        return jsonify(driver=create_user(body['username'], body['password'], 'driver', full_name=full_name)), 201
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.patch('/api/drivers/<int:driver_id>')
@role_required('admin')
def edit_driver(driver_id):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error='Cần JSON dữ liệu cập nhật tài xế'), 400
    full_name = body.get('full_name') if 'full_name' in body else None
    active = body.get('active') if 'active' in body else None
    password = body.get('password') if 'password' in body else None
    if full_name is not None and not isinstance(full_name, str):
        return jsonify(error='Tên tài xế không hợp lệ'), 400
    if active is not None and not isinstance(active, bool):
        return jsonify(error='Trạng thái hoạt động phải là kiểu boolean'), 400
    if password is not None:
        if not isinstance(password, str):
            return jsonify(error='Mật khẩu mới không hợp lệ'), 400
        if password == '':
            password = None
    try:
        driver = update_driver(driver_id, full_name=full_name, active=active, password=password)
        return jsonify(driver=driver)
    except LookupError as error:
        return jsonify(error=str(error)), 404
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.delete('/api/drivers/<int:driver_id>')
@role_required('admin')
def remove_driver(driver_id):
    try:
        delete_driver(driver_id)
        return jsonify(ok=True)
    except LookupError as error:
        return jsonify(error=str(error)), 404
    except ValueError as error:
        return jsonify(error=str(error)), 409


def extract_plan_metadata(job_id: str, preferred_algorithm: str = None) -> dict:
    saved = load_saved_job(job_id) or {}
    saved_config = saved.get('config', {})
    saved_results = saved.get('results', {})
    alg_name = preferred_algorithm
    if not alg_name or alg_name not in saved_results:
        alg_name = next(iter(saved_results.keys()), 'RAPO-2709') if saved_results else 'RAPO-2709'
    alg_data = saved_results.get(alg_name, {})
    routes = alg_data.get('routes', [])
    routes_summary = []
    for idx, r in enumerate(routes):
        visits = r.get('visits', [])
        cust_count = max(0, len(visits) - 2) if len(visits) >= 2 else 0
        routes_summary.append({
            'route_index': idx,
            'vehicle': r.get('vehicle', idx + 1),
            'stops_count': cust_count,
            'load': r.get('load', 0),
            'duration': r.get('duration', 0),
        })
    return {
        'dataset': saved_config.get('dataset', 'dataset không xác định'),
        'instance': saved.get('instance'),
        'trip_name': saved_config.get('trip_name'),
        'departure_time': saved_config.get('earliest_departure'),
        'latest_departure': saved_config.get('latest_departure'),
        'algorithm': alg_name,
        'total_routes': len(routes),
        'routes_summary': routes_summary,
    }


@app.get('/api/trips')
@login_required
def trips():
    items = list_trips_for_user(g.user)
    with JOB_LOCK:
        active_statuses = {job_id: job.get('status') for job_id, job in JOBS.items()}
    plan_metadata = {}
    for item in items:
        if item['job_id'] not in plan_metadata:
            plan_metadata[item['job_id']] = extract_plan_metadata(item['job_id'], item.get('algorithm'))
        item['plan'] = plan_metadata[item['job_id']]
        job_status = active_statuses.get(item['job_id'])
        if job_status is None:
            result_path = OUTPUT / f"{item['job_id']}.json"
            if result_path.is_file():
                try:
                    job_status = 'completed' if json.loads(result_path.read_text(encoding='utf-8')).get('results') else 'failed'
                except (OSError, ValueError):
                    job_status = 'failed'
            else:
                job_status = 'missing'
        item['job_status'] = job_status
        stops = list_trip_stops(item['id'])
        item['stops_completed'] = sum(stop['status'] == 'completed' for stop in stops)
        item['stops_total'] = len(stops)

    admin_jobs = []
    if g.user['role'] == 'admin':
        known_job_ids = list(plan_metadata.keys())
        if OUTPUT.is_dir():
            for json_file in sorted(OUTPUT.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
                job_id = json_file.stem
                if len(job_id) == 32 and job_id not in plan_metadata:
                    meta = extract_plan_metadata(job_id)
                    if meta.get('total_routes', 0) > 0:
                        plan_metadata[job_id] = meta
                        known_job_ids.append(job_id)
        for job_id in known_job_ids:
            job_trips = [it for it in items if it['job_id'] == job_id]
            admin_jobs.append({
                'job_id': job_id,
                'plan': plan_metadata[job_id],
                'trips': job_trips,
            })

    return jsonify(trips=items, jobs=admin_jobs if g.user['role'] == 'admin' else None)


@app.post('/api/trips')
@role_required('admin')
def assign_trip():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error='Cần thông tin route và Driver'), 400
    job_id = body.get('job_id')
    algorithm = body.get('algorithm')
    route_index = body.get('route_index')
    driver_id = body.get('driver_id')
    if (not isinstance(job_id, str) or len(job_id) != 32
            or any(character not in '0123456789abcdef' for character in job_id)):
        return jsonify(error='job_id không hợp lệ'), 400
    if not isinstance(algorithm, str) or type(route_index) is not int:
        return jsonify(error='Thuật toán hoặc route_index không hợp lệ'), 400
    if type(driver_id) is not int or not driver_exists(driver_id):
        return jsonify(error='Hãy chọn một tài xế đang hoạt động'), 400
    payload = load_saved_job(job_id)
    result = payload.get('results', {}).get(algorithm) if payload else None
    if result is None or not 0 <= route_index < len(result.get('routes', [])):
        return jsonify(error='Không tìm thấy route trong kết quả đã hoàn tất'), 404
    route = result['routes'][route_index]
    customer_ids = [int(customer_id) for customer_id in route.get('visits', [])[1:-1]]
    if not customer_ids:
        return jsonify(error='Không thể giao một route rỗng'), 400

    db = get_db()
    existing = db.execute(
        "SELECT id, status FROM trips WHERE job_id = ? AND route_index = ?",
        (job_id, route_index),
    ).fetchone()
    if existing:
        try:
            updated = reassign_trip(existing['id'], driver_id, g.user)
            return jsonify(trip=updated, stops=list_trip_stops(existing['id'])), 200
        except (ValueError, LookupError, PermissionError) as error:
            return jsonify(error=str(error)), 400

    trip_id = uuid.uuid4().hex
    try:
        create_trip(
            trip_id, job_id, algorithm, route_index, driver_id,
            int(g.user['id']), customer_ids,
        )
    except ValueError as error:
        return jsonify(error=str(error)), 409
    row = trip_by_id(trip_id)
    return jsonify(trip=trip_payload(row), stops=list_trip_stops(trip_id)), 201


@app.get('/api/trips/<trip_id>')
@login_required
def get_trip(trip_id):
    row = trip_by_id(trip_id)
    if row is None:
        return jsonify(error=f'Không tìm thấy chuyến giao hàng {trip_id} (404)'), 404
    if g.user['role'] == 'driver' and int(row['driver_id']) != int(g.user['id']):
        return jsonify(error='Chuyến này không được gán cho tài xế hiện tại'), 403
    payload = payload_for_trip(row)
    if payload is None:
        return jsonify(error='Không tìm thấy route đã gán trong kết quả tối ưu'), 409
    return jsonify(trip=trip_payload(row), stops=list_trip_stops(trip_id), payload=payload)


@app.patch('/api/trips/<trip_id>/status')
@login_required
def change_trip_status(trip_id):
    row = trip_by_id(trip_id)
    if row is None:
        return jsonify(error=f'Không tìm thấy chuyến giao hàng {trip_id} (404)'), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get('status'), str):
        return jsonify(error='Cần trạng thái chuyến'), 400
    if g.user['role'] == 'driver' and body['status'] == 'in_progress':
        with JOB_LOCK:
            active_job = JOBS.get(row['job_id'])
            completed_in_memory = active_job is not None and active_job.get('status') == 'completed'
        result_path = OUTPUT / f"{row['job_id']}.json"
        completed_on_disk = False
        if result_path.is_file():
            try:
                completed_on_disk = bool(json.loads(result_path.read_text(encoding='utf-8')).get('results'))
            except (OSError, ValueError):
                completed_on_disk = False
        if not completed_in_memory and not completed_on_disk:
            return jsonify(error='Tuyến chưa tối ưu xong; chưa thể bắt đầu chuyến'), 409
    try:
        update_trip_status(row, body['status'], g.user)
    except PermissionError as error:
        return jsonify(error=str(error)), 403
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(trip=trip_payload(trip_by_id(trip_id)))


@app.route('/api/trips/<trip_id>/driver', methods=['PATCH', 'POST'])
@role_required('admin')
def change_trip_driver(trip_id):
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or type(body.get('driver_id')) is not int:
        return jsonify(error='Cần driver_id mới cho chuyến'), 400
    try:
        updated = reassign_trip(trip_id, body['driver_id'], g.user)
        return jsonify(trip=updated)
    except LookupError as error:
        return jsonify(error=str(error)), 404
    except (ValueError, PermissionError) as error:
        return jsonify(error=str(error)), 400


@app.patch('/api/trips/<trip_id>/stops/<int:stop_index>')
@role_required('driver')
def complete_stop(trip_id, stop_index):
    row = trip_by_id(trip_id)
    if row is None:
        return jsonify(error=f'Không tìm thấy chuyến giao hàng {trip_id} (404)'), 404
    try:
        complete_trip_stop(row, stop_index, g.user)
    except PermissionError as error:
        return jsonify(error=str(error)), 403
    except ValueError as error:
        return jsonify(error=str(error)), 400
    updated = trip_by_id(trip_id)
    return jsonify(trip=trip_payload(updated), stops=list_trip_stops(trip_id))


@app.get('/api/datasets')
@login_required
def datasets():
    def dataset_name(key):
        return (f'Hong Kong n={key[2:]} · HYB · BPR-heavy'
                if key.startswith('hk') else f'Hồ Chí Minh n={key[4:]} · traffic mô phỏng')
    return jsonify(algorithms=ALGORITHMS, datasets=[dict(id=key, name=dataset_name(key),
                   missing=sidecar_status(path), algorithms=DATASET_ALGORITHMS[key])
                   for key, path in DATASETS.items()])


@app.get('/api/datasets/<key>')
@login_required
def preview(key):
    if key not in DATASETS:
        return jsonify(error=f'Dataset {key} không tồn tại (404)'), 404
    try:
        instance, graph, _ = load_data(key)
        return jsonify(points=map_points(instance, graph), instance=instance.instance_name,
                       capacity=instance.vehicle_capacity, customers=instance.num_customers)
    except Exception as error:
        return jsonify(error=str(error)), 422


@app.post('/api/geocode')
@role_required('admin')
def geocode():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get('dataset') not in DATASETS:
        return jsonify(error='Dataset không được hỗ trợ'), 400
    if not body['dataset'].startswith('hcmc'):
        return jsonify(error='Tìm địa chỉ hiện chỉ hỗ trợ road graph Hồ Chí Minh'), 400
    address = body.get('address')
    if not isinstance(address, str):
        return jsonify(error='Cần nhập địa chỉ'), 400
    try:
        candidates = geocode_address(address)
        _, graph, _ = load_data(body['dataset'])
        node_index = _road_node_index(graph[0])
        ranked = []
        for candidate in candidates:
            node = node_index.nearest(candidate['lat'], candidate['lon'])
            distance = _distance_m(candidate['lat'], candidate['lon'], node.lat, node.lon)
            match_score, house_match = _score_geocode_candidate(address, candidate)
            ranked.append({
                **candidate,
                'snap_distance_m': round(distance, 1),
                'snap_status': _snap_policy(distance),
                'match_score': match_score,
                'house_match': house_match,
            })
        ranked.sort(key=lambda item: (-item['match_score'], item['snap_distance_m'],
                                      item['provider_order'], item['source_rank']))
        query_house = _query_house_number(address)
        house_warning = bool(query_house and ranked[0]['house_match'] != 'exact')
        snap_status = ranked[0]['snap_status']
        requires_confirmation = house_warning or snap_status != 'auto'
        public_fields = ('lat', 'lon', 'display_name', 'provider', 'snap_distance_m',
                         'snap_status', 'match_score', 'house_match')
        public_candidates = [
            {field: candidate[field] for field in public_fields}
            for candidate in ranked[:10]
        ]
        warnings = []
        if house_warning:
            warnings.append(f'Không tìm thấy kết quả khớp chính xác số nhà {query_house}.')
        if snap_status == 'confirm':
            warnings.append(
                f'Điểm tìm được cách đường gần nhất {ranked[0]["snap_distance_m"]:.1f} m; '
                'cần chọn và xác nhận.'
            )
        elif snap_status == 'rejected':
            warnings.append(
                f'Điểm tìm được cách đường gần nhất {ranked[0]["snap_distance_m"]:.1f} m, '
                'vượt giới hạn 300 m nên không thể dùng.'
            )
        return jsonify(
            result=public_candidates[0],
            candidates=public_candidates,
            requires_confirmation=requires_confirmation,
            warning=' '.join(warnings) or None,
        )
    except ValueError as error:
        return jsonify(error=str(error)), 422


@app.post('/api/snap-point')
@role_required('admin')
def snap_point():
    """Check a manually entered/map-picked coordinate against the local road graph."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get('dataset') not in DATASETS:
        return jsonify(error='Dataset không được hỗ trợ'), 400
    if not body['dataset'].startswith('hcmc'):
        return jsonify(error='Kiểm tra tọa độ hiện chỉ hỗ trợ road graph Hồ Chí Minh'), 400
    try:
        lat = _finite_number(body.get('lat'), 'Vĩ độ', minimum=-90, maximum=90)
        lon = _finite_number(body.get('lon'), 'Kinh độ', minimum=-180, maximum=180)
        _, graph, _ = load_data(body['dataset'])
        node = _road_node_index(graph[0]).nearest(lat, lon)
        distance = _distance_m(lat, lon, node.lat, node.lon)
        return jsonify(
            lat=lat, lon=lon,
            road_node_id=node.id, road_lat=node.lat, road_lon=node.lon,
            snap_distance_m=round(distance, 1), snap_status=_snap_policy(distance),
        )
    except ValueError as error:
        return jsonify(error=str(error)), 422


@app.post('/api/custom-points/preview')
@role_required('admin')
def preview_custom_points():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get('dataset') not in DATASETS:
        return jsonify(error='Dataset không được hỗ trợ'), 400
    if not body['dataset'].startswith('hcmc'):
        return jsonify(error='Điểm tự nhập hiện chỉ hỗ trợ road graph Hồ Chí Minh'), 400
    trip_name = str(body.get('trip_name') or 'Chuyến tự nhập').strip()[:100]
    try:
        base_instance, graph, _ = load_data(body['dataset'])
        instance = build_custom_instance(base_instance, graph, body.get('custom_points'), trip_name)
        return jsonify(points=map_points(instance, graph), instance=instance.instance_name,
                       capacity=instance.vehicle_capacity, customers=instance.num_customers,
                       vehicles=instance.num_vehicles)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        return jsonify(error=str(error)), 422


@app.post('/api/runs')
@role_required('admin')
def start_run():
    config = request.get_json(silent=True)
    if not isinstance(config, dict):
        return jsonify(error='Cần JSON cấu hình'), 400
    if not isinstance(config.get('dataset'), str) or config['dataset'] not in DATASETS:
        return jsonify(error='Dataset không được hỗ trợ'), 400
    algorithms = config.get('algorithms')
    supported = DATASET_ALGORITHMS[config['dataset']]
    if not isinstance(algorithms, list) or not algorithms or any(a not in supported for a in algorithms):
        return jsonify(error='Chọn thuật toán được hỗ trợ: ' + ', '.join(supported)), 400
    # Client cũ không gửi số seed vẫn chạy một lần như trước.
    config.setdefault('num_seeds', 1)
    config.setdefault('earliest_departure', 8 * 3600)
    config.setdefault('latest_departure', min(86399, config['earliest_departure'] + 3600))
    for key, low, high in [('population_size', 2, 50), ('budget', 2, 10000), ('seed', 0, 999999), ('num_seeds', 1, 100), ('earliest_departure', 0, 86399)]:
        value = config.get(key)
        if type(value) is not int or not low <= value <= high:
            return jsonify(error=f'{key} phải là số nguyên từ {low} đến {high}'), 400
    latest_departure = config.get('latest_departure')
    if latest_departure is not None:
        if type(latest_departure) is not int or not 0 <= latest_departure <= 86399:
            return jsonify(error='latest_departure phải là số nguyên từ 0 đến 86399'), 400
        if latest_departure < config['earliest_departure']:
            return jsonify(error='Giờ xuất phát muộn nhất phải lớn hơn hoặc bằng giờ xuất phát sớm nhất'), 400
    if config['budget'] < config['population_size']:
        return jsonify(error='NFE phải lớn hơn hoặc bằng population size'), 400
    if config['seed'] + config['num_seeds'] - 1 > 999999:
        return jsonify(error='Seed cuối cùng không được vượt quá 999999'), 400
    trip_name = str(config.get('trip_name') or f'Chuyến {config["dataset"]}').strip()
    if not trip_name or len(trip_name) > 100:
        return jsonify(error='Tên chuyến phải có từ 1 đến 100 ký tự'), 400
    custom_points = config.get('custom_points')
    if custom_points is not None:
        if not config['dataset'].startswith('hcmc'):
            return jsonify(error='Điểm tự nhập hiện chỉ hỗ trợ road graph Hồ Chí Minh'), 400
        try:
            custom_points = normalize_custom_points(custom_points)
            base_instance, graph, _ = load_data(config['dataset'])
            build_custom_instance(base_instance, graph, custom_points, trip_name)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception as error:
            return jsonify(error=str(error)), 422
    missing = sidecar_status(DATASETS[config['dataset']])
    if missing:
        return jsonify(error='Thiếu sidecar: ' + ', '.join(missing)), 422
    config = {key: config[key] for key in ('dataset', 'population_size', 'budget', 'seed', 'num_seeds', 'earliest_departure')} | {
        'algorithms': list(dict.fromkeys(algorithms)), 'trip_name': trip_name}
    if latest_departure is not None:
        config['latest_departure'] = latest_departure
    if custom_points is not None:
        config['custom_points'] = custom_points
    with JOB_LOCK:
        # Chỉ nhận một job mỗi lúc, tránh hai phiên làm sai lệch runtime.
        if any(j['status'] in ('queued', 'running') for j in JOBS.values()):
            return jsonify(error='Có lượt chạy đang xử lý; hãy chờ hoàn tất'), 409
        job_id = uuid.uuid4().hex
        JOBS[job_id] = dict(id=job_id, config=config, status='queued', stage='Đang chờ worker')
    EXECUTOR.submit(run_job, job_id, config)
    return jsonify(id=job_id), 202


@app.get('/api/runs/<job_id>')
@role_required('admin')
def get_run(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        return jsonify(error='Mã lượt chạy không hợp lệ (404)'), 404
    stops = list_job_stops(job_id)
    with JOB_LOCK:
        if job_id in JOBS:
            job_data = dict(JOBS[job_id])
            job_data['stops'] = stops
            return jsonify(job_data)
    path = OUTPUT / f'{job_id}.json'
    if not path.is_file():
        return jsonify(error=f'Không tìm thấy file dữ liệu cho lượt chạy {job_id} (404)'), 404
    payload = json.loads(path.read_text(encoding='utf-8'))
    return jsonify(id=job_id, config=payload['config'], status='completed' if payload['results'] else 'failed',
                   stage='Đã nạp kết quả lưu', payload=payload, stops=stops)


@app.get('/api/runs/<job_id>/stops')
@role_required('admin')
def get_run_stops(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        return jsonify(error='Mã lượt chạy không hợp lệ (404)'), 404
    return jsonify(stops=list_job_stops(job_id))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    browser_timer = Timer(1.0, lambda: webbrowser.open_new_tab('http://127.0.0.1:8000'))
    browser_timer.daemon = True
    browser_timer.start()
    app.run(host='127.0.0.1', port=8000, debug=False, threaded=True)
