"""Kiểm tra API, tọa độ và đường dự phòng của app bản đồ, không gọi Internet."""

import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from core.data.models import Customer, RoadNode, TDVRPInstance
from web_map import app as backend
from web_map.auth import create_user, init_db


def write_completed_job(job_id, routes=None):
    routes = routes or [
        {'vehicle': 1, 'visits': [0, 1, 2, 0], 'load': 2, 'departure_time': 28800,
         'end_time': 29400, 'travel_time': 400, 'service_time': 200, 'waiting_time': 0,
         'duration': 600, 'objective': 600, 'node_path': [0, 1, 2, 0],
         'edge_path': ['A', 'B', 'C'], 'dataset_distance_m': 1000,
         'geometry': {'type': 'LineString', 'coordinates': [[106, 10], [106.1, 10.1]]},
         'geometry_source': 'dataset-road-graph', 'warning': None, 'legs': [], 'stops': []},
    ]
    payload = {
        'config': {'dataset': 'hcmc10', 'trip_name': 'Chuyến kiểm thử',
                   'population_size': 4, 'budget': 8, 'seed': 42,
                   'num_seeds': 1, 'earliest_departure': 28800, 'algorithms': ['HCMC-TD-ERAPO']},
        'points': [
            {'id': customer_id, 'lat': 10 + customer_id / 100, 'lon': 106,
             'demand': 0 if customer_id == 0 else 1, 'depot': customer_id == 0}
            for customer_id in range(5)
        ],
        'results': {'HCMC-TD-ERAPO': {'fitness': 600, 'nfe': 8, 'runtime_seconds': 1,
                                       'departure_time_mode': 'exact_ndcpwlf', 'routes': routes}},
        'errors': {},
    }
    (backend.OUTPUT / f'{job_id}.json').write_text(json.dumps(payload), encoding='utf-8')
    return payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    backend.app.config.update(
        TESTING=True,
        DATABASE=str(tmp_path / 'auth.sqlite3'),
        SECRET_KEY='test-secret-key',
    )
    output = tmp_path / 'results'
    output.mkdir()
    monkeypatch.setattr(backend, 'OUTPUT', output)
    backend.JOBS.clear()
    with backend.app.app_context():
        init_db(seed_defaults=False)
        create_user('admin_test', 'admin-pass-123', 'admin')
        create_user('driver_test', 'driver-pass-123', 'driver')
    test_client = backend.app.test_client()
    response = test_client.post('/api/auth/login', json={
        'username': 'admin_test', 'password': 'admin-pass-123',
    })
    assert response.status_code == 200
    return test_client


@pytest.mark.parametrize('changes', [
    {'dataset': '../secret'}, {'dataset': []}, {'algorithms': ['UNKNOWN']},
    {'algorithms': []}, {'population_size': True}, {'budget': 1},
    {'budget': 2}, {'seed': -1},
])
def test_api_rejects_invalid_run_before_worker(client, changes):
    config = dict(dataset='hk25', driver_id=2, algorithms=['RAPO-2709'], population_size=10, budget=290, seed=42)
    config.update(changes)
    assert client.post('/api/runs', json=config).status_code == 400


def test_no_arbitrary_file_access(client):
    assert client.get('/api/runs/not-a-job').status_code == 404
    assert client.get('/api/datasets/secret').status_code == 404


def test_osrm_splits_without_losing_visits(monkeypatch):
    calls = []
    def fake_segment(points):
        calls.append(points)
        return list(points)
    monkeypatch.setattr(backend, 'osrm_segment', fake_segment)
    points = [(114 + i / 1000, 22.3) for i in range(105)]
    geometry = backend.osrm_geometry(points)
    assert geometry['coordinates'] == points
    assert [len(c) for c in calls] == [50, 50, 7]
    assert calls[0][-1] == calls[1][0]


def test_osrm_failure_keeps_dataset_metrics_and_lon_lat(monkeypatch):
    def fail(points):
        raise TimeoutError('OSRM timeout')
    monkeypatch.setattr(backend, 'osrm_geometry', fail)
    nodes = {10: SimpleNamespace(lat=22.3, lon=114.1), 20: SimpleNamespace(lat=22.4, lon=114.2)}
    instance = SimpleNamespace(depot=0, road_node_ids=[10, 20])
    route = SimpleNamespace(customer_ids=[1], node_path=[10, 20, 10], edge_path=['E1', 'E1'],
                            legs=[], vehicle_index=0,
                            load=7, start_time=60, end_time=360, travel_time=200,
                            service_time=90, waiting_time=10, duration=300, distance=800, stops=[])
    result = backend.route_payload(route, instance, (nodes, {}))
    assert result['visits'] == [0, 1, 0]
    assert result['geometry_source'] == 'dataset-road-graph'
    assert result['geometry']['coordinates'][0] == (114.1, 22.3)
    assert result['objective'] == 290
    assert result['duration'] == 300
    assert result['departure_time'] == 60
    assert result['edge_path'] == ['E1', 'E1']


def test_completed_job_survives_restart(client, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, 'OUTPUT', tmp_path)
    job_id = 'a' * 32
    (tmp_path / f'{job_id}.json').write_text('{"config": {}, "results": {"RAPO-2709": {}}}', encoding='utf-8')
    response = client.get('/api/runs/' + job_id)
    assert response.status_code == 200
    assert response.json['status'] == 'completed'


def test_anonymous_user_cannot_read_data_or_start_run(tmp_path):
    backend.app.config.update(TESTING=True, DATABASE=str(tmp_path / 'anonymous.sqlite3'))
    with backend.app.app_context():
        init_db(seed_defaults=False)
    anonymous = backend.app.test_client()
    assert anonymous.get('/api/datasets').status_code == 401
    assert anonymous.post('/api/runs', json={}).status_code == 401


def test_driver_cannot_start_optimization(client):
    client.post('/api/auth/logout')
    assert client.post('/api/auth/login', json={
        'username': 'driver_test', 'password': 'driver-pass-123',
    }).status_code == 200
    assert client.post('/api/runs', json={}).status_code == 403


def test_admin_creates_driver_account(client):
    response = client.post('/api/drivers', json={
        'username': 'driver_two', 'password': 'driver-two-pass',
    })
    assert response.status_code == 201
    assert response.json['driver']['role'] == 'driver'
    usernames = {item['username'] for item in client.get('/api/drivers').json['drivers']}
    assert 'driver_two' in usernames


def test_driver_sees_only_assigned_trip_and_updates_in_order(client, monkeypatch):
    monkeypatch.setattr(backend.EXECUTOR, 'submit', lambda *args, **kwargs: None)
    config = {
        'dataset': 'hcmc10', 'algorithms': ['HCMC-TD-ERAPO'],
        'population_size': 4, 'budget': 8, 'seed': 42, 'num_seeds': 1,
        'earliest_departure': 8 * 3600,
    }
    created = client.post('/api/runs', json=config)
    assert created.status_code == 202
    job_id = created.json['id']
    write_completed_job(job_id)
    assigned = client.post('/api/trips', json={
        'job_id': job_id, 'algorithm': 'HCMC-TD-ERAPO', 'route_index': 0, 'driver_id': 2,
    })
    assert assigned.status_code == 201
    trip_id = assigned.json['trip']['id']

    client.post('/api/auth/logout')
    client.post('/api/auth/login', json={
        'username': 'driver_test', 'password': 'driver-pass-123',
    })
    trips = client.get('/api/trips').json['trips']
    assert [trip['id'] for trip in trips] == [trip_id]
    assert trips[0]['plan']['dataset'] == 'hcmc10'
    assert trips[0]['plan']['trip_name'] == 'Chuyến kiểm thử'
    assert trips[0]['plan']['departure_time'] == 28800
    assert client.get('/api/runs/' + job_id).status_code == 403
    detail = client.get('/api/trips/' + trip_id)
    assert detail.status_code == 200
    assert len(detail.json['payload']['results']['HCMC-TD-ERAPO']['routes']) == 1
    assert {point['id'] for point in detail.json['payload']['points']} == {0, 1, 2}
    assert client.patch('/api/trips/' + trip_id + '/status', json={'status': 'completed'}).status_code == 400
    assert client.patch('/api/trips/' + trip_id + '/stops/0').status_code == 400
    assert client.patch('/api/trips/' + trip_id + '/status', json={'status': 'in_progress'}).status_code == 200
    assert client.patch('/api/trips/' + trip_id + '/stops/1').status_code == 400
    first = client.patch('/api/trips/' + trip_id + '/stops/0')
    assert first.status_code == 200
    assert first.json['trip']['status'] == 'in_progress'
    repeated = client.patch('/api/trips/' + trip_id + '/stops/0')
    assert repeated.status_code == 200
    assert repeated.json['stops'][0]['status'] == 'completed'
    final = client.patch('/api/trips/' + trip_id + '/stops/1')
    assert final.status_code == 200
    assert final.json['trip']['status'] == 'completed'
    assert all(stop['status'] == 'completed' for stop in final.json['stops'])


def test_other_driver_cannot_open_assigned_job(client, monkeypatch):
    monkeypatch.setattr(backend.EXECUTOR, 'submit', lambda *args, **kwargs: None)
    created = client.post('/api/runs', json={
        'dataset': 'hcmc10', 'algorithms': ['HCMC-TD-ERAPO'],
        'population_size': 4, 'budget': 8, 'seed': 42,
    })
    assert created.status_code == 202
    write_completed_job(created.json['id'])
    assigned = client.post('/api/trips', json={
        'job_id': created.json['id'], 'algorithm': 'HCMC-TD-ERAPO', 'route_index': 0, 'driver_id': 2,
    })
    trip_id = assigned.json['trip']['id']
    client.post('/api/drivers', json={'username': 'other_driver', 'password': 'other-pass-123'})
    client.post('/api/auth/logout')
    client.post('/api/auth/login', json={'username': 'other_driver', 'password': 'other-pass-123'})
    assert client.get('/api/runs/' + created.json['id']).status_code == 403
    assert client.get('/api/trips/' + trip_id).status_code == 403
    assert client.get('/api/trips').json['trips'] == []


def test_multi_vehicle_result_assigns_each_route_to_a_different_driver(client, monkeypatch):
    monkeypatch.setattr(backend.EXECUTOR, 'submit', lambda *args, **kwargs: None)
    second = client.post('/api/drivers', json={
        'username': 'driver_second', 'password': 'driver-second-pass',
    }).json['driver']
    created = client.post('/api/runs', json={
        'dataset': 'hcmc10', 'algorithms': ['HCMC-TD-ERAPO'],
        'population_size': 4, 'budget': 8, 'seed': 42,
    })
    job_id = created.json['id']
    routes = [
        {'vehicle': 1, 'visits': [0, 1, 2, 0]},
        {'vehicle': 2, 'visits': [0, 3, 4, 0]},
    ]
    write_completed_job(job_id, routes)
    first = client.post('/api/trips', json={
        'job_id': job_id, 'algorithm': 'HCMC-TD-ERAPO', 'route_index': 0, 'driver_id': 2,
    })
    other = client.post('/api/trips', json={
        'job_id': job_id, 'algorithm': 'HCMC-TD-ERAPO', 'route_index': 1, 'driver_id': second['id'],
    })
    assert first.status_code == other.status_code == 201
    duplicate = client.post('/api/trips', json={
        'job_id': job_id, 'algorithm': 'HCMC-TD-ERAPO', 'route_index': 0, 'driver_id': second['id'],
    })
    assert duplicate.status_code == 409

    client.post('/api/auth/logout')
    client.post('/api/auth/login', json={'username': 'driver_test', 'password': 'driver-pass-123'})
    driver_routes = client.get('/api/trips').json['trips']
    assert [(trip['route_index'], trip['stops_total']) for trip in driver_routes] == [(0, 2)]

    # Driver starts trip and completes stop 0
    trip_id = first.json['trip']['id']
    client.patch(f'/api/trips/{trip_id}/status', json={'status': 'in_progress'})
    patch_res = client.patch(f'/api/trips/{trip_id}/stops/0')
    assert patch_res.status_code == 200

    # Admin views full job and sees the synchronized stop status
    client.post('/api/auth/logout')
    client.post('/api/auth/login', json={'username': 'admin_test', 'password': 'admin-pass-123'})
    full_run = client.get('/api/runs/' + job_id)
    assert full_run.status_code == 200
    assert 'stops' in full_run.json
    completed_stops = [s for s in full_run.json['stops'] if s['status'] == 'completed']
    assert len(completed_stops) == 1
    assert completed_stops[0]['customer_id'] == 1

    run_stops_only = client.get(f'/api/runs/{job_id}/stops')
    assert run_stops_only.status_code == 200
    assert len([s for s in run_stops_only.json['stops'] if s['status'] == 'completed']) == 1


def test_admin_previews_and_queues_manually_entered_delivery_points(client, monkeypatch):
    dataset_points = client.get('/api/datasets/hcmc10').json['points']
    custom_points = {
        'depot': {'name': 'Kho Tân Bình', 'address': 'Kho Tân Bình, TP.HCM',
                  'lat': dataset_points[0]['lat'], 'lon': dataset_points[0]['lon']},
        'num_vehicles': 1,
        'vehicle_capacity': 10,
        'customers': [{
            'name': 'Cửa hàng A', 'address': '140 Lê Trọng Tấn, TP.HCM',
            'lat': dataset_points[1]['lat'], 'lon': dataset_points[1]['lon'],
            'demand': 2, 'service_time': 300, 'time_window': [28800, 43200],
        }],
    }
    preview = client.post('/api/custom-points/preview', json={
        'dataset': 'hcmc10', 'trip_name': 'Ca sáng', 'custom_points': custom_points,
    })
    assert preview.status_code == 200
    assert preview.json['customers'] == 1
    assert preview.json['vehicles'] == 1
    assert [point['name'] for point in preview.json['points']] == ['Kho Tân Bình', 'Cửa hàng A']
    assert preview.json['points'][1]['address'] == '140 Lê Trọng Tấn, TP.HCM'
    assert preview.json['points'][1]['snap_distance_m'] == 0

    monkeypatch.setattr(backend.EXECUTOR, 'submit', lambda *args, **kwargs: None)
    created = client.post('/api/runs', json={
        'trip_name': 'Ca sáng', 'dataset': 'hcmc10', 'algorithms': ['HCMC-TD-ERAPO'],
        'population_size': 4, 'budget': 8, 'seed': 42, 'num_seeds': 1,
        'earliest_departure': 28800, 'custom_points': custom_points,
    })
    assert created.status_code == 202
    queued = backend.JOBS[created.json['id']]['config']
    assert queued['trip_name'] == 'Ca sáng'
    assert queued['custom_points']['customers'][0]['service_time'] == 300


def test_manual_delivery_points_reject_insufficient_capacity(client, monkeypatch):
    monkeypatch.setattr(backend.EXECUTOR, 'submit', lambda *args, **kwargs: None)
    point = client.get('/api/datasets/hcmc10').json['points'][0]
    response = client.post('/api/runs', json={
        'trip_name': 'Quá tải', 'dataset': 'hcmc10', 'algorithms': ['HCMC-TD-ERAPO'],
        'population_size': 4, 'budget': 8, 'seed': 42,
        'custom_points': {
            'depot': {'lat': point['lat'], 'lon': point['lon']},
            'num_vehicles': 1, 'vehicle_capacity': 1,
            'customers': [{'lat': point['lat'], 'lon': point['lon'], 'demand': 2}],
        },
    })
    assert response.status_code == 400
    assert 'Nhu cầu điểm giao 1' in response.json['error']


@pytest.mark.parametrize(('distance_m', 'expected'), [
    (100, 'auto'),
    (100.1, 'confirm'),
    (300, 'confirm'),
    (300.1, 'rejected'),
])
def test_snap_policy_boundaries(distance_m, expected):
    assert backend._snap_policy(distance_m) == expected


def test_snap_point_api_reports_auto_confirm_and_rejected(client, monkeypatch):
    node = RoadNode(id=1, lat=10.0, lon=106.0)
    monkeypatch.setattr(backend, 'load_data', lambda _key: (None, ({1: node}, {}), None))

    def check(distance_m):
        return client.post('/api/snap-point', json={
            'dataset': 'hcmc10',
            'lat': node.lat + distance_m / 111_320,
            'lon': node.lon,
        })

    assert check(50).json['snap_status'] == 'auto'
    assert check(150).json['snap_status'] == 'confirm'
    assert check(350).json['snap_status'] == 'rejected'


def test_custom_instance_requires_confirmation_and_rejects_over_300m():
    node = RoadNode(id=1, lat=10.0, lon=106.0)
    base = TDVRPInstance(
        instance_name='test', benchmark_name='test', num_customers=1, num_vehicles=1,
        vehicle_capacity=10, coordinates=[(10.0, 106.0), (10.0, 106.0)],
        demands=[0, 1], service_times=[0, 0], road_node_ids=[1, 1], depot=0,
        reference_lla={'lat': 10.0, 'lon': 106.0, 'alt': 0.0}, horizon=(0, 86400),
        sample_step=300, graph_path='', traffic_path='', metadata={},
        customers=[Customer(1, 1, 1, 0, (10.0, 106.0))],
        time_windows=[(0, 86400), (0, 86400)],
    )

    def custom(distance_m, confirmed=False):
        return {
            'depot': {'lat': 10.0, 'lon': 106.0},
            'num_vehicles': 1,
            'vehicle_capacity': 10,
            'customers': [{
                'lat': 10.0 + distance_m / 111_320,
                'lon': 106.0,
                'demand': 1,
                'snap_confirmed': confirmed,
            }],
        }

    with pytest.raises(ValueError, match='cần xác nhận'):
        backend.build_custom_instance(base, ({1: node}, {}), custom(150), 'test')
    accepted = backend.build_custom_instance(base, ({1: node}, {}), custom(150, True), 'test')
    assert accepted.metadata['custom_points'][1]['snap_status'] == 'confirm'
    with pytest.raises(ValueError, match='vượt giới hạn 300 m'):
        backend.build_custom_instance(base, ({1: node}, {}), custom(350, True), 'test')


def test_admin_geocodes_address_and_returns_graph_distance(client, monkeypatch):
    point = client.get('/api/datasets/hcmc10').json['points'][0]
    response_body = json.dumps({'features': [{
        'geometry': {'type': 'Point', 'coordinates': [point['lon'], point['lat']]},
        'properties': {'name': 'Trường Đại học Công Thương', 'housenumber': '140',
                       'street': 'Lê Trọng Tấn', 'city': 'Hồ Chí Minh', 'country': 'Việt Nam'},
    }]}).encode()
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return BytesIO(response_body)

    backend.geocode_address.cache_clear()
    monkeypatch.setattr(backend, 'GEOCODE_LAST_REQUEST', 0.0)
    monkeypatch.setattr(backend, 'urlopen', fake_urlopen)
    response = client.post('/api/geocode', json={
        'dataset': 'hcmc10', 'address': '140 Lê Trọng Tấn',
    })
    assert response.status_code == 200
    assert response.json['result']['lat'] == point['lat']
    assert response.json['result']['snap_distance_m'] == 0
    assert response.json['result']['snap_status'] == 'auto'
    assert response.json['result']['provider'] == 'Photon'
    assert 'photon.komoot.io' in requested_urls[0]


def test_geocode_prefers_exact_house_number_across_both_providers(client, monkeypatch):
    point = client.get('/api/datasets/hcmc10').json['points'][0]
    photon = {'features': [{
        'geometry': {'type': 'Point', 'coordinates': [point['lon'], point['lat']]},
        'properties': {'name': 'Nhà thi đấu Tây Thạnh', 'housenumber': '97',
                       'street': 'Dương Đức Hiền', 'district': 'Tân Phú'},
    }]}
    nominatim = [{
        'lat': point['lat'], 'lon': point['lon'],
        'display_name': '242/11 Đường Tây Thạnh, Tân Phú, Hồ Chí Minh',
        'address': {'house_number': '242/11', 'road': 'Đường Tây Thạnh'},
    }]

    def fake_urlopen(request, timeout):
        payload = photon if 'photon.komoot.io' in request.full_url else nominatim
        return BytesIO(json.dumps(payload).encode())

    backend.geocode_address.cache_clear()
    monkeypatch.setattr(backend, 'GEOCODE_LAST_REQUEST', 0.0)
    monkeypatch.setattr(backend.time, 'sleep', lambda _seconds: None)
    monkeypatch.setattr(backend, 'urlopen', fake_urlopen)
    response = client.post('/api/geocode', json={
        'dataset': 'hcmc10', 'address': '242/11 Tây Thạnh, Tân Phú',
    })

    assert response.status_code == 200
    assert response.json['result']['provider'] == 'Nominatim'
    assert response.json['result']['house_match'] == 'exact'
    assert response.json['requires_confirmation'] is False
    assert {item['provider'] for item in response.json['candidates']} == {'Photon', 'Nominatim'}


def test_geocode_requires_confirmation_when_house_number_is_not_found(client, monkeypatch):
    point = client.get('/api/datasets/hcmc10').json['points'][0]
    photon = {'features': [{
        'geometry': {'type': 'Point', 'coordinates': [point['lon'], point['lat']]},
        'properties': {'name': 'Nhà thi đấu Tây Thạnh', 'housenumber': '97',
                       'street': 'Dương Đức Hiền', 'district': 'Tân Phú'},
    }]}

    def fake_urlopen(request, timeout):
        payload = photon if 'photon.komoot.io' in request.full_url else []
        return BytesIO(json.dumps(payload).encode())

    backend.geocode_address.cache_clear()
    monkeypatch.setattr(backend, 'GEOCODE_LAST_REQUEST', 0.0)
    monkeypatch.setattr(backend.time, 'sleep', lambda _seconds: None)
    monkeypatch.setattr(backend, 'urlopen', fake_urlopen)
    response = client.post('/api/geocode', json={
        'dataset': 'hcmc10', 'address': '242/11 Tây Thạnh, Tân Phú',
    })

    assert response.status_code == 200
    assert response.json['requires_confirmation'] is True
    assert response.json['result']['house_match'] == 'wrong'
    assert '242/11' in response.json['warning']


def test_root_and_static_assets_served(client):
    res = client.get('/')
    assert res.status_code == 200
    assert b'RAPO' in res.data
    assert b'/static/css/layout.css' in res.data

    static_assets = [
        '/static/css/variables.css',
        '/static/css/layout.css',
        '/static/css/components.css',
        '/static/js/app.js',
        '/static/js/api.js',
        '/static/js/auth.js',
        '/static/js/map.js',
        '/static/js/trips.js',
        '/static/js/geocoding.js',
        '/static/js/optimizer.js',
    ]
    for asset in static_assets:
        res = client.get(asset)
        assert res.status_code == 200, f'Asset {asset} failed with {res.status_code}'


def test_hcmc_dataset_uses_and_runs_hcmc_td_erapo(client, monkeypatch):
    catalog = client.get('/api/datasets').json['datasets']
    hcmc_entries = [item for item in catalog if item['id'].startswith('hcmc')]
    assert hcmc_entries
    assert all(item['algorithms'] == ['HCMC-TD-ERAPO'] for item in hcmc_entries)

    # Mock OSRM so route_payload falls back cleanly to local dataset road-graph offline
    def fail_osrm(_points):
        raise RuntimeError('offline test')

    monkeypatch.setattr(backend, 'osrm_geometry', fail_osrm)
    job_id = 'b' * 32
    backend.JOBS[job_id] = {'id': job_id, 'status': 'queued', 'stage': 'test'}
    cfg = {
        'dataset': 'hcmc10',
        'trip_name': 'Chạy thực tế HCMC-TD-ERAPO',
        'population_size': 4,
        'budget': 10,
        'seed': 42,
        'num_seeds': 1,
        'earliest_departure': 28800,
        'algorithms': ['HCMC-TD-ERAPO'],
    }
    backend.run_job(job_id, cfg)
    job = backend.JOBS[job_id]
    assert job['status'] == 'completed', job.get('stage')
    res = job['payload']['results']['HCMC-TD-ERAPO']
    assert res['feasible'] is True
    assert res['nfe'] == 10
    assert res['departure_time_mode'] == 'exact_ndcpwlf'
    assert len(res['routes']) == 1
    assert res['routes'][0]['geometry_source'] == 'dataset-road-graph'
