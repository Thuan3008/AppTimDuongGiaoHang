/**
 * Route Optimization, Algorithms, Polling & Result Rendering
 */

import { $, api, fmt, clock, status, download, algorithmLabel } from './api.js';
import { getCurrentUser } from './auth.js';
import { renderPoints, renderRouteLayers, focusRoute, clearLines, routeColors, fit } from './map.js';
import {
  assignRoute,
  getCurrentJobId,
  setCurrentJobId,
  setCurrentTrip,
  getCurrentTrip,
  getAssignedTrip,
  statusLabel,
} from './trips.js';
import { resolveAllAddresses, collectCustomPoints, setManualMode, addDeliveryPoint } from './geocoding.js';

let payload = null;
let catalog = [];
let busy = false;
let startedAt = 0;
let pollTimer = null;
let deliveryStatuses = [];

export function getPayload() {
  return payload;
}

export function setCatalog(cat) {
  catalog = cat;
}

export function getCatalog() {
  return catalog;
}

export function isBusy() {
  return busy;
}

export function getStartedAt() {
  return startedAt;
}

export function setBusy(value) {
  busy = value;
  const runBtn = $('run');
  const datasetSelect = $('dataset');
  if (runBtn) runBtn.disabled = value;
  if (datasetSelect) datasetSelect.disabled = value;
  for (const el of document.querySelectorAll('aside.sidebar input')) {
    el.disabled = value;
  }
}

export function clearResult() {
  payload = null;
  deliveryStatuses = [];
  setCurrentTrip(null);
  setCurrentJobId(null);
  clearLines();

  $('result-algorithm')?.classList.add('hidden');
  $('comparison')?.classList.add('hidden');
  $('stops-panel')?.classList.add('hidden');
  $('itinerary-panel')?.classList.add('hidden');
  $('routes')?.replaceChildren();
  $('itinerary-routes')?.replaceChildren();

  for (const id of ['fitness', 'route-count', 'runtime', 'best-seed']) {
    const el = $(id);
    if (el) el.textContent = '—';
  }

  const warnings = $('warnings');
  if (warnings) warnings.textContent = '';
  if ($('json')) $('json').disabled = true;
  if ($('csv')) $('csv').disabled = true;
}

export function renderAlgorithms(names, selected) {
  const box = $('algorithm-list');
  if (!box) return;
  box.replaceChildren();

  for (const name of names) {
    const label = document.createElement('label');
    label.className = 'check';

    const input = document.createElement('input');
    input.type = 'checkbox';
    input.name = 'algorithm';
    input.value = name;
    input.checked = selected ? selected.includes(name) : name === names[0];

    label.append(input, ` ${algorithmLabel(name)}`);
    box.appendChild(label);
  }
}

export async function preview(onPointsLoaded) {
  clearResult();
  const datasetValue = $('dataset').value;
  const entry = catalog.find((ds) => ds.id === datasetValue);
  renderAlgorithms(entry ? entry.algorithms : [], null);

  setBusy(true);
  try {
    const data = await api('/api/datasets/' + datasetValue);
    renderPoints(data.points);
    if (onPointsLoaded) onPointsLoaded(data.points);
    status(`Đã nạp ${data.customers} khách hàng · Sức chứa ${data.capacity}`);
  } finally {
    setBusy(false);
  }
}

function cell(row, text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
  return td;
}

function deliveryStatusLabel(value) {
  return {
    completed: 'Đã giao',
    pending: 'Chờ giao',
  }[value] || 'Chờ giao';
}

function renderDetailedItinerary(routes) {
  const panel = $('itinerary-panel');
  const box = $('itinerary-routes');
  if (!panel || !box || !payload) return;

  box.replaceChildren();
  const depot = payload.points.find((point) => point.depot);
  const depotId = Number(depot?.id ?? 0);
  const points = new Map(payload.points.map((point) => [Number(point.id), point]));
  const statuses = new Map();
  for (const stop of deliveryStatuses) {
    const cid = Number(stop.customer_id);
    statuses.set(cid, stop.status);
    if (stop.route_index != null) {
      statuses.set(`${stop.route_index}_${cid}`, stop.status);
    }
  }
  let deliveryCount = 0;

  routes.forEach((route, routeIndex) => {
    const stops = (route.stops || []).filter((stop) => Number(stop.customer_id) !== depotId);
    if (!stops.length) return;
    deliveryCount += stops.length;

    const section = document.createElement('section');
    section.className = 'itinerary-route';

    const heading = document.createElement('button');
    heading.type = 'button';
    heading.className = 'itinerary-route-heading';
    heading.onclick = () => focusRoute(routeIndex);

    const dot = document.createElement('span');
    dot.className = 'swatch';
    dot.style.backgroundColor = routeColors[routeIndex % routeColors.length];

    const title = document.createElement('strong');
    title.textContent = `Xe ${route.vehicle}`;

    const summary = document.createElement('span');
    summary.textContent = `${stops.length} điểm · tải ${fmt(route.load)} · ${clock(route.departure_time)}–${clock(route.end_time)}`;
    heading.append(dot, title, summary);

    const wrap = document.createElement('div');
    wrap.className = 'table-wrap itinerary-table-wrap';
    const table = document.createElement('table');
    table.className = 'itinerary-table';
    const thead = document.createElement('thead');
    const header = document.createElement('tr');
    ['STT', 'Điểm giao', 'Địa chỉ / tọa độ', 'Nhu cầu', 'ETA', 'Phục vụ', 'Rời điểm', 'Từ điểm trước', 'Trạng thái']
      .forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
      });
    thead.appendChild(header);

    const tbody = document.createElement('tbody');
    stops.forEach((stop, index) => {
      const customerId = Number(stop.customer_id);
      const point = points.get(customerId) || {};
      const leg = (route.legs || []).find((item) => Number(item.to_customer_id) === customerId);
      const statusValue = statuses.get(`${routeIndex}_${customerId}`) ?? statuses.get(customerId);
      const row = document.createElement('tr');

      cell(row, index + 1);
      cell(row, point.name || `Khách hàng ${customerId}`, 'itinerary-customer');
      const address = String(point.address ?? '').trim();
      const coordinates = Number.isFinite(point.lat) && Number.isFinite(point.lon)
        ? `${point.lat.toFixed(6)}, ${point.lon.toFixed(6)}`
        : '—';
      cell(row, address || coordinates, 'itinerary-address');
      cell(row, fmt(stop.demand ?? point.demand ?? 0));
      cell(row, clock(stop.arrival_time));
      cell(row, `${fmt(stop.service_time)} s`);
      cell(row, clock(stop.departure_time));
      cell(row, `${fmt(leg?.travel_time ?? 0)} s`);

      const statusCell = cell(row, '');
      const badge = document.createElement('span');
      badge.className = `delivery-status ${statusValue === 'completed' ? 'completed' : 'pending'}`;
      badge.textContent = deliveryStatusLabel(statusValue);
      statusCell.appendChild(badge);
      tbody.appendChild(row);
    });

    table.append(thead, tbody);
    wrap.appendChild(table);
    section.append(heading, wrap);
    box.appendChild(section);
  });

  panel.classList.toggle('hidden', deliveryCount === 0);
}

export function updateDetailedItineraryStatuses(stops = []) {
  deliveryStatuses = Array.isArray(stops) ? stops : [];
  const select = $('result-algorithm');
  const result = select && payload ? payload.results[select.value] : null;
  if (result) renderDetailedItinerary(result.routes || []);
}

export function showAlgorithm() {
  const select = $('result-algorithm');
  if (!select || !payload) return;
  const name = select.value;
  const result = payload.results[name];
  if (!result) return;

  const routesBox = $('routes');
  routesBox.replaceChildren();

  const isDriver = getCurrentUser()?.role === 'driver';
  const deliveryDuration = result.routes.reduce((total, route) => total + Number(route.duration), 0);
  $('fitness').textContent = fmt(isDriver ? deliveryDuration : result.fitness);
  $('runtime').textContent = fmt(result.runtime_seconds);
  $('route-count').textContent = result.routes.length;
  $('best-seed').textContent = result.seed ?? '—';

  renderRouteLayers(result.routes);
  renderDetailedItinerary(result.routes);

  const currentUser = getCurrentUser();
  const currentJobId = getCurrentJobId();
  const currentTrip = getCurrentTrip();

  result.routes.forEach((r, i) => {
    const color = routeColors[i % routeColors.length];
    const row = document.createElement('tr');
    row.className = 'route-row';

    const label = cell(row, `Xe ${r.vehicle}`);
    const dot = document.createElement('span');
    dot.className = 'swatch';
    dot.style.backgroundColor = color;
    label.prepend(dot);

    cell(row, r.visits.join(' → '), 'visits');
    cell(row, fmt(r.load));
    cell(row, clock(r.departure_time));
    cell(row, fmt(r.travel_time));
    cell(row, fmt(r.service_time));
    cell(row, fmt(r.duration));
    cell(
      row,
      r.geometry_source === 'osrm'
        ? 'OSRM'
        : r.geometry_source === 'dataset-road-graph'
        ? 'Road graph dataset'
        : 'Đường thẳng dự phòng'
    );

    const actionCell = cell(row, '');
    const assigned = currentJobId ? getAssignedTrip(currentJobId, i) : null;

    if (assigned) {
      const badge = document.createElement('span');
      badge.className = 'delivery-status completed';
      badge.textContent = `Đã giao: ${assigned.driver_username}`;
      badge.title = `Trạng thái: ${statusLabel(assigned.status)} · Đã giao: ${assigned.stops_completed}/${assigned.stops_total} điểm`;
      actionCell.appendChild(badge);
    } else if (currentUser && currentUser.role === 'admin' && currentJobId && r.visits.length > 2) {
      const driverSelect = document.createElement('select');
      const driverSource = $('driver');
      if (driverSource) {
        driverSelect.replaceChildren(
          ...[...driverSource.options].map((opt) => new Option(opt.text, opt.value))
        );
      }
      const assignBtn = document.createElement('button');
      assignBtn.textContent = 'Giao Driver';
      assignBtn.onclick = async (event) => {
        event.stopPropagation();
        await assignRoute(name, i, Number(driverSelect.value), async () => {
          if (window.reloadTripsApp) await window.reloadTripsApp();
          showAlgorithm();
        });
      };
      actionCell.append(driverSelect, assignBtn);
    } else if (currentTrip) {
      actionCell.textContent = currentTrip.driver_username;
    } else {
      actionCell.textContent = 'Route rỗng';
    }

    row.onclick = () => focusRoute(i);
    routesBox.appendChild(row);
  });

  const fallback = result.routes.filter((r) => r.geometry_source !== 'osrm');
  const warnEl = $('warnings');
  if (warnEl) {
    warnEl.textContent = fallback.length
      ? `${fallback.length} tuyến dùng đường dự phòng vì OSRM không khả dụng. ${fallback[0].warning || ''}`
      : '';
  }

  const evalNote = $('evaluation-note');
  if (evalNote) {
    evalNote.textContent = isDriver
      ? 'Tổng thời gian giao hàng dự kiến gồm thời gian di chuyển và phục vụ, tính từ lúc khởi hành đến khi về kho. Thời gian thực tế có thể thay đổi theo quá trình giao hàng.'
      : `${result.seed != null ? `Nghiệm đang hiển thị thuộc seed ${result.seed}` : 'Nghiệm chạy tất định (không dùng seed)'} · Thời gian từ dataset · Departure evaluator: ${result.departure_time_mode}. Đường OSRM chỉ phục vụ hiển thị và có thể khác đường thực nghiệm. Duration gồm thời gian từ khởi hành đến về kho; objective = di chuyển + phục vụ.`;
  }

  if ($('csv')) $('csv').disabled = false;
}

export function renderResult(data, stops = []) {
  payload = data;
  deliveryStatuses = Array.isArray(stops) ? stops : [];
  renderPoints(data.points);

  const names = Object.keys(data.results);
  const select = $('result-algorithm');
  if (select) {
    select.replaceChildren(...names.map((name) => new Option(algorithmLabel(name), name)));
    select.classList.toggle('hidden', !names.length);
  }

  const compBox = $('comparison-rows');
  const isDriver = getCurrentUser()?.role === 'driver';
  if (compBox) {
    compBox.replaceChildren();
    for (const name of isDriver ? [] : names) {
      const r = data.results[name];
      const row = document.createElement('tr');
      [algorithmLabel(name), r.seed ?? '—', fmt(r.fitness), r.nfe, fmt(r.runtime_seconds), r.departure_time_mode].forEach((v) =>
        cell(row, v)
      );
      compBox.appendChild(row);
    }
  }

  const compPanel = $('comparison');
  if (compPanel) compPanel.classList.toggle('hidden', isDriver || !names.length);

  if ($('json')) $('json').disabled = false;

  if (names.length) showAlgorithm();

  const errors = Object.entries(data.errors || {})
    .map(([a, e]) => `${algorithmLabel(a)}: ${e}`)
    .join(' | ');
  status(errors || 'Hoàn tất · Kết quả đã lưu trên máy', Boolean(errors));
}

export async function poll(id, onDone) {
  try {
    const job = await api('/api/runs/' + id);
    status(job.stage);

    if (job.payload) {
      setCurrentJobId(id);
      setCurrentTrip(null);
      renderResult(job.payload);
      setBusy(false);
      if (onDone) onDone(job.payload);
      return;
    }

    if (job.status === 'failed') {
      setBusy(false);
      status(job.stage, true);
      return;
    }

    pollTimer = setTimeout(() => poll(id, onDone), 1500);
  } catch (error) {
    setBusy(false);
    status(`${error.message}. Tải lại trang để kết nối lại lượt chạy.`, true);
  }
}

export async function startOptimization(onJobCreated) {
  const [hours, minutes] = ($('earliest-departure').value || '08:00').split(':').map(Number);
  const config = {
    trip_name: $('trip-name').value.trim(),
    dataset: $('dataset').value,
    algorithms: [...document.querySelectorAll('[name=algorithm]:checked')].map((el) => el.value),
    population_size: Number($('population').value),
    budget: Number($('budget').value),
    seed: Number($('seed').value),
    num_seeds: Number($('num-seeds').value),
    earliest_departure: hours * 3600 + minutes * 60,
  };

  try {
    if ($('manual-toggle').checked) {
      status('Đang tìm địa chỉ…');
      await resolveAllAddresses();
      config.custom_points = collectCustomPoints();
    }
  } catch (error) {
    return status(error.message, true);
  }

  if (!config.trip_name) return status('Nhập tên chuyến', true);
  if (!config.algorithms.length) return status('Chọn ít nhất một thuật toán', true);
  if (config.budget < config.population_size) return status('NFE phải lớn hơn hoặc bằng P', true);

  clearResult();
  setBusy(true);
  startedAt = Date.now();
  status('Đang tối ưu các route…');

  try {
    const job = await api('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    setCurrentJobId(job.id);
    try {
      localStorage.setItem('rapo-map-job', job.id);
    } catch {}

    if (onJobCreated) onJobCreated(job.id);
    await poll(job.id);
  } catch (error) {
    setBusy(false);
    status(error.message, true);
  }
}

export async function applyConfig(config) {
  $('dataset').value = config.dataset;
  $('trip-name').value = config.trip_name || `Chuyến ${config.dataset}`;
  $('population').value = config.population_size;
  $('budget').value = config.budget;
  $('seed').value = config.seed;
  $('num-seeds').value = config.num_seeds || 1;

  const seconds = config.earliest_departure == null ? 8 * 3600 : config.earliest_departure;
  $('earliest-departure').value = `${String(Math.floor(seconds / 3600)).padStart(2, '0')}:${String(
    Math.floor((seconds % 3600) / 60)
  ).padStart(2, '0')}`;

  const custom = config.custom_points;
  $('manual-toggle').checked = Boolean(custom);
  $('delivery-points').replaceChildren();

  if (custom) {
    $('manual-vehicles').value = custom.num_vehicles;
    $('manual-capacity').value = custom.vehicle_capacity;
    $('depot-name').value = custom.depot.name || 'Kho';
    $('depot-address').value = custom.depot.address || '';
    $('depot-lat').value = custom.depot.lat;
    $('depot-lon').value = custom.depot.lon;
    if (custom.depot.address) $('depot-address').dataset.resolvedAddress = custom.depot.address;
    if (custom.depot.snap_status) {
      $('depot-address').dataset.snapCheckedLat = String(custom.depot.lat);
      $('depot-address').dataset.snapCheckedLon = String(custom.depot.lon);
      $('depot-address').dataset.snapConfirmed = custom.depot.snap_confirmed ? 'true' : 'false';
    }
    for (const point of custom.customers) addDeliveryPoint(point);
  }

  await setManualMode(Boolean(custom), catalog, { refresh: false });
  renderAlgorithms(
    catalog.find((ds) => ds.id === config.dataset)?.algorithms || [],
    config.algorithms
  );
}

export function exportJSON() {
  if (!payload) return;
  download(
    JSON.stringify(payload, null, 2),
    'application/json',
    `rapo_map_${payload.config.dataset}_seed${payload.config.seed}.json`
  );
}

export function exportCSV() {
  if (!payload) return;
  const name = $('result-algorithm').value;
  const rows = [
    [
      'vehicle',
      'visits',
      'departure_seconds',
      'travel_seconds',
      'service_seconds',
      'duration_seconds',
      'objective_seconds',
      'edge_path',
      'geometry_source',
    ],
    ...payload.results[name].routes.map((r) => [
      r.vehicle,
      r.visits.join(' -> '),
      r.departure_time,
      r.travel_time,
      r.service_time,
      r.duration,
      r.objective,
      (r.edge_path || []).join(' -> '),
      r.geometry_source,
    ]),
  ];

  download(
    '\ufeff' +
      rows
        .map((row) =>
          row.map((v) => '"' + String(v).replaceAll('"', '""') + '"').join(',')
        )
        .join('\r\n'),
    'text/csv',
    `${algorithmLabel(name).toLowerCase().replaceAll('-', '_')}_${payload.config.dataset}_routes.csv`
  );
}
