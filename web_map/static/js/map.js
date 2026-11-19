/**
 * Leaflet Map Controller: Markers, Polylines, Layers & Coordinate Picking
 */

import { $, fmt, clock } from './api.js';

export const routeColors = [
  '#2563eb', // Royal Blue
  '#ea580c', // Bright Orange
  '#7c3aed', // Violet
  '#059669', // Emerald
  '#db2777', // Pink
  '#ca8a04', // Amber
  '#475569', // Slate
];

let map = null;
let markers = null;
let lines = null;
let points = [];
let routeLayers = [];
let pendingMapSelection = null;

export function getMapInstance() {
  return map;
}

export function getPoints() {
  return points;
}

export function setPendingMapSelection(selection) {
  pendingMapSelection = selection;
  if (map) {
    map.getContainer().style.cursor = selection ? 'crosshair' : '';
  }
}

export function getPendingMapSelection() {
  return pendingMapSelection;
}

export function initializeMap(onMapClickPick) {
  if (map) return;
  if (!window.L) {
    throw Error('Không tải được Leaflet. Kiểm tra kết nối Internet rồi tải lại trang.');
  }

  // Tọa độ trung tâm mặc định: TP.HCM
  map = L.map('map', {
    zoomControl: true,
  }).setView([10.79, 106.66], 12);

  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map).on('tileerror', () => {
    const warn = $('tile-warning');
    if (warn) warn.textContent = 'Không tải được một số ô nền OpenStreetMap. Các điểm và đường tuyến vẫn hiển thị.';
  });

  markers = L.layerGroup().addTo(map);
  lines = L.layerGroup().addTo(map);

  map.on('click', async (event) => {
    if (!pendingMapSelection) return;
    const target = pendingMapSelection;

    const candidate = {
      lat: event.latlng.lat,
      lon: event.latlng.lng,
      display_name: 'Vị trí chọn trên bản đồ',
      provider: 'Bản đồ',
      house_match: 'exact',
    };

    const accepted = onMapClickPick
      ? await onMapClickPick(target, candidate, event.latlng)
      : false;
    if (!accepted) return;

    pendingMapSelection = null;
    map.getContainer().style.cursor = '';
    L.circleMarker(event.latlng, {
      radius: 8,
      color: '#dc2626',
      fillColor: '#ef4444',
      fillOpacity: 0.85,
      weight: 2,
    }).addTo(markers);
  });

  // Tự động invalidate size khi container thay đổi kích thước
  setTimeout(() => map.invalidateSize(), 200);
}

export function fit() {
  if (map && points.length) {
    map.fitBounds(points.map((p) => [p.lat, p.lon]), {
      padding: [40, 40],
      maxZoom: 15,
    });
  }
}

export function renderPoints(data) {
  points = data;
  if (!markers) return;
  markers.clearLayers();

  for (const p of points) {
    const isDepot = Boolean(p.depot);
    const icon = L.divIcon({
      className: '',
      html: `<div class="pin ${isDepot ? 'depot' : ''}">${isDepot ? 'K' : p.id}</div>`,
      iconSize: isDepot ? [34, 34] : [26, 26],
      iconAnchor: isDepot ? [17, 17] : [13, 13],
    });

    const content = document.createElement('div');
    const address = p.address ? ` | ${p.address}` : '';
    const snap = p.snap_distance_m ? ` | ghép đường: ${fmt(p.snap_distance_m)} m` : '';
    content.textContent = `${p.name || (isDepot ? 'Kho / Depot' : 'Khách hàng ' + p.id)}${address} | lat: ${p.lat.toFixed(6)}, lon: ${p.lon.toFixed(6)} | demand: ${p.demand}${snap}`;

    L.marker([p.lat, p.lon], {
      icon,
      zIndexOffset: isDepot ? 1000 : 0,
    }).addTo(markers).bindPopup(content);
  }

  const customerEl = $('customers');
  if (customerEl) {
    customerEl.textContent = points.filter((p) => !p.depot).length;
  }
  fit();
}

export function clearLines() {
  if (lines) lines.clearLayers();
  routeLayers = [];
}

export function renderRouteLayers(routes) {
  clearLines();

  routes.forEach((r, i) => {
    const color = routeColors[i % routeColors.length];
    const layer = L.geoJSON(r.geometry, {
      style: {
        color,
        weight: 5,
        opacity: 0.85,
        dashArray: r.geometry_source === 'straight-line' ? '6 8' : null,
      },
    }).addTo(lines);

    const popup = document.createElement('div');
    popup.innerHTML = `<strong>Xe ${r.vehicle}</strong><br>Thứ tự: ${r.visits.join(' → ')}<br>Khởi hành: ${clock(r.departure_time)}`;
    layer.bindPopup(popup);

    routeLayers.push(layer);
  });

  return routeLayers;
}

export function focusRoute(index) {
  if (map && routeLayers[index]) {
    const layer = routeLayers[index];
    map.fitBounds(layer.getBounds(), { padding: [40, 40] });
    layer.openPopup();
  }
}
