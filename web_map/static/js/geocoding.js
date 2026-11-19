/**
 * Geocoding, Address Search & Custom Delivery Points Manager
 */

import { $, api, fmt, status, inputTime, timeSeconds } from './api.js';
import { setPendingMapSelection } from './map.js';

export function reindexDeliveryPoints() {
  [...document.querySelectorAll('.delivery-point')].forEach((card, index) => {
    const title = card.querySelector('.point-title');
    if (title) title.textContent = `Điểm giao ${index + 1}`;
  });
}

export function invalidateResolvedAddress(addressInput, latInput, lonInput, note) {
  if (addressInput.dataset.resolvedAddress && addressInput.value.trim() !== addressInput.dataset.resolvedAddress) {
    delete addressInput.dataset.resolvedAddress;
    delete addressInput.dataset.snapConfirmed;
    delete addressInput.dataset.snapCheckedLat;
    delete addressInput.dataset.snapCheckedLon;
    latInput.value = '';
    lonInput.value = '';
    note.textContent = 'Địa chỉ đã thay đổi, cần tìm lại.';
  }
}

function rememberSnapCheck(addressInput, candidate) {
  addressInput.dataset.snapConfirmed = candidate.snap_status === 'confirm' ? 'true' : 'false';
  addressInput.dataset.snapCheckedLat = String(candidate.lat);
  addressInput.dataset.snapCheckedLon = String(candidate.lon);
}

function snapCheckIsCurrent(addressInput, latInput, lonInput) {
  return addressInput.dataset.snapCheckedLat === String(latInput.valueAsNumber)
    && addressInput.dataset.snapCheckedLon === String(lonInput.valueAsNumber);
}

export function selectGeocodeCandidate(candidate, address, addressInput, latInput, lonInput, note, candidates) {
  if (candidate.snap_status === 'rejected') {
    status(`Điểm này cách đường gần nhất ${fmt(candidate.snap_distance_m)} m, vượt giới hạn 300 m. Hãy chọn lại vị trí.`, true);
    return false;
  }
  if (candidate.snap_status === 'confirm') {
    const accepted = window.confirm(
      `Điểm này cách đường gần nhất ${fmt(candidate.snap_distance_m)} m. Bạn có chắc muốn dùng vị trí này không?`
    );
    if (!accepted) return false;
  }
  latInput.value = candidate.lat;
  lonInput.value = candidate.lon;
  addressInput.dataset.resolvedAddress = address;
  rememberSnapCheck(addressInput, candidate);
  renderGeocodeCandidates(note, candidates, candidate, address, addressInput, latInput, lonInput, null);
  return true;
}

export function renderGeocodeCandidates(note, candidates, selected, address, addressInput, latInput, lonInput, warning) {
  note.replaceChildren();
  const message = document.createElement('div');
  if (warning) {
    message.className = 'geocode-warning';
    message.textContent = warning + ' Hãy chọn một kết quả bên dưới hoặc nhập tọa độ nâng cao.';
  } else if (selected) {
    message.textContent = `Đã chọn: ${selected.display_name} · ${Number(selected.lat).toFixed(6)}, ${Number(selected.lon).toFixed(6)} · cách đường ${fmt(selected.snap_distance_m)} m`;
  }
  note.appendChild(message);

  if (warning) {
    const mapButton = document.createElement('button');
    mapButton.type = 'button';
    mapButton.className = 'geocode-choice';
    mapButton.textContent = '📍 Chọn vị trí trực tiếp trên bản đồ';
    mapButton.onclick = () => {
      setPendingMapSelection({
        address,
        addressInput,
        latInput,
        lonInput,
        note,
        candidates,
      });
      status('Bấm chuột vào đúng vị trí giao hàng trên bản đồ');
    };
    note.appendChild(mapButton);
  }

  if (candidates && candidates.length) {
    const details = document.createElement('details');
    details.className = 'geocode-results';
    details.open = Boolean(warning);

    const summary = document.createElement('summary');
    summary.textContent = `${warning ? 'Chọn' : 'Xem'} kết quả gợi ý (${candidates.length})`;
    details.appendChild(summary);

    const houseLabels = {
      exact: 'khớp số nhà',
      missing: 'thiếu số nhà',
      wrong: 'sai số nhà',
      not_requested: 'không có số nhà',
    };
    const snapLabels = {
      auto: 'tự động chấp nhận',
      confirm: 'cần xác nhận',
      rejected: 'không thể dùng',
    };

    for (const candidate of candidates) {
      const choice = document.createElement('button');
      choice.type = 'button';
      choice.className = 'geocode-choice';
      choice.disabled = candidate.snap_status === 'rejected';
      choice.textContent = `${candidate.display_name} · ${candidate.provider} · ${
        houseLabels[candidate.house_match] || candidate.house_match
      } · cách đường ${fmt(candidate.snap_distance_m)} m (${snapLabels[candidate.snap_status] || candidate.snap_status}) · ${Number(candidate.lat).toFixed(6)}, ${Number(candidate.lon).toFixed(6)}`;
      choice.onclick = () => selectGeocodeCandidate(candidate, address, addressInput, latInput, lonInput, note, candidates);
      details.appendChild(choice);
    }
    note.appendChild(details);
  }
}

export async function checkPointSnap(lat, lon) {
  return api('/api/snap-point', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset: $('dataset').value, lat, lon }),
  });
}

export async function resolveAddress(addressInput, latInput, lonInput, note, button) {
  const address = addressInput.value.trim();
  if (!address) {
    if (Number.isFinite(latInput.valueAsNumber) && Number.isFinite(lonInput.valueAsNumber)) {
      if (snapCheckIsCurrent(addressInput, latInput, lonInput)) return;
      button.disabled = true;
      note.textContent = 'Đang kiểm tra khoảng cách tới đường…';
      try {
        const snap = await checkPointSnap(latInput.valueAsNumber, lonInput.valueAsNumber);
        const candidate = {
          ...snap,
          display_name: 'Tọa độ nhập thủ công',
          provider: 'Tọa độ',
          house_match: 'not_requested',
        };
        if (!selectGeocodeCandidate(candidate, '', addressInput, latInput, lonInput, note, [candidate])) {
          throw Error('Vị trí chưa được xác nhận');
        }
        return;
      } finally {
        button.disabled = false;
      }
    }
    throw Error('Nhập địa chỉ hoặc tọa độ nâng cao');
  }

  if (addressInput.dataset.resolvedAddress === address
      && Number.isFinite(latInput.valueAsNumber)
      && Number.isFinite(lonInput.valueAsNumber)
      && snapCheckIsCurrent(addressInput, latInput, lonInput)) {
    return;
  }

  button.disabled = true;
  note.textContent = 'Đang tìm địa chỉ…';

  try {
    const data = await api('/api/geocode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: $('dataset').value, address }),
    });

    if (data.requires_confirmation) {
      delete addressInput.dataset.resolvedAddress;
      latInput.value = '';
      lonInput.value = '';
      renderGeocodeCandidates(note, data.candidates, null, address, addressInput, latInput, lonInput, data.warning);
      throw Error(data.warning + ' Chưa tự động chọn tọa độ.');
    }

    if (!selectGeocodeCandidate(data.result, address, addressInput, latInput, lonInput, note, data.candidates)) {
      throw Error('Vị trí chưa được xác nhận');
    }
  } finally {
    button.disabled = false;
  }
}

export async function handleMapClickPick(target, candidate) {
  try {
    status('Đang kiểm tra vị trí đã chọn với road graph…');
    const snap = await checkPointSnap(candidate.lat, candidate.lon);
    const checkedCandidate = { ...candidate, ...snap };
    const accepted = selectGeocodeCandidate(
      checkedCandidate,
      target.address,
      target.addressInput,
      target.latInput,
      target.lonInput,
      target.note,
      target.candidates
    );
    if (accepted) status(`Đã chọn vị trí · cách đường gần nhất ${fmt(snap.snap_distance_m)} m`);
    return accepted;
  } catch (error) {
    status(error.message, true);
    return false;
  }
}

export async function resolveAllAddresses() {
  await resolveAddress(
    $('depot-address'),
    $('depot-lat'),
    $('depot-lon'),
    $('depot-geocode-note'),
    $('find-depot')
  );

  for (const card of document.querySelectorAll('.delivery-point')) {
    await resolveAddress(
      card.querySelector('.point-address'),
      card.querySelector('.point-lat'),
      card.querySelector('.point-lon'),
      card.querySelector('.geocode-note'),
      card.querySelector('.find-address')
    );
  }
}

export function addDeliveryPoint(value = {}) {
  const card = document.createElement('div');
  card.className = 'delivery-point';
  card.innerHTML = `
    <div class="delivery-point-head">
      <strong class="point-title"></strong>
      <button class="remove-point secondary" type="button" title="Xóa điểm">Xóa</button>
    </div>
    <label>Tên điểm / người nhận</label>
    <input class="point-name" type="text" maxlength="100" placeholder="Ví dụ: Anh Nam">
    <label>Địa chỉ giao hàng</label>
    <div class="address-line">
      <input class="point-address" type="text" maxlength="300" placeholder="Số nhà, đường, phường/quận, TP.HCM">
      <button class="find-address secondary" type="button">Tìm</button>
    </div>
    <div class="geocode-note">Nhập địa chỉ rồi bấm Tìm hoặc Xem điểm.</div>
    <details class="coord-advanced">
      <summary>Tọa độ nâng cao (tùy chọn)</summary>
      <div class="manual-grid">
        <div><label>Vĩ độ</label><input class="point-lat" type="number" step="any" placeholder="Tự điền"></div>
        <div><label>Kinh độ</label><input class="point-lon" type="number" step="any" placeholder="Tự điền"></div>
      </div>
    </details>
    <div class="manual-grid">
      <div><label>Khối lượng (kg)</label><input class="point-demand" type="number" min="0.01" step="any" value="1"></div>
      <div><label>Phục vụ (phút)</label><input class="point-service" type="number" min="0" step="0.5" value="5"></div>
      <div><label>Từ giờ (tùy chọn)</label><input class="point-window-start" type="time"></div>
      <div><label>Đến giờ (tùy chọn)</label><input class="point-window-end" type="time"></div>
    </div>
  `;

  card.querySelector('.point-name').value = value.name || '';
  card.querySelector('.point-address').value = value.address || '';
  card.querySelector('.point-lat').value = value.lat ?? '';
  card.querySelector('.point-lon').value = value.lon ?? '';
  card.querySelector('.point-demand').value = value.demand ?? 1;
  card.querySelector('.point-service').value = value.service_time == null ? 5 : Number(value.service_time) / 60;

  if (value.time_window) {
    card.querySelector('.point-window-start').value = inputTime(value.time_window[0]);
    card.querySelector('.point-window-end').value = inputTime(value.time_window[1]);
  }

  if (value.address && value.lat != null && value.lon != null) {
    card.querySelector('.point-address').dataset.resolvedAddress = value.address;
  }
  if (value.lat != null && value.lon != null && value.snap_status) {
    rememberSnapCheck(card.querySelector('.point-address'), value);
    if (value.snap_confirmed) card.querySelector('.point-address').dataset.snapConfirmed = 'true';
  }

  card.querySelector('.remove-point').onclick = () => {
    card.remove();
    reindexDeliveryPoints();
  };

  card.querySelector('.find-address').onclick = () =>
    resolveAddress(
      card.querySelector('.point-address'),
      card.querySelector('.point-lat'),
      card.querySelector('.point-lon'),
      card.querySelector('.geocode-note'),
      card.querySelector('.find-address')
    ).catch((err) => status(err.message, true));

  card.querySelector('.point-address').oninput = () =>
    invalidateResolvedAddress(
      card.querySelector('.point-address'),
      card.querySelector('.point-lat'),
      card.querySelector('.point-lon'),
      card.querySelector('.geocode-note')
    );

  for (const input of [card.querySelector('.point-lat'), card.querySelector('.point-lon')]) {
    input.oninput = () => {
      const addressInput = card.querySelector('.point-address');
      delete addressInput.dataset.snapConfirmed;
      delete addressInput.dataset.snapCheckedLat;
      delete addressInput.dataset.snapCheckedLon;
    };
  }

  $('delivery-points').appendChild(card);
  reindexDeliveryPoints();
}

function requiredNumber(input, label) {
  const value = input.valueAsNumber;
  if (!Number.isFinite(value)) throw Error(`Cần nhập ${label}`);
  return value;
}

export function collectCustomPoints() {
  const customers = [...document.querySelectorAll('.delivery-point')].map((card, index) => {
    const start = card.querySelector('.point-window-start').value;
    const end = card.querySelector('.point-window-end').value;
    if (Boolean(start) !== Boolean(end)) {
      throw Error(`Điểm giao ${index + 1} phải nhập đủ hai đầu khung giờ`);
    }
    const timeWindow = start ? [timeSeconds(start), timeSeconds(end)] : null;
    if (timeWindow && timeWindow[0] >= timeWindow[1]) {
      throw Error(`Khung giờ điểm giao ${index + 1} không hợp lệ`);
    }

    return {
      name: card.querySelector('.point-name').value.trim() || `Điểm giao ${index + 1}`,
      address: card.querySelector('.point-address').value.trim(),
      lat: requiredNumber(card.querySelector('.point-lat'), `vĩ độ điểm giao ${index + 1}`),
      lon: requiredNumber(card.querySelector('.point-lon'), `kinh độ điểm giao ${index + 1}`),
      demand: requiredNumber(card.querySelector('.point-demand'), `nhu cầu điểm giao ${index + 1}`),
      service_time: requiredNumber(card.querySelector('.point-service'), `thời gian phục vụ điểm giao ${index + 1}`) * 60,
      time_window: timeWindow,
      snap_confirmed: card.querySelector('.point-address').dataset.snapConfirmed === 'true',
    };
  });

  if (!customers.length) throw Error('Cần thêm ít nhất một điểm giao');

  return {
    depot: {
      name: $('depot-name').value.trim() || 'Kho',
      address: $('depot-address').value.trim(),
      lat: requiredNumber($('depot-lat'), 'vĩ độ kho'),
      lon: requiredNumber($('depot-lon'), 'kinh độ kho'),
      snap_confirmed: $('depot-address').dataset.snapConfirmed === 'true',
    },
    num_vehicles: Number($('manual-vehicles').value),
    vehicle_capacity: Number($('manual-capacity').value),
    customers,
  };
}

export async function setManualMode(enabled, catalog, { refresh = true, previewCallback } = {}) {
  $('manual-fields').classList.toggle('hidden', !enabled);

  for (const option of $('dataset').options) {
    const entry = catalog.find((item) => item.id === option.value);
    option.disabled = Boolean(entry?.missing.length) || (enabled && !option.value.startsWith('hcmc'));
  }

  if (enabled && !$('dataset').value.startsWith('hcmc')) {
    const hcmc = catalog.find((item) => item.id.startsWith('hcmc') && !item.missing.length);
    if (hcmc) $('dataset').value = hcmc.id;
  }

  if (enabled && !document.querySelector('.delivery-point')) {
    addDeliveryPoint();
  }

  if (refresh && previewCallback) {
    await previewCallback();
  }
}

export async function previewCustomPoints(onPreviewLoaded) {
  try {
    status('Đang tìm địa chỉ…');
    await resolveAllAddresses();
    const custom = collectCustomPoints();

    const data = await api('/api/custom-points/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: $('dataset').value,
        trip_name: $('trip-name').value.trim(),
        custom_points: custom,
      }),
    });

    if (onPreviewLoaded) onPreviewLoaded(data.points);
    status(`Đã kiểm tra ${data.customers} điểm giao · ${data.vehicles} xe · tải trọng ${data.capacity}`);
  } catch (error) {
    status(error.message, true);
  }
}
