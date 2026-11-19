/**
 * Trip Management & Driver Assignment Module
 */

import { $, api, status, algorithmLabel, fmt, clock } from './api.js';
import { getCurrentUser } from './auth.js';

let currentTrip = null;
let currentJobId = null;
let allAssignedTrips = [];

export function getCurrentTrip() {
  return currentTrip;
}

export function setCurrentTrip(trip) {
  currentTrip = trip;
}

export function getCurrentJobId() {
  return currentJobId;
}

export function setCurrentJobId(jobId) {
  currentJobId = jobId;
}

export function getAssignedTrip(jobId, routeIndex) {
  if (!jobId) return null;
  return allAssignedTrips.find(
    (t) => t.job_id === jobId && Number(t.route_index) === Number(routeIndex)
  ) || null;
}

export function getAllAssignedTrips() {
  return allAssignedTrips;
}

export function statusLabel(value) {
  return {
    assigned: 'Đã giao',
    in_progress: 'Đang giao',
    completed: 'Hoàn thành',
    cancelled: 'Đã huỷ',
  }[value] || value;
}

export async function loadDrivers() {
  const select = $('driver');
  if (!select) return;
  const data = await api('/api/drivers');
  select.replaceChildren(...data.drivers.map((item) => new Option(item.username, item.id)));
  if (!data.drivers.length) {
    const runBtn = $('run');
    if (runBtn) runBtn.disabled = true;
    status('Hãy tạo ít nhất một tài khoản Driver trước', true);
  }
}

export async function viewAllJobRoutes(jobId, onDataLoaded, focusIndex = null) {
  try {
    const data = await api('/api/runs/' + jobId);
    if (!data.payload) {
      throw new Error('Không tìm thấy dữ liệu toàn bộ chuyến');
    }
    setCurrentJobId(jobId);
    setCurrentTrip(null);
    const primaryAlg = Object.keys(data.payload.results || {})[0];
    const totalVehicles = data.payload.results?.[primaryAlg]?.routes?.length || '';
    if (onDataLoaded) onDataLoaded(data.payload, data.stops || [], jobId);
    status(`Đang xem toàn bộ chuyến (${totalVehicles} xe) · Mã ${jobId.slice(0, 8)}`);
    if (focusIndex != null) {
      const { focusRoute } = await import('./map.js');
      setTimeout(() => focusRoute(focusIndex), 100);
    }
    if (window.reloadTripsApp) window.reloadTripsApp();
  } catch (error) {
    status(error.message, true);
  }
}

export async function loadTrips(onOpenTrip, onOpenFullJob) {
  const box = $('trip-list');
  if (!box) return;
  const currentUser = getCurrentUser();
  if (!currentUser) return;

  const data = await api('/api/trips');
  allAssignedTrips = data.trips || [];
  box.replaceChildren();

  // If driver and no trips
  if (currentUser.role !== 'admin' && !allAssignedTrips.length) {
    const empty = document.createElement('p');
    empty.className = 'muted';
    empty.textContent = 'Chưa có chuyến được giao cho tài khoản này.';
    box.appendChild(empty);
    return;
  }

  // Build groups
  let groups = [];
  if (currentUser.role === 'admin' && Array.isArray(data.jobs) && data.jobs.length) {
    groups = data.jobs.map((j) => ({
      jobId: j.job_id,
      plan: j.plan || {},
      trips: j.trips || [],
    }));
  } else {
    const grouped = new Map();
    for (const trip of allAssignedTrips) {
      if (!grouped.has(trip.job_id)) grouped.set(trip.job_id, []);
      grouped.get(trip.job_id).push(trip);
    }
    for (const [jobId, trips] of grouped) {
      groups.push({
        jobId,
        plan: trips[0]?.plan || {},
        trips,
      });
    }
  }

  if (!groups.length) {
    const empty = document.createElement('p');
    empty.className = 'muted';
    empty.textContent = currentUser.role === 'admin'
      ? 'Chưa có chuyến tối ưu nào được tạo.'
      : 'Chưa có chuyến được giao cho tài khoản này.';
    box.appendChild(empty);
    return;
  }

  for (const { jobId, plan, trips } of groups) {
    const group = document.createElement('details');
    const summary = document.createElement('summary');
    const body = document.createElement('div');
    const completed = trips.filter((t) => t.status === 'completed').length;
    const totalRoutes = plan.total_routes || trips.length || 0;

    group.className = 'trip-group';
    group.open = Boolean(
      (currentTrip && currentTrip.job_id === jobId) ||
      (currentJobId && currentJobId === jobId)
    );
    body.className = 'trip-group-body';

    summary.append(document.createTextNode(plan.trip_name || `Chuyến ${plan.instance || plan.dataset || jobId.slice(0, 8)}`));
    const meta = document.createElement('span');
    meta.className = 'trip-group-meta';
    if (totalRoutes > 0 && totalRoutes !== trips.length) {
      meta.textContent = `Mã ${jobId.slice(0, 8)} · ${trips.length}/${totalRoutes} xe đã giao · ${completed}/${trips.length} hoàn thành`;
    } else {
      meta.textContent = `Mã ${jobId.slice(0, 8)} · ${trips.length} xe · ${completed}/${trips.length} hoàn thành`;
    }
    summary.appendChild(meta);

    // Admin action: View full dispatch solution (all vehicles on the map)
    if (currentUser.role === 'admin') {
      const viewAllBtn = document.createElement('button');
      viewAllBtn.type = 'button';
      viewAllBtn.className = 'primary trip-view-all-btn';
      viewAllBtn.textContent = `🗺️ Xem toàn bộ chuyến (${totalRoutes || trips.length} xe trên bản đồ)`;
      viewAllBtn.onclick = (e) => {
        e.stopPropagation();
        viewAllJobRoutes(jobId, onOpenFullJob);
      };
      body.appendChild(viewAllBtn);
    }

    // Render route cards
    if (currentUser.role === 'admin' && Array.isArray(plan.routes_summary) && plan.routes_summary.length) {
      for (const r of plan.routes_summary) {
        const assignedTrip = trips.find((t) => Number(t.route_index) === Number(r.route_index));
        const card = document.createElement('div');
        const title = document.createElement('strong');
        const info = document.createElement('div');

        if (assignedTrip) {
          card.className = 'trip-card';
          title.textContent = `Xe ${r.vehicle} · ${algorithmLabel(assignedTrip.algorithm)} · ${statusLabel(assignedTrip.status)}`;
          info.className = 'muted';
          info.textContent = `Driver: ${assignedTrip.driver_username} · Đã giao: ${assignedTrip.stops_completed}/${assignedTrip.stops_total} điểm`;

          const openBtn = document.createElement('button');
          openBtn.className = 'secondary';
          openBtn.textContent = 'Mở riêng xe này';
          openBtn.onclick = () => onOpenTrip(assignedTrip);
          card.append(title, info, openBtn);
        } else {
          card.className = 'trip-card unassigned';
          title.textContent = `Xe ${r.vehicle} · ${algorithmLabel(plan.algorithm || 'RAPO')} · Chưa phân công`;
          info.className = 'muted';
          info.textContent = `${r.stops_count} điểm giao · Tải: ${fmt(r.load)} · ${clock(r.duration)}`;

          const viewBtn = document.createElement('button');
          viewBtn.className = 'secondary';
          viewBtn.textContent = 'Xem xe này';
          viewBtn.onclick = () => viewAllJobRoutes(jobId, onOpenFullJob, r.route_index);
          card.append(title, info, viewBtn);
        }

        body.appendChild(card);
      }
    } else {
      for (const trip of trips) {
        const card = document.createElement('div');
        const title = document.createElement('strong');
        const info = document.createElement('div');
        const openBtn = document.createElement('button');

        card.className = 'trip-card';
        title.textContent = `Xe ${trip.route_index + 1} · ${algorithmLabel(trip.algorithm)} · ${statusLabel(trip.status)}`;
        info.className = 'muted';
        info.textContent = `Driver: ${trip.driver_username} · Đã giao: ${trip.stops_completed}/${trip.stops_total} điểm`;
        
        openBtn.className = 'secondary';
        openBtn.textContent = 'Mở tuyến';
        openBtn.onclick = () => onOpenTrip(trip);
        card.append(title, info, openBtn);

        if (currentUser.role === 'driver' && trip.status === 'assigned') {
          const startBtn = document.createElement('button');
          startBtn.textContent = 'Bắt đầu chuyến';
          startBtn.onclick = async () => {
            await changeTrip(trip, 'in_progress', onOpenTrip);
          };
          card.appendChild(startBtn);
        }
        body.appendChild(card);
      }
    }

    group.append(summary, body);
    box.appendChild(group);
  }

  // Quick return button if admin is currently viewing a single trip
  if (currentUser.role === 'admin' && currentTrip && onOpenFullJob) {
    const backBtn = document.createElement('button');
    backBtn.type = 'button';
    backBtn.className = 'secondary trip-quick-back';
    backBtn.textContent = '⬅️ Quay lại xem toàn bộ các xe';
    backBtn.onclick = () => viewAllJobRoutes(currentTrip.job_id, onOpenFullJob);
    box.appendChild(backBtn);
  }
}

export async function assignRoute(algorithm, routeIndex, driverId, onAssigned) {
  if (!driverId) return status('Chọn Driver cho route', true);
  try {
    await api('/api/trips', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_id: currentJobId,
        algorithm,
        route_index: routeIndex,
        driver_id: driverId,
      }),
    });
    if (onAssigned) await onAssigned();
    status(`Đã giao route xe ${routeIndex + 1}`, false);
  } catch (error) {
    status(error.message, true);
  }
}

export async function openTrip(trip, onDataLoaded) {
  if (!trip || !trip.id) {
    status('Không tìm thấy thông tin chuyến hợp lệ để mở', true);
    return;
  }
  try {
    const data = await api('/api/trips/' + trip.id);
    currentJobId = trip.job_id;
    currentTrip = data.trip;
    if (onDataLoaded) onDataLoaded(data.payload, data.stops);
    status(`Đang xem route xe ${trip.route_index + 1} · ${trip.driver_username}`);
    // Refresh sidebar so quick return button displays
    if (window.reloadTripsApp) window.reloadTripsApp();
  } catch (error) {
    status(error.message, true);
  }
}

export async function changeTrip(trip, statusValue, onOpened) {
  if (!trip || !trip.id) {
    status('Không tìm thấy thông tin chuyến để cập nhật', true);
    return;
  }
  try {
    await api(`/api/trips/${trip.id}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: statusValue }),
    });
    const updatedTrip = { ...trip, status: statusValue };
    if (onOpened) {
      await onOpened(updatedTrip);
    }
    status('Đã bắt đầu chuyến', false);
  } catch (error) {
    status(error.message, true);
  }
}

export async function completeStop(stopIndex, onStopCompleted) {
  if (!currentTrip || !currentTrip.id) return false;
  try {
    const data = await api(`/api/trips/${currentTrip.id}/stops/${stopIndex}`, {
      method: 'PATCH',
      timeoutMs: 10000,
    });
    currentTrip = data.trip;
    if (onStopCompleted) onStopCompleted(data.stops);
    status(
      currentTrip.status === 'completed'
        ? 'Đã hoàn thành toàn bộ chuyến'
        : 'Đã hoàn thành điểm giao',
      false
    );
    if (window.reloadTripsApp) window.reloadTripsApp();
    return true;
  } catch (error) {
    try {
      const latest = await api(`/api/trips/${currentTrip.id}`, { timeoutMs: 5000 });
      const target = latest.stops.find((stop) => stop.stop_index === stopIndex);
      if (target?.status === 'completed') {
        currentTrip = latest.trip;
        if (onStopCompleted) onStopCompleted(latest.stops);
        status('Điểm giao đã được ghi nhận hoàn thành', false);
        return true;
      }
    } catch {}
    status(error.message, true);
    return false;
  }
}

export function renderStops(stops, payload, onCompleteClick) {
  const rows = $('stop-rows');
  const panel = $('stops-panel');
  if (!rows || !panel) return;

  rows.replaceChildren();
  panel.classList.remove('hidden');

  const currentUser = getCurrentUser();
  const firstPending = stops.find((s) => s.status === 'pending');
  const pointNames = new Map((payload?.points || []).map((p) => [Number(p.id), p.name]));

  for (const stop of stops) {
    const tr = document.createElement('tr');
    
    // Thứ tự
    const tdIdx = document.createElement('td');
    tdIdx.textContent = stop.stop_index + 1;
    tr.appendChild(tdIdx);

    // Tên điểm / khách hàng
    const tdCustomer = document.createElement('td');
    tdCustomer.textContent = pointNames.get(Number(stop.customer_id)) || `Khách hàng ${stop.customer_id}`;
    tr.appendChild(tdCustomer);

    // Trạng thái
    const tdStatus = document.createElement('td');
    tdStatus.textContent = stop.status === 'completed' ? 'Đã giao' : 'Chờ giao';
    tr.appendChild(tdStatus);

    // Hoàn thành lúc
    const tdTime = document.createElement('td');
    tdTime.textContent = stop.completed_at || '—';
    tr.appendChild(tdTime);

    // Thao tác
    const tdAction = document.createElement('td');
    if (
      currentUser &&
      currentUser.role === 'driver' &&
      currentTrip &&
      currentTrip.status === 'in_progress' &&
      firstPending &&
      stop.stop_index === firstPending.stop_index
    ) {
      const btn = document.createElement('button');
      btn.textContent = 'Hoàn thành điểm';
      btn.onclick = async () => {
        btn.disabled = true;
        btn.textContent = 'Đang cập nhật…';
        try {
          const completed = onCompleteClick ? await onCompleteClick(stop.stop_index) : false;
          if (!completed && btn.isConnected) {
            btn.disabled = false;
            btn.textContent = 'Thử lại';
          }
        } catch (err) {
          if (btn.isConnected) {
            btn.disabled = false;
            btn.textContent = 'Thử lại';
          }
        }
      };
      tdAction.appendChild(btn);
    }
    tr.appendChild(tdAction);

    rows.appendChild(tr);
  }
}
