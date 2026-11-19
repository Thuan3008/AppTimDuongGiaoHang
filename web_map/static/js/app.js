/**
 * Main Application Entry Point & Module Connector
 */

import { $, api, status } from './api.js';
import { checkAuth, submitLogin, logout, addDriver } from './auth.js';
import { initializeMap, fit, renderPoints, getPoints } from './map.js';
import {
  loadTrips,
  loadDrivers,
  openTrip,
  completeStop,
  renderStops,
  getCurrentJobId,
  setCurrentJobId,
  getCurrentTrip,
  setCurrentTrip,
} from './trips.js';
import {
  addDeliveryPoint,
  previewCustomPoints,
  resolveAddress,
  invalidateResolvedAddress,
  handleMapClickPick,
  setManualMode,
} from './geocoding.js';
import {
  setCatalog,
  getCatalog,
  preview,
  startOptimization,
  poll,
  renderResult,
  updateDetailedItineraryStatuses,
  showAlgorithm,
  applyConfig,
  exportJSON,
  exportCSV,
  isBusy,
  getStartedAt,
  setBusy,
} from './optimizer.js';

function setTheme(dark) {
  document.documentElement.classList.toggle('dark', dark);
  const btn = $('theme-toggle');
  if (btn) btn.textContent = dark ? '☀️ Light' : '🌙 Dark';
  try {
    localStorage.setItem('rapo-map-theme', dark ? 'dark' : 'light');
  } catch {}
}

// --------------------------------------------------------------------------
// Responsive Layout & Dynamic DOM Relocation
// --------------------------------------------------------------------------
function adaptLayoutForScreen() {
  const isMobile = window.innerWidth < 992;
  const resultsContainer = $('results-container');
  const desktopWorkspace = document.querySelector('section.workspace');
  const mobileSheet = $('panel-results-mobile');

  if (isMobile && resultsContainer && mobileSheet && resultsContainer.parentElement !== mobileSheet) {
    mobileSheet.appendChild(resultsContainer);
  } else if (!isMobile && resultsContainer && desktopWorkspace && resultsContainer.parentElement !== desktopWorkspace) {
    desktopWorkspace.appendChild(resultsContainer);
  }
}

export function switchMobileTab(tab) {
  const tabDispatch = $('tab-dispatch');
  const tabResults = $('tab-results');
  const panelDispatch = $('panel-dispatch');
  const panelResults = $('panel-results-mobile');

  if (!tabDispatch || !tabResults || !panelDispatch || !panelResults) return;

  if (tab === 'results') {
    tabResults.classList.add('active');
    tabDispatch.classList.remove('active');
    panelResults.classList.add('active');
    panelDispatch.classList.remove('active');
  } else {
    tabDispatch.classList.add('active');
    tabResults.classList.remove('active');
    panelDispatch.classList.add('active');
    panelResults.classList.remove('active');
  }
}

function setupMobileTabs() {
  $('tab-dispatch')?.addEventListener('click', () => switchMobileTab('dispatch'));
  $('tab-results')?.addEventListener('click', () => switchMobileTab('results'));
}

// --------------------------------------------------------------------------
// Mobile Bottom Sheet Gesture Controller
// --------------------------------------------------------------------------
let applySheetState = null;

function setupBottomSheet() {
  const sidebar = document.querySelector('aside.sidebar');
  const handleWrap = document.querySelector('.bottom-sheet-handle-wrap');
  if (!sidebar || !handleWrap) return;

  const states = ['sheet-collapsed', 'sheet-half', 'sheet-expanded'];
  let currentStateIndex = 0; // 0: collapsed, 1: half, 2: expanded

  applySheetState = function (index) {
    currentStateIndex = Math.max(0, Math.min(2, index));
    states.forEach((cls) => sidebar.classList.remove(cls));
    sidebar.classList.add(states[currentStateIndex]);
  };

  // Khởi tạo mặc định: collapsed
  applySheetState(0);

  // Click vào handle để chuyển vòng tròn: Collapsed -> Half -> Full -> Collapsed
  handleWrap.addEventListener('click', () => {
    applySheetState((currentStateIndex + 1) % 3);
  });

  // Cử chỉ vuốt chạm (Touch swipe)
  let startY = 0;
  let currentY = 0;
  let isDragging = false;

  handleWrap.addEventListener('touchstart', (e) => {
    startY = e.touches[0].clientY;
    currentY = startY;
    isDragging = true;
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    currentY = e.touches[0].clientY;
  }, { passive: true });

  window.addEventListener('touchend', () => {
    if (!isDragging) return;
    isDragging = false;
    const deltaY = currentY - startY;
    if (deltaY < -40) {
      // Vuốt lên -> mở rộng hơn
      applySheetState(currentStateIndex + 1);
    } else if (deltaY > 40) {
      // Vuốt xuống -> thu gọn hơn
      applySheetState(currentStateIndex - 1);
    }
  });
}

// --------------------------------------------------------------------------
// Application Boot
// --------------------------------------------------------------------------
async function boot(user) {
  initializeMap(handleMapClickPick);
  adaptLayoutForScreen();

  const handleOpenTrip = (trip) => openTrip(trip, (payload, stops) => {
    renderResult(payload, stops);
    const updateStops = (currentStops) => {
      renderStops(currentStops, payload, (stopIdx) => completeStop(stopIdx, (newStops) => {
        updateStops(newStops);
        updateDetailedItineraryStatuses(newStops);
      }));
    };
    updateStops(stops);
    if (window.innerWidth < 992) {
      switchMobileTab('results');
      if (applySheetState) applySheetState(1); // Mở nấc Half screen để xem đơn
    }
  });

  const handleOpenFullJob = (payload, stops = [], jobId = null) => {
    if (jobId) setCurrentJobId(jobId);
    setCurrentTrip(null);
    renderResult(payload, stops);
    $('stops-panel')?.classList.add('hidden');
    if (window.innerWidth < 992) {
      switchMobileTab('results');
      if (applySheetState) applySheetState(1);
    }
  };

  const reloadTrips = async () => {
    await loadTrips(handleOpenTrip, handleOpenFullJob);
    const activeJobId = getCurrentJobId();
    const activeTrip = getCurrentTrip();
    if (activeJobId && !activeTrip) {
      try {
        const data = await api(`/api/runs/${activeJobId}/stops`);
        if (data && Array.isArray(data.stops)) {
          updateDetailedItineraryStatuses(data.stops);
        }
      } catch {}
    }
  };
  window.reloadTripsApp = reloadTrips;

  await reloadTrips();

  if (user.role === 'driver') {
    status('Chọn một chuyến được giao để xem lộ trình');
    return;
  }

  const catalogData = (await api('/api/datasets')).datasets;
  setCatalog(catalogData);

  const datasetSelect = $('dataset');
  datasetSelect.replaceChildren();

  for (const ds of catalogData) {
    const opt = new Option(ds.name + (ds.missing.length ? ' · thiếu sidecar' : ''), ds.id);
    opt.disabled = ds.missing.length > 0;
    datasetSelect.appendChild(opt);
  }

  const ready = catalogData.find((ds) => !ds.missing.length);
  if (!ready) throw Error('Chưa có dataset đủ sidecar.');

  datasetSelect.value = ready.id;
  $('trip-name').value = `Chuyến giao ${new Date().toLocaleDateString('vi-VN')}`;

  await loadDrivers();
  await preview();

  let saved;
  try {
    saved = localStorage.getItem('rapo-map-job');
  } catch {}

  if (saved) {
    try {
      const job = await api('/api/runs/' + saved);
      if (job.config) await applyConfig(job.config);
      if (job.payload) {
        setCurrentJobId(saved);
        renderResult(job.payload);
        status('Đã nạp kết quả gần nhất · có thể giao từng route');
        return;
      }
      if (job.status === 'queued' || job.status === 'running') {
        setBusy(true);
        await poll(saved);
        return;
      }
    } catch {
      try {
        localStorage.removeItem('rapo-map-job');
      } catch {}
    }
  }

  status('Sẵn sàng chạy tối ưu');
}

// --------------------------------------------------------------------------
// Event Listeners Registration
// --------------------------------------------------------------------------
function setupEventListeners() {
  window.addEventListener('resize', adaptLayoutForScreen);

  const loginForm = $('login-form');
  if (loginForm) loginForm.onsubmit = (e) => submitLogin(e, boot);
  const loginSubmit = $('login-submit');
  if (loginSubmit) loginSubmit.onclick = (e) => submitLogin(e, boot);
  $('logout').onclick = logout;
  $('theme-toggle').onclick = () => setTheme(!document.documentElement.classList.contains('dark'));

  $('run').onclick = () => startOptimization((jobId) => {
    if (window.innerWidth < 992) {
      switchMobileTab('results');
      if (applySheetState) applySheetState(1);
    }
  });

  $('add-driver').onclick = () => addDriver(loadDrivers);
  $('add-point').onclick = () => addDeliveryPoint();
  $('preview-points').onclick = () => previewCustomPoints((pts) => renderPoints(pts));

  $('find-depot').onclick = () =>
    resolveAddress(
      $('depot-address'),
      $('depot-lat'),
      $('depot-lon'),
      $('depot-geocode-note'),
      $('find-depot')
    ).catch((err) => status(err.message, true));

  $('depot-address').oninput = () =>
    invalidateResolvedAddress(
      $('depot-address'),
      $('depot-lat'),
      $('depot-lon'),
      $('depot-geocode-note')
    );

  for (const input of [$('depot-lat'), $('depot-lon')]) {
    input.oninput = () => {
      delete $('depot-address').dataset.snapConfirmed;
      delete $('depot-address').dataset.snapCheckedLat;
      delete $('depot-address').dataset.snapCheckedLon;
    };
  }

  $('manual-toggle').onchange = () =>
    setManualMode($('manual-toggle').checked, getCatalog(), {
      refresh: true,
      previewCallback: preview,
    }).then(() => {
      const pts = getPoints();
      if ($('manual-toggle').checked && pts.length) {
        $('depot-lat').value ||= pts[0].lat;
        $('depot-lon').value ||= pts[0].lon;
      }
    }).catch((err) => status(err.message, true));

  $('fit').onclick = fit;
  $('btn-fit-map')?.addEventListener('click', fit);
  $('result-algorithm').onchange = showAlgorithm;
  $('dataset').onchange = () =>
    preview().then(() => {
      const pts = getPoints();
      if ($('manual-toggle').checked && pts.length) {
        $('depot-lat').value ||= pts[0].lat;
        $('depot-lon').value ||= pts[0].lon;
      }
    }).catch((err) => status(err.message, true));

  $('json').onclick = exportJSON;
  $('csv').onclick = exportCSV;

  // Đồng hồ đo thời gian chạy
  setInterval(() => {
    const clockEl = $('clock');
    if (!clockEl) return;
    if (isBusy() && getStartedAt()) {
      clockEl.textContent = `Đang chạy · ${Math.floor((Date.now() - getStartedAt()) / 1000)}s`;
    } else {
      clockEl.textContent = 'RAPO / MAP';
    }
  }, 1000);

  // Tự động đồng bộ trạng thái các chuyến và tiến độ giao hàng định kỳ
  setInterval(() => {
    if (window.reloadTripsApp && !isBusy()) {
      window.reloadTripsApp();
    }
  }, 10000);
}

// --------------------------------------------------------------------------
// Main Initialization
// --------------------------------------------------------------------------
async function init() {
  let savedTheme = 'light';
  try {
    savedTheme = localStorage.getItem('rapo-map-theme') || 'light';
  } catch {}
  setTheme(savedTheme === 'dark');

  setupBottomSheet();
  setupMobileTabs();
  setupEventListeners();

  await checkAuth(boot);
}

window.handleLoginSubmit = (e) => submitLogin(e, boot);

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
