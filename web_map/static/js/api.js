/**
 * API client and utility formatting functions
 */

export const $ = (id) => document.getElementById(id);

export const fmt = (n) => Number(n).toLocaleString('vi-VN', { maximumFractionDigits: 2 });

export const algorithmLabel = (name) => ({
  'RAPO-2709': 'RAPO',
  'HCMC-TD-ERAPO': 'HCMC-TD-ERAPO',
}[name] || name);

export function status(text, error = false) {
  const el = $('status');
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('error', error);
  el.classList.toggle('success', !error && text.includes('Hoàn tất'));
}

export function clock(seconds) {
  const s = Math.round(seconds);
  const days = Math.floor(s / 86400);
  return (days ? `+${days}d ` : '') + [
    Math.floor((s % 86400) / 3600),
    Math.floor((s % 3600) / 60),
    s % 60,
  ].map((v) => String(v).padStart(2, '0')).join(':');
}

export function inputTime(seconds) {
  if (seconds == null) return '';
  const value = Math.round(Number(seconds));
  return `${String(Math.floor(value / 3600) % 24).padStart(2, '0')}:${String(Math.floor((value % 3600) / 60)).padStart(2, '0')}`;
}

export function timeSeconds(value) {
  if (!value) return null;
  const [hours, minutes] = value.split(':').map(Number);
  return hours * 3600 + minutes * 60;
}

export async function api(url, options = {}) {
  const { timeoutMs = 30000, ...fetchOptions } = options;
  const controller = new AbortController();
  const suppliedSignal = fetchOptions.signal;
  const timeoutId = suppliedSignal ? null : setTimeout(() => controller.abort(), timeoutMs);
  if (!suppliedSignal) fetchOptions.signal = controller.signal;

  let response;
  try {
    response = await fetch(url, fetchOptions);
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw Error('Backend phản hồi quá lâu. Hệ thống sẽ kiểm tra lại trạng thái điểm giao.');
    }
    throw Error('Không kết nối được backend. Hãy chạy lại web_map rồi tải lại trang.');
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
  let data;
  try {
    data = await response.json();
  } catch {
    if (response.status === 404) {
      throw Error(`Lỗi HTTP 404: Không tìm thấy dữ liệu yêu cầu (${url})`);
    } else if (response.status === 403) {
      throw Error(`Lỗi HTTP 403: Không có quyền truy cập (${url})`);
    } else if (response.status === 500) {
      throw Error(`Lỗi HTTP 500: Máy chủ gặp lỗi nội bộ (${url})`);
    }
    throw Error(`HTTP ${response.status}: backend không trả JSON (${url})`);
  }
  if (!response.ok) {
    const error = Error(data.error || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

export function download(data, type, name) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
