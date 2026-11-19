/**
 * Authentication and User Permissions Module
 */

import { $, api, status } from './api.js';

let currentUser = null;

export function getCurrentUser() {
  return currentUser;
}

export function setCurrentUser(user) {
  currentUser = user;
}

export function updateAuthUI(user) {
  currentUser = user;
  $('login-screen').classList.add('hidden');
  $('app-main').classList.remove('hidden');
  $('logout').classList.remove('hidden');
  
  $('user-badge').textContent = `${user.username} · ${user.role.toUpperCase()}`;
  $('admin-controls').classList.toggle('hidden', user.role !== 'admin');
  $('driver-controls').classList.toggle('hidden', user.role !== 'driver');
  $('admin-note').classList.toggle('hidden', user.role !== 'admin');
  $('trip-list-title').textContent = user.role === 'admin' ? 'Các chuyến đã tạo' : 'Chuyến được giao';
  const isDriver = user.role === 'driver';
  $('duration-label').textContent = isDriver ? 'Tổng thời gian giao hàng (giây)' : 'Fitness (giây)';
  $('runtime-metric').classList.toggle('hidden', isDriver);
  $('seed-metric').classList.toggle('hidden', isDriver);
  document.querySelector('.metrics').classList.toggle('driver-metrics', isDriver);
  if (isDriver) $('comparison').classList.add('hidden');
}

export async function submitLogin(event, onLoggedIn) {
  if (event) {
    try {
      event.preventDefault();
      event.stopPropagation();
    } catch {}
  }
  const errorEl = $('login-error');
  if (errorEl) {
    errorEl.textContent = '';
    errorEl.classList.add('hidden');
  }
  
  const username = ($('login-username')?.value || '').trim();
  const password = $('login-password')?.value || '';

  if (!username || !password) {
    if (errorEl) {
      errorEl.textContent = 'Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu';
      errorEl.classList.remove('hidden');
    }
    return;
  }

  const submitBtn = $('login-submit');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Đang đăng nhập…';
  }
  
  try {
    const data = await api('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    setCurrentUser(data.user);
    updateAuthUI(data.user);
    if (onLoggedIn) await onLoggedIn(data.user);
  } catch (error) {
    if (errorEl) {
      errorEl.textContent = error.message;
      errorEl.classList.remove('hidden');
    }
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Đăng nhập';
    }
  }
}

export async function logout() {
  try {
    await api('/api/auth/logout', { method: 'POST' });
  } finally {
    try {
      localStorage.removeItem('rapo-map-job');
    } catch {}
    location.reload();
  }
}

export async function addDriver(onSuccess) {
  const username = $('new-driver-username').value.trim();
  const password = $('new-driver-password').value;
  
  try {
    await api('/api/drivers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    $('new-driver-username').value = '';
    $('new-driver-password').value = '';
    if (onSuccess) await onSuccess();
    status(`Đã tạo Driver: ${username}`, false);
  } catch (error) {
    status(error.message, true);
  }
}

export async function checkAuth(onLoggedIn) {
  try {
    const data = await api('/api/auth/me');
    setCurrentUser(data.user);
    updateAuthUI(data.user);
    if (onLoggedIn) await onLoggedIn(data.user);
  } catch (error) {
    if (error.status !== 401) {
      const errorEl = $('login-error');
      errorEl.textContent = error.message;
      errorEl.classList.remove('hidden');
    }
  }
}
