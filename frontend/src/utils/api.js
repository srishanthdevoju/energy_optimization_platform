const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export function apiFetch(path, options = {}) {
  // Ensure path starts with a slash if API_BASE is set
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return fetch(`${API_BASE}${cleanPath}`, options);
}
