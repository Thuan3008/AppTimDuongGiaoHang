import type { OptimizationPayload } from "@/types/rapo"

export const ROUTE_COLORS = [
  "#2563eb", // Royal Blue
  "#ea580c", // Bright Orange
  "#7c3aed", // Violet
  "#059669", // Emerald
  "#db2777", // Pink
  "#ca8a04", // Amber
  "#475569", // Slate
]

export class ApiError extends Error {
  status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

export function fmt(n: number | string | null | undefined): string {
  if (n == null || n === "") return "—"
  const value = Number(n)
  if (!Number.isFinite(value)) return "—"
  return value.toLocaleString("vi-VN", { maximumFractionDigits: 2 })
}

export function algorithmLabel(name: string): string {
  const map: Record<string, string> = {
    "RAPO-2709": "RAPO",
    "HCMC-TD-ERAPO": "HCMC-TD-ERAPO",
  }
  return map[name] || name
}

export function statusLabel(value: string): string {
  const map: Record<string, string> = {
    assigned: "Đã giao",
    in_progress: "Đang giao",
    completed: "Hoàn thành",
    cancelled: "Đã huỷ",
  }
  return map[value] || value
}

export function deliveryStatusLabel(value?: string): string {
  return value === "completed" ? "Đã giao" : "Chờ giao"
}

export function driverDisplayName(
  username: string,
  fullName?: string | null
): string {
  const cleaned = (fullName || "").trim()
  return cleaned ? `${cleaned} (@${username})` : username
}

export function clock(seconds: number): string {
  const s = Math.round(seconds)
  const days = Math.floor(s / 86400)
  return (
    (days ? `+${days}d ` : "") +
    [
      Math.floor((s % 86400) / 3600),
      Math.floor((s % 3600) / 60),
      s % 60,
    ]
      .map((v) => String(v).padStart(2, "0"))
      .join(":")
  )
}

export function inputTime(seconds: number | null | undefined): string {
  if (seconds == null) return ""
  const value = Math.round(Number(seconds))
  return `${String(Math.floor(value / 3600) % 24).padStart(2, "0")}:${String(
    Math.floor((value % 3600) / 60)
  ).padStart(2, "0")}`
}

export function timeSeconds(value: string): number | null {
  if (!value) return null
  const [hours, minutes] = value.split(":").map(Number)
  if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return null
  return hours * 3600 + minutes * 60
}

export interface ApiOptions extends RequestInit {
  timeoutMs?: number
}

export async function api<T = unknown>(
  url: string,
  options: ApiOptions = {}
): Promise<T> {
  const { timeoutMs = 30000, ...fetchOptions } = options
  const controller = new AbortController()
  const suppliedSignal = fetchOptions.signal
  const timeoutId = suppliedSignal
    ? null
    : window.setTimeout(() => controller.abort(), timeoutMs)

  if (!suppliedSignal) {
    fetchOptions.signal = controller.signal
  }

  let response: Response
  try {
    response = await fetch(url, {
      credentials: "same-origin",
      ...fetchOptions,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(
        "Backend phản hồi quá lâu. Hệ thống sẽ kiểm tra lại trạng thái điểm giao."
      )
    }
    throw new ApiError(
      "Không kết nối được backend. Hãy chạy lại web_map rồi tải lại trang."
    )
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId)
    }
  }

  let data: Record<string, unknown>
  try {
    data = (await response.json()) as Record<string, unknown>
  } catch {
    if (response.status === 404) {
      throw new ApiError(
        `Lỗi HTTP 404: Không tìm thấy dữ liệu yêu cầu (${url})`,
        404
      )
    }
    if (response.status === 403) {
      throw new ApiError(
        `Lỗi HTTP 403: Không có quyền truy cập (${url})`,
        403
      )
    }
    if (response.status === 500) {
      throw new ApiError(
        `Lỗi HTTP 500: Máy chủ gặp lỗi nội bộ (${url})`,
        500
      )
    }
    throw new ApiError(
      `HTTP ${response.status}: backend không trả JSON (${url})`,
      response.status
    )
  }

  if (!response.ok) {
    const message =
      typeof data.error === "string" && data.error
        ? data.error
        : `HTTP ${response.status}`
    throw new ApiError(message, response.status)
  }

  return data as T
}

export function downloadFile(data: string, type: string, name: string): void {
  const url = URL.createObjectURL(new Blob([data], { type }))
  const a = document.createElement("a")
  a.href = url
  a.download = name
  a.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function exportPayloadJSON(payload: OptimizationPayload): void {
  downloadFile(
    JSON.stringify(payload, null, 2),
    "application/json",
    `rapo_map_${payload.config.dataset}_seed${payload.config.seed}.json`
  )
}

export function exportAlgorithmCSV(
  payload: OptimizationPayload,
  algorithmName: string
): void {
  const result = payload.results[algorithmName]
  if (!result) return

  const rows = [
    [
      "vehicle",
      "visits",
      "departure_seconds",
      "travel_seconds",
      "service_seconds",
      "duration_seconds",
      "objective_seconds",
      "edge_path",
      "geometry_source",
    ],
    ...result.routes.map((r) => [
      r.vehicle,
      r.visits.join(" -> "),
      r.departure_time,
      r.travel_time,
      r.service_time,
      r.duration,
      r.objective,
      (r.edge_path || []).join(" -> "),
      r.geometry_source,
    ]),
  ]

  downloadFile(
    "\ufeff" +
      rows
        .map((row) =>
          row.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(",")
        )
        .join("\r\n"),
    "text/csv",
    `${algorithmLabel(algorithmName).toLowerCase().replaceAll("-", "_")}_${payload.config.dataset}_routes.csv`
  )
}
