import { useEffect, useMemo, useState } from "react"
import {
  FileTextIcon,
  Loader2Icon,
  MapPinIcon,
  PrinterIcon,
  TruckIcon,
  XIcon,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  algorithmLabel,
  api,
  clock,
  fmt,
} from "@/lib/api"
import type {
  MapPoint,
  OptimizationPayload,
  RouteResult,
  Trip,
  TripStop,
} from "@/types/rapo"

export interface ManifestStop {
  stopIndex: number
  customerId: number
  name: string
  address?: string
  lat?: number
  lon?: number
  demand: number
  arrivalTime?: number
  departureTime?: number
  serviceTime?: number
  status?: string
  completedAt?: string | null
}

export interface ManifestData {
  jobId: string
  tripName: string
  dataset: string
  algorithm: string
  vehicle: number
  routeIndex: number
  capacity: number
  load: number
  departureTime: number
  endTime: number
  duration: number
  travelTime?: number
  serviceTime?: number
  driverName?: string
  driverUsername?: string
  driverId?: number
  tripStatus?: string
  stops: ManifestStop[]
  depot?: {
    name?: string
    address?: string
    lat?: number
    lon?: number
  }
}

export interface ManifestTarget {
  jobId: string
  routeIndex: number
  vehicle: number
  algorithm: string
  tripName: string
  dataset?: string
  trip?: Trip | null
  routeResult?: RouteResult
  points?: MapPoint[]
  capacity?: number
}

interface DispatchManifestModalProps {
  open: boolean
  onClose: () => void
  target?: ManifestTarget | null
  initialManifest?: ManifestData | null
}

function formatDuration(seconds: number | undefined | null): string {
  if (seconds == null || isNaN(seconds)) return "0s"
  const total = Math.round(seconds)
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}p`
  }
  if (minutes > 0) {
    return `${minutes} phút ${secs > 0 ? `${secs}s` : ""}`
  }
  return `${secs}s`
}

function formatDate(date: Date = new Date()): string {
  const day = String(date.getDate()).padStart(2, "0")
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const year = date.getFullYear()
  return `${day}/${month}/${year}`
}

function formatTime(date: Date = new Date()): string {
  const hours = String(date.getHours()).padStart(2, "0")
  const minutes = String(date.getMinutes()).padStart(2, "0")
  const seconds = String(date.getSeconds()).padStart(2, "0")
  return `${hours}:${minutes}:${seconds}`
}

function BarcodeSvg({ value }: { value: string }) {
  const bars: { width: number; space: number }[] = []
  for (let i = 0; i < 28; i++) {
    const code = value.charCodeAt(i % value.length) || 65
    bars.push({
      width: (code % 3) + 1.2,
      space: ((code >> 2) % 2) + 1.1,
    })
  }
  let currentX = 6

  return (
    <div className="flex flex-col items-end">
      <svg className="h-8 w-44" viewBox="0 0 160 32">
        {bars.map((bar, i) => {
          const x = currentX
          currentX += bar.width + bar.space
          if (x > 152) return null
          return (
            <rect
              key={i}
              x={x}
              y={2}
              width={bar.width}
              height={26}
              fill="#18181b"
            />
          )
        })}
      </svg>
      <span className="font-mono text-[9px] font-semibold tracking-widest text-zinc-600 uppercase">
        *{value}*
      </span>
    </div>
  )
}

export function DispatchManifestModal({
  open,
  onClose,
  target,
  initialManifest,
}: DispatchManifestModalProps) {
  const [asyncLoading, setAsyncLoading] = useState<boolean>(false)
  const [asyncError, setAsyncError] = useState<string | null>(null)
  const [asyncManifest, setAsyncManifest] = useState<ManifestData | null>(null)
  const [showSignatures, setShowSignatures] = useState<boolean>(true)
  const [printDate] = useState<string>(formatDate())
  const [printTime] = useState<string>(formatTime())

  // Derive manifest synchronously when pre-passed data exists
  const syncManifest = useMemo<ManifestData | null>(() => {
    if (initialManifest) return initialManifest
    if (!target?.routeResult || !target?.points) return null

    const points = target.points
    const route = target.routeResult
    const pointMap = new Map<number, MapPoint>()
    let depotPoint: MapPoint | undefined
    for (const p of points) {
      pointMap.set(Number(p.id), p)
      if (p.depot) depotPoint = p
    }

    const customerStops: ManifestStop[] = []
    let stopOrder = 1
    for (const st of route.stops || []) {
      const cid = Number(st.customer_id)
      const pt = pointMap.get(cid)
      if (pt?.depot || cid === 0) continue
      customerStops.push({
        stopIndex: stopOrder++,
        customerId: cid,
        name: pt?.name || `Khách hàng #${cid}`,
        address:
          pt?.address || (pt ? `${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}` : "—"),
        lat: pt?.lat,
        lon: pt?.lon,
        demand: Number(st.demand ?? pt?.demand ?? 0),
        arrivalTime: st.arrival_time,
        departureTime: st.departure_time,
        serviceTime: st.service_time,
        status: "pending",
      })
    }

    const trip = target.trip
    const driverName = trip?.driver_full_name || trip?.driver_username || ""
    const capacityVal = target.capacity || 100

    return {
      jobId: target.jobId,
      tripName: target.tripName || `Chuyến ${target.dataset || ""}`,
      dataset: target.dataset || "—",
      algorithm: target.algorithm,
      vehicle: target.vehicle,
      routeIndex: target.routeIndex,
      capacity: capacityVal,
      load: route.load,
      departureTime: route.departure_time,
      endTime: route.end_time,
      duration: route.duration,
      travelTime: route.travel_time,
      serviceTime: route.service_time,
      driverName: driverName || undefined,
      driverUsername: trip?.driver_username || undefined,
      driverId: trip?.driver_id,
      tripStatus: trip?.status || "unassigned",
      stops: customerStops,
      depot: depotPoint
        ? {
            name: depotPoint.name || "Kho hàng trung tâm (Depot)",
            address:
              depotPoint.address ||
              `${depotPoint.lat.toFixed(5)}, ${depotPoint.lon.toFixed(5)}`,
            lat: depotPoint.lat,
            lon: depotPoint.lon,
          }
        : undefined,
    }
  }, [initialManifest, target])

  // Fetch asynchronously when pre-passed data is not available
  useEffect(() => {
    if (!open || syncManifest || !target) {
      return
    }

    let active = true

    async function fetchData() {
      setAsyncLoading(true)
      setAsyncError(null)
      try {
        let payload: OptimizationPayload | null = null
        let stopsInfo: TripStop[] = []
        let currentTrip: Trip | null = target?.trip || null

        if (target?.trip?.id) {
          const res = await api<{
            trip: Trip
            stops: TripStop[]
            payload: OptimizationPayload
          }>(`/api/trips/${target.trip.id}`)
          payload = res.payload
          stopsInfo = res.stops || []
          currentTrip = res.trip
        } else if (target?.jobId) {
          const res = await api<{
            payload?: OptimizationPayload
            stops?: TripStop[]
          }>(`/api/runs/${target.jobId}`)
          payload = res.payload || null
          stopsInfo = res.stops || []
        }

        if (!active) return

        if (!payload) {
          throw new Error("Không thể nạp dữ liệu lộ trình cho lượt chạy này")
        }

        const algResult =
          payload.results[target?.algorithm || ""] ||
          Object.values(payload.results)[0]
        if (!algResult) {
          throw new Error("Không tìm thấy kết quả tối ưu cho thuật toán này")
        }

        const route =
          algResult.routes.find((r) => r.vehicle === target?.vehicle) ||
          algResult.routes[target?.routeIndex || 0]

        if (!route) {
          throw new Error(`Không tìm thấy lộ trình của Xe ${target?.vehicle}`)
        }

        const pointMap = new Map<number, MapPoint>()
        let depotPoint: MapPoint | undefined
        for (const p of payload.points || []) {
          pointMap.set(Number(p.id), p)
          if (p.depot) depotPoint = p
        }

        const stopStatusMap = new Map<number, TripStop>()
        for (const s of stopsInfo) {
          stopStatusMap.set(Number(s.customer_id), s)
        }

        const customerStops: ManifestStop[] = []
        let stopOrder = 1
        for (const st of route.stops || []) {
          const cid = Number(st.customer_id)
          const pt = pointMap.get(cid)
          if (pt?.depot || cid === 0) continue

          const stopStatus = stopStatusMap.get(cid)
          customerStops.push({
            stopIndex: stopOrder++,
            customerId: cid,
            name: pt?.name || `Khách hàng #${cid}`,
            address:
              pt?.address ||
              (pt ? `${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}` : "—"),
            lat: pt?.lat,
            lon: pt?.lon,
            demand: Number(st.demand ?? pt?.demand ?? 0),
            arrivalTime: st.arrival_time,
            departureTime: st.departure_time,
            serviceTime: st.service_time,
            status: stopStatus?.status || "pending",
            completedAt: stopStatus?.completed_at,
          })
        }

        const capacityVal =
          payload.capacity ||
          payload.config?.custom_points?.vehicle_capacity ||
          100

        if (active) {
          setAsyncManifest({
            jobId: target?.jobId || "",
            tripName:
              payload.config?.trip_name ||
              target?.tripName ||
              `Chuyến ${payload.config?.dataset || ""}`,
            dataset: payload.config?.dataset || target?.dataset || "—",
            algorithm: target?.algorithm || "RAPO",
            vehicle: route.vehicle,
            routeIndex: target?.routeIndex ?? (route.vehicle - 1),
            capacity: capacityVal,
            load: route.load,
            departureTime: route.departure_time,
            endTime: route.end_time,
            duration: route.duration,
            travelTime: route.travel_time,
            serviceTime: route.service_time,
            driverName:
              currentTrip?.driver_full_name || currentTrip?.driver_username,
            driverUsername: currentTrip?.driver_username,
            driverId: currentTrip?.driver_id,
            tripStatus: currentTrip?.status || "unassigned",
            stops: customerStops,
            depot: depotPoint
              ? {
                  name: depotPoint.name || "Kho hàng trung tâm (Depot)",
                  address:
                    depotPoint.address ||
                    `${depotPoint.lat.toFixed(5)}, ${depotPoint.lon.toFixed(5)}`,
                  lat: depotPoint.lat,
                  lon: depotPoint.lon,
                }
              : undefined,
          })
        }
      } catch (err) {
        if (!active) return
        setAsyncError(
          err instanceof Error
            ? err.message
            : "Đã có lỗi khi nạp thông tin phiếu in"
        )
      } finally {
        if (active) setAsyncLoading(false)
      }
    }

    void fetchData()
    return () => {
      active = false
    }
  }, [open, syncManifest, target])

  const manifest = syncManifest || asyncManifest
  const loading = !syncManifest && asyncLoading
  const error = !syncManifest ? asyncError : null

  const handlePrint = () => {
    window.print()
  }

  const manifestCode = useMemo(() => {
    if (!manifest) return ""
    const shortJob = manifest.jobId.slice(0, 8).toUpperCase()
    return `LDP-${shortJob}-X0${manifest.vehicle}`
  }, [manifest])

  if (!open) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[100] flex flex-col bg-background/80 backdrop-blur-sm animate-in fade-in-0"
    >
      {/* Top action toolbar (Hidden when printing via .no-print) */}
      <div className="no-print flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-4 sm:px-6 shadow-xs">
        <div className="flex items-center gap-2.5">
          <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <FileTextIcon className="size-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm">
                Phiếu Lệnh Điều Phối & Bàn Giao
              </span>
              <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                Khổ A4 Chuẩn In
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground hidden sm:block">
              {manifest
                ? `${manifest.tripName} · Tuyến Xe ${manifest.vehicle} (${manifest.stops.length} điểm giao)`
                : "Đang nạp dữ liệu..."}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <label className="hidden md:flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none mr-2">
            <input
              type="checkbox"
              checked={showSignatures}
              onChange={(e) => setShowSignatures(e.target.checked)}
              className="rounded border-input text-primary focus:ring-primary"
            />
            Khung chữ ký 3 bên
          </label>

          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={loading || Boolean(error)}
            onClick={handlePrint}
            className="gap-1.5 font-medium shadow-xs"
          >
            <PrinterIcon className="size-4" />
            <span>In lệnh ngay (A4)</span>
          </Button>

          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onClose}
            className="size-8 p-0"
          >
            <XIcon className="size-4" />
            <span className="sr-only">Đóng</span>
          </Button>
        </div>
      </div>

      {/* Main preview scroll container */}
      <div className="flex-1 overflow-y-auto bg-zinc-200/70 p-3 sm:p-6 md:p-8 dark:bg-zinc-950/80">
        {loading ? (
          <div className="flex min-h-[400px] flex-col items-center justify-center gap-3">
            <Loader2Icon className="size-8 animate-spin text-primary" />
            <p className="text-sm font-medium text-muted-foreground">
              Đang tạo phiếu lệnh điều phối A4…
            </p>
          </div>
        ) : error ? (
          <div className="mx-auto max-w-md rounded-xl border border-destructive/30 bg-destructive/10 p-6 text-center">
            <p className="text-sm font-medium text-destructive">{error}</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onClose}
              className="mt-4"
            >
              Đóng
            </Button>
          </div>
        ) : manifest ? (
          <div
            id="dispatch-manifest-print-area"
            className="printable-manifest mx-auto w-full max-w-[820px] rounded-sm bg-white p-6 sm:p-9 text-zinc-900 shadow-2xl transition-all font-sans print:shadow-none print:p-0 print:m-0 print:max-w-none"
            style={{ minHeight: "1050px" }}
          >
            {/* Top Company & Header */}
            <div className="flex items-start justify-between border-b-2 border-zinc-900 pb-3">
              <div className="flex items-start gap-3">
                <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-zinc-900 text-white print:border print:border-black">
                  <TruckIcon className="size-6" />
                </div>
                <div>
                  <div className="font-extrabold uppercase tracking-wide text-xs text-zinc-900">
                    CÔNG TY VẬN TẢI & GIAO HÀNG TDVRP
                  </div>
                  <div className="text-[11px] font-medium text-zinc-600">
                    Trung tâm Điều phối Logistics Đa phương thức Thời gian thực
                  </div>
                  <div className="text-[10px] text-zinc-500">
                    Hotline: 1900 6868 · Email: dispatch@tdvrp-logistics.vn
                  </div>
                </div>
              </div>

              <div className="text-right">
                <BarcodeSvg value={manifestCode} />
                <div className="mt-1 text-[10px] text-zinc-500">
                  Lập phiếu: {printDate} · {printTime}
                </div>
              </div>
            </div>

            {/* Title */}
            <div className="my-4 text-center">
              <h1 className="text-xl sm:text-2xl font-black uppercase tracking-tight text-zinc-900">
                LỆNH ĐIỀU PHỐI & PHIẾU BÀN GIAO LỘ TRÌNH
              </h1>
              <p className="mt-0.5 text-xs font-semibold italic text-zinc-600">
                DELIVERY ROUTE DISPATCH & HANDOVER MANIFEST
              </p>
              <div className="mt-1 inline-flex items-center gap-1.5 rounded-full border border-zinc-300 bg-zinc-50 px-3 py-0.5 text-[11px] font-mono text-zinc-800">
                <span>MÃ PHIẾU: <strong>{manifestCode}</strong></span>
                <span>·</span>
                <span>MÃ JOB: <strong>{manifest.jobId.slice(0, 12)}</strong></span>
              </div>
            </div>

            {/* 4-Box Summary Grid */}
            <div className="mb-4 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
              <div className="rounded border border-zinc-300 bg-zinc-50/70 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                  1. Tài xế nhận lệnh
                </div>
                <div className="mt-1 font-bold text-zinc-900 truncate">
                  {manifest.driverName || "Chưa phân công"}
                </div>
                <div className="text-[11px] text-zinc-600">
                  {manifest.driverUsername
                    ? `@${manifest.driverUsername}`
                    : "—"}
                </div>
                <div className="mt-1 font-mono text-[10px] text-zinc-500">
                  Mã TX: {manifest.driverId ? `DRV-${manifest.driverId}` : "Chờ bàn giao"}
                </div>
              </div>

              <div className="rounded border border-zinc-300 bg-zinc-50/70 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                  2. Tuyến & Phương tiện
                </div>
                <div className="mt-1 font-bold text-zinc-900">
                  Tuyến: Xe {manifest.vehicle}
                </div>
                <div className="text-[11px] text-zinc-600 truncate">
                  {manifest.tripName}
                </div>
                <div className="mt-1 font-mono text-[10px] text-zinc-500">
                  Biển số: 59C1-{(8000 + manifest.vehicle)}
                </div>
              </div>

              <div className="rounded border border-zinc-300 bg-zinc-50/70 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                  3. Kế hoạch thời gian
                </div>
                <div className="mt-1 font-bold text-zinc-900">
                  Xuất kho: {clock(manifest.departureTime)}
                </div>
                <div className="text-[11px] text-zinc-600">
                  Dự kiến về: {clock(manifest.endTime)}
                </div>
                <div className="mt-1 font-mono text-[10px] text-zinc-500">
                  Tổng ca: {formatDuration(manifest.duration)}
                </div>
              </div>

              <div className="rounded border border-zinc-300 bg-zinc-50/70 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                  4. Tải trọng & Quy mô
                </div>
                <div className="mt-1 font-bold text-zinc-900">
                  Tải: {fmt(manifest.load)} / {manifest.capacity} kg
                </div>
                <div className="text-[11px] text-zinc-600">
                  {((manifest.load / manifest.capacity) * 100).toFixed(0)}% tải · {manifest.stops.length} điểm giao
                </div>
                <div className="mt-1 font-mono text-[10px] text-zinc-500">
                  Thuật toán: {algorithmLabel(manifest.algorithm)}
                </div>
              </div>
            </div>

            {/* Stop-by-stop Delivery Table */}
            <div className="mb-4 overflow-hidden rounded border border-zinc-300">
              <table className="w-full border-collapse text-left text-[11px]">
                <thead>
                  <tr className="border-b border-zinc-300 bg-zinc-100 font-bold text-zinc-800">
                    <th className="w-10 px-2 py-2 text-center">STT</th>
                    <th className="px-2.5 py-2">Khách hàng / Điểm dừng</th>
                    <th className="px-2.5 py-2">Địa chỉ giao hàng</th>
                    <th className="w-16 px-2 py-2 text-right">Khối lượng</th>
                    <th className="w-20 px-2 py-2 text-center">Giờ hẹn (ETA)</th>
                    <th className="w-14 px-2 py-2 text-center">Phục vụ</th>
                    <th className="w-36 px-2.5 py-2 text-center">
                      Ký nhận của khách
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200">
                  {/* Row 0: Depot departure */}
                  <tr className="bg-zinc-50/80 font-medium">
                    <td className="px-2 py-1.5 text-center font-mono font-bold text-zinc-700">
                      0
                    </td>
                    <td className="px-2.5 py-1.5">
                      <div className="flex items-center gap-1 font-bold text-zinc-900">
                        <MapPinIcon className="size-3 text-red-600 shrink-0" />
                        <span>KHO TRUNG TÂM (DEPOT)</span>
                      </div>
                      <div className="text-[10px] text-zinc-500">
                        Xuất phát ca giao hàng
                      </div>
                    </td>
                    <td className="px-2.5 py-1.5 text-[10px] text-zinc-600">
                      {manifest.depot?.address || "Kho hàng điều phối trung tâm"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono font-bold text-zinc-900">
                      {fmt(manifest.load)} kg
                    </td>
                    <td className="px-2 py-1.5 text-center font-mono font-bold text-emerald-700">
                      {clock(manifest.departureTime)}
                    </td>
                    <td className="px-2 py-1.5 text-center font-mono text-[10px] text-zinc-500">
                      Xuất bến
                    </td>
                    <td className="px-2.5 py-1 text-center">
                      <span className="text-[10px] italic text-zinc-500">
                        Thủ kho duyệt
                      </span>
                    </td>
                  </tr>

                  {/* Customer stops */}
                  {manifest.stops.map((stop) => (
                    <tr key={stop.stopIndex} className="hover:bg-zinc-50/50">
                      <td className="px-2 py-2 text-center font-mono font-bold text-zinc-800">
                        {stop.stopIndex}
                      </td>
                      <td className="px-2.5 py-2">
                        <div className="font-bold text-zinc-900">
                          {stop.name}
                        </div>
                        <div className="font-mono text-[9px] text-zinc-500">
                          Mã KH: #{stop.customerId}
                        </div>
                      </td>
                      <td className="px-2.5 py-2 text-[10px] text-zinc-700 leading-tight">
                        {stop.address}
                      </td>
                      <td className="px-2 py-2 text-right font-mono font-bold text-zinc-900 whitespace-nowrap">
                        {fmt(stop.demand)} kg
                      </td>
                      <td className="px-2 py-2 text-center font-mono text-zinc-900 whitespace-nowrap font-semibold">
                        {stop.arrivalTime ? clock(stop.arrivalTime) : "—"}
                      </td>
                      <td className="px-2 py-2 text-center font-mono text-[10px] text-zinc-600 whitespace-nowrap">
                        {stop.serviceTime ? `${fmt(stop.serviceTime)}s` : "—"}
                      </td>
                      <td className="px-2.5 py-1 text-center">
                        <div className="h-9 w-full rounded border border-dashed border-zinc-400 bg-zinc-50/40 flex flex-col justify-end items-center pb-0.5">
                          <span className="text-[8px] text-zinc-400 italic">
                            (Ký & ghi rõ họ tên)
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}

                  {/* Row end: Return to depot */}
                  <tr className="bg-zinc-50/80 font-medium">
                    <td className="px-2 py-1.5 text-center font-mono font-bold text-zinc-700">
                      ★
                    </td>
                    <td className="px-2.5 py-1.5">
                      <div className="flex items-center gap-1 font-bold text-zinc-900">
                        <MapPinIcon className="size-3 text-red-600 shrink-0" />
                        <span>KHO TRUNG TÂM (DEPOT) - KẾT THÚC</span>
                      </div>
                      <div className="text-[10px] text-zinc-500">
                        Xe hoàn thành hành trình về kho
                      </div>
                    </td>
                    <td className="px-2.5 py-1.5 text-[10px] text-zinc-600">
                      {manifest.depot?.address || "Kho hàng điều phối trung tâm"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono font-bold text-zinc-500">
                      0.0 kg
                    </td>
                    <td className="px-2 py-1.5 text-center font-mono font-bold text-blue-700">
                      {clock(manifest.endTime)}
                    </td>
                    <td className="px-2 py-1.5 text-center font-mono text-[10px] text-zinc-500">
                      Nhập bến
                    </td>
                    <td className="px-2.5 py-1 text-center">
                      <span className="text-[10px] italic text-zinc-500">
                        Thủ kho nhận lại xe
                      </span>
                    </td>
                  </tr>
                </tbody>

                {/* Table Summary Footer */}
                <tfoot>
                  <tr className="border-t-2 border-zinc-400 bg-zinc-100 font-bold text-zinc-900">
                    <td colSpan={3} className="px-3 py-1.5 text-right uppercase text-[10px] text-zinc-600">
                      Tổng cộng tuyến xe {manifest.vehicle}:
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono font-black text-xs text-zinc-900">
                      {fmt(manifest.load)} kg
                    </td>
                    <td colSpan={2} className="px-2 py-1.5 text-center text-[10px] text-zinc-700">
                      {manifest.stops.length} điểm · {formatDuration(manifest.duration)}
                    </td>
                    <td className="px-2.5 py-1.5 text-center text-[10px] text-zinc-500 italic">
                      Đối chiếu đủ kiện
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>

            {/* Logistics Instructions */}
            <div className="mb-4 rounded border border-zinc-300 bg-zinc-50/60 p-2.5 text-[10px] text-zinc-700 leading-relaxed">
              <div className="font-bold uppercase tracking-wider text-zinc-800 mb-1">
                Lưu ý vận hành & Quy định an toàn giao nhận:
              </div>
              <ul className="list-disc pl-4 space-y-0.5">
                <li>
                  Tài xế có nghĩa vụ tuân thủ thứ tự lộ trình tối ưu <strong>TDVRP (Time-Dependent)</strong> để đảm bảo không bị ùn tắc và giao đúng khung giờ hẹn (ETA).
                </li>
                <li>
                  Kiểm tra kỹ niêm phong thùng hàng, số lượng kiện và khối lượng hàng hoá trước khi rời kho xuất bến.
                </li>
                <li>
                  Yêu cầu khách hàng kiểm tra nguyên vẹn hàng hoá và ký nhận rõ họ tên vào cột "Ký nhận của khách".
                </li>
                <li>
                  Trường hợp phát sinh sự cố giao thông hoặc khách hàng vắng mặt, liên hệ ngay Tổng đài Điều phối: <strong>1900 6868</strong> để được hỗ trợ chuyển hướng.
                </li>
              </ul>
            </div>

            {/* 3 Signatures Section */}
            {showSignatures && (
              <div className="mt-6 pt-3 border-t border-zinc-300">
                <div className="grid grid-cols-3 gap-4 text-center text-xs">
                  <div>
                    <div className="font-bold uppercase tracking-wide text-zinc-800">
                      Thủ kho xuất hàng
                    </div>
                    <div className="text-[10px] italic text-zinc-500">
                      (Ký & ghi rõ họ tên)
                    </div>
                    <div className="h-16 flex items-end justify-center">
                      <span className="text-[10px] text-zinc-400">................................................</span>
                    </div>
                    <div className="text-[10px] text-zinc-500">
                      Ngày {printDate}
                    </div>
                  </div>

                  <div>
                    <div className="font-bold uppercase tracking-wide text-zinc-800">
                      Tài xế nhận lệnh & hàng
                    </div>
                    <div className="text-[10px] italic text-zinc-500">
                      (Ký & ghi rõ họ tên)
                    </div>
                    <div className="h-16 flex items-end justify-center">
                      <span className="text-[10px] text-zinc-400">................................................</span>
                    </div>
                    <div className="text-[10px] font-semibold text-zinc-800">
                      {manifest.driverName || "................................................"}
                    </div>
                  </div>

                  <div>
                    <div className="font-bold uppercase tracking-wide text-zinc-800">
                      Trưởng điều phối vận tải
                    </div>
                    <div className="text-[10px] italic text-zinc-500">
                      (Ký duyệt & đóng dấu)
                    </div>
                    <div className="h-16 flex items-end justify-center">
                      <span className="text-[10px] text-zinc-400">................................................</span>
                    </div>
                    <div className="text-[10px] text-zinc-500">
                      Ngày {printDate}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Page Footer */}
            <div className="mt-6 border-t border-zinc-200 pt-2 flex items-center justify-between text-[9px] text-zinc-400 font-mono">
              <span>Hệ thống Điều phối Lộ trình TDVRP · Phiếu in lưu hành nội bộ</span>
              <span>Trang 1 / 1 · In lúc {printTime}</span>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
