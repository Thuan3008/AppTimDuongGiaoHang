import { useMemo, useState } from "react"
import {
  CheckCircle2Icon,
  FileJsonIcon,
  FileSpreadsheetIcon,
  Loader2Icon,
  Maximize2Icon,
  PrinterIcon,
  SendIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { RouteGanttChart } from "@/components/RouteGanttChart"
import { VehicleCapacityBalance } from "@/components/VehicleCapacityBalance"
import {
  DispatchManifestModal,
  type ManifestTarget,
} from "@/components/DispatchManifestModal"
import {
  ROUTE_COLORS,
  algorithmLabel,
  clock,
  deliveryStatusLabel,
  driverDisplayName,
  fmt,
  statusLabel,
} from "@/lib/api"
import type {
  Driver,
  MapPoint,
  OptimizationPayload,
  Trip,
  TripStop,
  User,
} from "@/types/rapo"

interface ResultsWorkspaceProps {
  user: User
  points: MapPoint[]
  payload: OptimizationPayload | null
  capacity?: number
  selectedAlgorithm: string
  drivers: Driver[]
  allTrips: Trip[]
  currentJobId: string | null
  currentTrip: Trip | null
  deliveryStatuses: TripStop[]
  showStopsPanel: boolean
  onFocusRoute: (index: number) => void
  onFitMap: () => void
  onExportJSON: () => void
  onExportCSV: () => void
  onAssignRoute: (
    algorithm: string,
    routeIndex: number,
    driverId: number
  ) => Promise<void>
  onCompleteStop: (stopIndex: number) => Promise<boolean>
}

export function ResultsWorkspace({
  user,
  points,
  payload,
  capacity = 100,
  selectedAlgorithm,
  drivers,
  allTrips,
  currentJobId,
  currentTrip,
  deliveryStatuses,
  showStopsPanel,
  onFocusRoute,
  onFitMap,
  onExportJSON,
  onExportCSV,
  onAssignRoute,
  onCompleteStop,
}: ResultsWorkspaceProps) {
  const [routeDriverMap, setRouteDriverMap] = useState<Record<number, number>>(
    {}
  )
  const [assigningIndex, setAssigningIndex] = useState<number | null>(null)
  const [completingStop, setCompletingStop] = useState<number | null>(null)
  const [failedStop, setFailedStop] = useState<number | null>(null)
  const [manifestTarget, setManifestTarget] = useState<ManifestTarget | null>(
    null
  )

  const effectiveCapacity = useMemo(() => {
    return (
      payload?.capacity ||
      payload?.config?.custom_points?.vehicle_capacity ||
      capacity ||
      100
    )
  }, [payload, capacity])

  const isDriver = user.role === "driver"
  const algorithmNames = useMemo(
    () => (payload ? Object.keys(payload.results || {}) : []),
    [payload]
  )
  const activeResult =
    payload && selectedAlgorithm ? payload.results[selectedAlgorithm] : null
  const routes = useMemo(() => activeResult?.routes ?? [], [activeResult])

  const customerCount = useMemo(
    () => points.filter((p) => !p.depot).length,
    [points]
  )

  const deliveryDuration = useMemo(
    () => routes.reduce((sum, route) => sum + Number(route.duration || 0), 0),
    [routes]
  )

  const getAssignedTrip = (routeIndex: number): Trip | null => {
    if (!currentJobId) return null
    return (
      allTrips.find(
        (t) =>
          t.job_id === currentJobId &&
          Number(t.route_index) === Number(routeIndex)
      ) ?? null
    )
  }

  // Build point map and status lookup for detailed itinerary
  const pointMap = useMemo(() => {
    const source = payload?.points ?? points
    return new Map(source.map((p) => [Number(p.id), p]))
  }, [payload, points])

  const depotId = useMemo(() => {
    const source = payload?.points ?? points
    return Number(source.find((p) => p.depot)?.id ?? 0)
  }, [payload, points])

  const statusLookup = useMemo(() => {
    const map = new Map<string, string>()
    for (const stop of deliveryStatuses) {
      const cid = Number(stop.customer_id)
      map.set(String(cid), stop.status)
      if (stop.route_index != null) {
        map.set(`${stop.route_index}_${cid}`, stop.status)
      }
    }
    return map
  }, [deliveryStatuses])

  const itineraryRoutes = useMemo(() => {
    return routes
      .map((route, routeIndex) => ({
        route,
        routeIndex,
        stops: (route.stops || []).filter(
          (stop) => Number(stop.customer_id) !== depotId
        ),
      }))
      .filter((item) => item.stops.length > 0)
  }, [routes, depotId])

  const firstPendingStop = useMemo(
    () => deliveryStatuses.find((s) => s.status === "pending"),
    [deliveryStatuses]
  )

  const evaluationNote = activeResult
    ? isDriver
      ? "Tổng thời gian giao hàng dự kiến gồm thời gian di chuyển và phục vụ, tính từ lúc khởi hành đến khi về kho. Thời gian thực tế có thể thay đổi theo quá trình giao hàng."
      : `Nghiệm đang hiển thị thuộc seed ${activeResult.seed ?? "không xác định"} · Thời gian từ dataset · Departure evaluator: ${activeResult.departure_time_mode}. Đường OSRM chỉ phục vụ hiển thị và có thể khác đường thực nghiệm. Duration gồm thời gian từ khởi hành đến về kho; objective = di chuyển + phục vụ.`
    : "Thời gian từ dataset · Objective = di chuyển + phục vụ. Bấm vào hàng để xem riêng từng tuyến."

  return (
    <div className="flex flex-col gap-4">
      {/* Metric Cards */}
      <div
        className={`grid gap-3 ${
          isDriver
            ? "grid-cols-1 sm:grid-cols-3"
            : "grid-cols-2 sm:grid-cols-3 xl:grid-cols-5"
        }`}
      >
        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Khách hàng</CardDescription>
            <CardTitle className="font-mono text-xl font-bold">
              {points.length > 0 ? customerCount : "—"}
            </CardTitle>
          </CardHeader>
        </Card>



        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">
              {isDriver ? "Tổng thời gian giao hàng (giây)" : "Fitness (giây)"}
            </CardDescription>
            <CardTitle className="font-mono text-xl font-bold">
              {activeResult
                ? fmt(isDriver ? deliveryDuration : activeResult.fitness)
                : "—"}
            </CardTitle>
          </CardHeader>
        </Card>



        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Tuyến xe</CardDescription>
            <CardTitle className="font-mono text-xl font-bold">
              {activeResult ? routes.length : "—"}
            </CardTitle>
          </CardHeader>
        </Card>



        {!isDriver && (
          <>
            <Card size="sm">
              <CardHeader className="pb-1">
                <CardDescription className="text-xs">
                  Thời gian chạy (s)
                </CardDescription>
                <CardTitle className="font-mono text-xl font-bold">
                  {activeResult ? fmt(activeResult.runtime_seconds) : "—"}
                </CardTitle>
              </CardHeader>
            </Card>



            <Card size="sm">
              <CardHeader className="pb-1">
                <CardDescription className="text-xs">
                  Seed tốt nhất
                </CardDescription>
                <CardTitle className="font-mono text-xl font-bold">
                  {activeResult?.seed != null ? activeResult.seed : "—"}
                </CardTitle>
              </CardHeader>
            </Card>


          </>
        )}
      </div>

      {/* Routes Table */}
      <Card size="sm">
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
          <div className="min-w-0">
            <CardTitle>Thông tin các tuyến</CardTitle>
            <CardDescription className="mt-1 text-xs leading-relaxed">
              {evaluationNote}
            </CardDescription>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-1.5">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onFitMap}
              title="Xem toàn cảnh các xe trên bản đồ"
            >
              <Maximize2Icon data-icon="inline-start" />
              Toàn cảnh
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={!payload}
              onClick={onExportJSON}
            >
              <FileJsonIcon data-icon="inline-start" />
              Tải JSON
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={!activeResult}
              onClick={onExportCSV}
            >
              <FileSpreadsheetIcon data-icon="inline-start" />
              Tải CSV
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-border bg-muted/50 text-muted-foreground">
                  <th className="px-3 py-2 font-medium whitespace-nowrap">Xe</th>
                  <th className="px-3 py-2 font-medium">Thứ tự ghé thăm</th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap">
                    Tải hàng
                  </th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap">
                    Khởi hành
                  </th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap">
                    Di chuyển (s)
                  </th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap">
                    Phục vụ (s)
                  </th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap">
                    Duration (s)
                  </th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap">
                    Nguồn đường vẽ
                  </th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap">
                    Giao tuyến
                  </th>
                  <th className="px-3 py-2 font-medium whitespace-nowrap text-right">
                    In ấn
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {routes.length === 0 ? (
                  <tr>
                    <td
                      colSpan={10}
                      className="px-3 py-6 text-center text-muted-foreground"
                    >
                      Chưa có kết quả.
                    </td>
                  </tr>
                ) : (
                  routes.map((r, i) => {
                    const color = ROUTE_COLORS[i % ROUTE_COLORS.length]
                    const assigned = getAssignedTrip(i)
                    const selectedDriverId =
                      routeDriverMap[i] ?? drivers[0]?.id ?? 0

                    return (
                      <tr
                        key={`${r.vehicle}-${i}`}
                        onClick={() => onFocusRoute(i)}
                        className="cursor-pointer transition-colors hover:bg-muted/50"
                      >
                        <td className="px-3 py-2 font-medium whitespace-nowrap">
                          <span className="inline-flex items-center gap-1.5">
                            <span
                              className="inline-block size-2.5 shrink-0 rounded-full"
                              style={{ backgroundColor: color }}
                            />
                            Xe {r.vehicle}
                          </span>
                        </td>
                        <td className="max-w-[260px] px-3 py-2 font-mono text-[11px] leading-relaxed">
                          {r.visits.join(" → ")}
                        </td>
                        <td className="px-3 py-2 font-mono whitespace-nowrap">
                          {fmt(r.load)}
                        </td>
                        <td className="px-3 py-2 font-mono whitespace-nowrap">
                          {clock(r.departure_time)}
                        </td>
                        <td className="px-3 py-2 font-mono whitespace-nowrap">
                          {fmt(r.travel_time)}
                        </td>
                        <td className="px-3 py-2 font-mono whitespace-nowrap">
                          {fmt(r.service_time)}
                        </td>
                        <td className="px-3 py-2 font-mono font-medium whitespace-nowrap">
                          {fmt(r.duration)}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap">
                          <Badge variant="outline">
                            {r.geometry_source === "osrm"
                              ? "OSRM"
                              : r.geometry_source === "dataset-road-graph"
                                ? "Road graph dataset"
                                : "Đường thẳng dự phòng"}
                          </Badge>
                        </td>
                        <td
                          className="px-3 py-2 whitespace-nowrap"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {assigned ? (
                            <Badge
                              variant="secondary"
                              title={`Trạng thái: ${statusLabel(assigned.status)} · Đã giao: ${assigned.stops_completed}/${assigned.stops_total} điểm`}
                            >
                              Đã giao:{" "}
                              {driverDisplayName(
                                assigned.driver_username,
                                assigned.driver_full_name
                              )}
                            </Badge>
                          ) : !isDriver &&
                            currentJobId &&
                            r.visits.length > 2 ? (
                            <div className="flex items-center gap-1.5">
                              <select
                                value={selectedDriverId}
                                disabled={
                                  drivers.length === 0 || assigningIndex === i
                                }
                                onChange={(e) =>
                                  setRouteDriverMap((prev) => ({
                                    ...prev,
                                    [i]: Number(e.target.value),
                                  }))
                                }
                                className="h-7 rounded-md border border-input bg-background px-2 text-xs text-foreground outline-none focus-visible:border-ring"
                              >
                                {drivers.map((d) => (
                                  <option key={d.id} value={d.id}>
                                    {driverDisplayName(d.username, d.full_name)}
                                  </option>
                                ))}
                              </select>
                              <Button
                                type="button"
                                size="xs"
                                disabled={
                                  !selectedDriverId || assigningIndex === i
                                }
                                onClick={async () => {
                                  setAssigningIndex(i)
                                  try {
                                    await onAssignRoute(
                                      selectedAlgorithm,
                                      i,
                                      selectedDriverId
                                    )
                                  } finally {
                                    setAssigningIndex(null)
                                  }
                                }}
                              >
                                {assigningIndex === i ? (
                                  <Loader2Icon
                                    data-icon="inline-start"
                                    className="animate-spin"
                                  />
                                ) : (
                                  <SendIcon data-icon="inline-start" />
                                )}
                                Giao Driver
                              </Button>
                            </div>
                          ) : currentTrip ? (
                            <span className="text-muted-foreground">
                              {driverDisplayName(
                                currentTrip.driver_username,
                                currentTrip.driver_full_name
                              )}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">
                              Route rỗng
                            </span>
                          )}
                        </td>
                        <td
                          className="px-3 py-2 text-right whitespace-nowrap"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <Button
                            type="button"
                            variant="outline"
                            size="xs"
                            title={`In lệnh giao hàng A4 cho Xe ${r.vehicle}`}
                            onClick={() =>
                              setManifestTarget({
                                jobId:
                                  currentJobId ||
                                  payload?.config?.trip_name ||
                                  "tdvrp-job",
                                routeIndex: i,
                                vehicle: r.vehicle,
                                algorithm: selectedAlgorithm,
                                tripName:
                                  payload?.config?.trip_name ||
                                  `Chuyến ${payload?.config?.dataset || ""}`,
                                dataset: payload?.config?.dataset,
                                trip: assigned,
                                routeResult: r,
                                points: points,
                                capacity: effectiveCapacity,
                              })
                            }
                          >
                            <PrinterIcon data-icon="inline-start" />
                            In lệnh
                          </Button>
                        </td>
                      </tr>
                    )
                  })
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Vehicle Capacity Balance Progress Bars */}
      {routes.length > 0 && (
        <VehicleCapacityBalance
          routes={routes}
          capacity={effectiveCapacity}
          drivers={drivers}
          allTrips={allTrips}
          currentJobId={currentJobId}
          onFocusRoute={onFocusRoute}
        />
      )}

      {/* Mini Gantt Chart - Time-Dependent Vehicle Routing Schedule */}
      {routes.length > 0 && (
        <RouteGanttChart
          routes={routes}
          points={payload?.points ?? points}
          onFocusRoute={onFocusRoute}
        />
      )}



      {/* Driver Stops Progress Panel */}
      {showStopsPanel && (
        <Card size="sm">
          <CardHeader>
            <div className="flex items-center justify-between gap-2">
              <div>
                <CardTitle>Tiến độ từng điểm giao</CardTitle>
                {currentTrip && (
                  <CardDescription className="mt-0.5 text-xs">
                    Xe {currentTrip.route_index + 1} · Driver:{" "}
                    {driverDisplayName(
                      currentTrip.driver_username,
                      currentTrip.driver_full_name
                    )}{" "}
                    · Trạng thái: {statusLabel(currentTrip.status)}
                  </CardDescription>
                )}
              </div>
              {currentTrip && (
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="xs"
                    onClick={() => {
                      const rIdx = currentTrip.route_index
                      const r = routes[rIdx]
                      setManifestTarget({
                        jobId: currentJobId || currentTrip.job_id,
                        routeIndex: rIdx,
                        vehicle: r?.vehicle || rIdx + 1,
                        algorithm: selectedAlgorithm || currentTrip.algorithm,
                        tripName:
                          payload?.config?.trip_name ||
                          `Chuyến ${payload?.config?.dataset || ""}`,
                        dataset: payload?.config?.dataset,
                        trip: currentTrip,
                        routeResult: r,
                        points: points,
                        capacity: effectiveCapacity,
                      })
                    }}
                  >
                    <PrinterIcon data-icon="inline-start" />
                    In lệnh giao của tôi
                  </Button>
                  <Badge variant="secondary">
                    {
                      deliveryStatuses.filter((s) => s.status === "completed")
                        .length
                    }
                    /{deliveryStatuses.length} điểm
                  </Badge>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/50 text-muted-foreground">
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Thứ tự
                    </th>
                    <th className="px-3 py-2 font-medium">Khách hàng</th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Trạng thái
                    </th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Hoàn thành lúc
                    </th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Thao tác
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {deliveryStatuses.map((stop) => {
                    const point = pointMap.get(Number(stop.customer_id))
                    const canComplete =
                      isDriver &&
                      currentTrip?.status === "in_progress" &&
                      firstPendingStop?.stop_index === stop.stop_index
                    const isCompleting = completingStop === stop.stop_index
                    const hasFailed = failedStop === stop.stop_index

                    return (
                      <tr key={stop.stop_index}>
                        <td className="px-3 py-2 font-mono">
                          {stop.stop_index + 1}
                        </td>
                        <td className="px-3 py-2 font-medium">
                          {point?.name || `Khách hàng ${stop.customer_id}`}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap">
                          <Badge
                            variant={
                              stop.status === "completed"
                                ? "default"
                                : "secondary"
                            }
                          >
                            {deliveryStatusLabel(stop.status)}
                          </Badge>
                        </td>
                        <td className="px-3 py-2 font-mono text-muted-foreground whitespace-nowrap">
                          {stop.completed_at || "—"}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap">
                          {canComplete && (
                            <Button
                              type="button"
                              size="xs"
                              disabled={isCompleting}
                              onClick={async () => {
                                setCompletingStop(stop.stop_index)
                                setFailedStop(null)
                                try {
                                  const ok = await onCompleteStop(
                                    stop.stop_index
                                  )
                                  if (!ok) setFailedStop(stop.stop_index)
                                } finally {
                                  setCompletingStop(null)
                                }
                              }}
                            >
                              {isCompleting ? (
                                <Loader2Icon
                                  data-icon="inline-start"
                                  className="animate-spin"
                                />
                              ) : (
                                <CheckCircle2Icon data-icon="inline-start" />
                              )}
                              {isCompleting
                                ? "Đang cập nhật…"
                                : hasFailed
                                  ? "Thử lại"
                                  : "Hoàn thành điểm"}
                            </Button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>


      )}

      {/* Detailed Itinerary Timeline */}
      {itineraryRoutes.length > 0 && (
        <Card size="sm">
          <CardHeader>
            <CardTitle>Lộ trình chi tiết theo điểm giao</CardTitle>
            <CardDescription className="text-xs">
              ETA và thời gian chặng được lấy từ bộ đánh giá TDVRP, không lấy từ
              đường vẽ OSRM.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {itineraryRoutes.map(({ route, routeIndex, stops }) => {
              const color = ROUTE_COLORS[routeIndex % ROUTE_COLORS.length]
              return (
                <section
                  key={`${route.vehicle}-${routeIndex}`}
                  className="flex flex-col gap-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-muted/40 px-3 py-1.5 transition-colors hover:bg-muted/70">
                    <button
                      type="button"
                      onClick={() => onFocusRoute(routeIndex)}
                      className="flex flex-wrap items-center gap-2 text-left text-xs hover:underline"
                    >
                      <span
                        className="inline-block size-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <strong className="font-semibold text-foreground">
                        Xe {route.vehicle}
                      </strong>
                      <span className="text-muted-foreground">
                        {stops.length} điểm · tải {fmt(route.load)} ·{" "}
                        {clock(route.departure_time)}–{clock(route.end_time)}
                      </span>
                    </button>
                    <Button
                      type="button"
                      variant="outline"
                      size="xs"
                      onClick={() =>
                        setManifestTarget({
                          jobId:
                            currentJobId ||
                            payload?.config?.trip_name ||
                            "tdvrp-job",
                          routeIndex,
                          vehicle: route.vehicle,
                          algorithm: selectedAlgorithm,
                          tripName:
                            payload?.config?.trip_name ||
                            `Chuyến ${payload?.config?.dataset || ""}`,
                          dataset: payload?.config?.dataset,
                          trip: getAssignedTrip(routeIndex),
                          routeResult: route,
                          points: points,
                          capacity: effectiveCapacity,
                        })
                      }
                    >
                      <PrinterIcon data-icon="inline-start" />
                      In lệnh giao
                    </Button>
                  </div>

                  <div className="overflow-x-auto rounded-lg border border-border">
                    <table className="w-full border-collapse text-left text-xs">
                      <thead>
                        <tr className="border-b border-border bg-muted/30 text-muted-foreground">
                          <th className="px-2.5 py-2 font-medium">STT</th>
                          <th className="px-2.5 py-2 font-medium">Điểm giao</th>
                          <th className="px-2.5 py-2 font-medium">
                            Địa chỉ / tọa độ
                          </th>
                          <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                            Nhu cầu
                          </th>
                          <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                            ETA
                          </th>
                          <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                            Phục vụ
                          </th>
                          <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                            Rời điểm
                          </th>
                          <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                            Từ điểm trước
                          </th>
                          <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                            Trạng thái
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {stops.map((stop, idx) => {
                          const customerId = Number(stop.customer_id)
                          const point = pointMap.get(customerId)
                          const leg = (route.legs || []).find(
                            (item) => Number(item.to_customer_id) === customerId
                          )
                          const statusValue =
                            statusLookup.get(`${routeIndex}_${customerId}`) ??
                            statusLookup.get(String(customerId))
                          const address = String(point?.address ?? "").trim()
                          const coords =
                            point &&
                            Number.isFinite(point.lat) &&
                            Number.isFinite(point.lon)
                              ? `${point.lat.toFixed(6)}, ${point.lon.toFixed(6)}`
                              : "—"

                          return (
                            <tr key={`${customerId}-${idx}`}>
                              <td className="px-2.5 py-2 font-mono">
                                {idx + 1}
                              </td>
                              <td className="px-2.5 py-2 font-medium whitespace-nowrap">
                                {point?.name || `Khách hàng ${customerId}`}
                              </td>
                              <td className="max-w-[240px] truncate px-2.5 py-2 text-muted-foreground">
                                {address || coords}
                              </td>
                              <td className="px-2.5 py-2 font-mono whitespace-nowrap">
                                {fmt(stop.demand ?? point?.demand ?? 0)}
                              </td>
                              <td className="px-2.5 py-2 font-mono whitespace-nowrap">
                                {clock(stop.arrival_time)}
                              </td>
                              <td className="px-2.5 py-2 font-mono whitespace-nowrap">
                                {fmt(stop.service_time)} s
                              </td>
                              <td className="px-2.5 py-2 font-mono whitespace-nowrap">
                                {clock(stop.departure_time)}
                              </td>
                              <td className="px-2.5 py-2 font-mono whitespace-nowrap">
                                {fmt(leg?.travel_time ?? 0)} s
                              </td>
                              <td className="px-2.5 py-2 whitespace-nowrap">
                                <Badge
                                  variant={
                                    statusValue === "completed"
                                      ? "default"
                                      : "secondary"
                                  }
                                >
                                  {deliveryStatusLabel(statusValue)}
                                </Badge>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>
              )
            })}
          </CardContent>
        </Card>


      )}

      {/* Algorithm Comparison Table (Admin only) */}
      {!isDriver && payload && algorithmNames.length > 0 && (
        <Card size="sm">
          <CardHeader>
            <CardTitle>Kết quả cùng lượt chạy</CardTitle>
            <CardDescription className="text-xs">
              RAPO dùng departure exact; Pufferfish và GA xuất trace với
              departure quét lưới.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/50 text-muted-foreground">
                    <th className="px-3 py-2 font-medium">Thuật toán</th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Seed tốt nhất
                    </th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Fitness (s)
                    </th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      NFE
                    </th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Runtime (s)
                    </th>
                    <th className="px-3 py-2 font-medium whitespace-nowrap">
                      Departure evaluator
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {algorithmNames.map((name) => {
                    const r = payload.results[name]
                    return (
                      <tr key={name}>
                        <td className="px-3 py-2 font-medium">
                          {algorithmLabel(name)}
                        </td>
                        <td className="px-3 py-2 font-mono">
                          {r.seed ?? "—"}
                        </td>
                        <td className="px-3 py-2 font-mono font-medium">
                          {fmt(r.fitness)}
                        </td>
                        <td className="px-3 py-2 font-mono">{r.nfe}</td>
                        <td className="px-3 py-2 font-mono">
                          {fmt(r.runtime_seconds)}
                        </td>
                        <td className="px-3 py-2 text-muted-foreground">
                          {r.departure_time_mode}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>


      )}

      {/* A4 Dispatch Manifest Modal */}
      <DispatchManifestModal
        open={Boolean(manifestTarget)}
        onClose={() => setManifestTarget(null)}
        target={manifestTarget}
      />
    </div>
  )
}
