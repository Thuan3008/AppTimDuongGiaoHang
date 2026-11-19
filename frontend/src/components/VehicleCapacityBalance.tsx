import { useMemo } from "react"
import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  GaugeIcon,
  ScaleIcon,
  ShieldCheckIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ROUTE_COLORS, driverDisplayName, fmt } from "@/lib/api"
import type { Driver, RouteResult, Trip } from "@/types/rapo"

interface VehicleCapacityBalanceProps {
  routes: RouteResult[]
  capacity: number
  drivers?: Driver[]
  allTrips?: Trip[]
  currentJobId?: string | null
  onFocusRoute?: (index: number) => void
}

interface VehicleBalanceItem {
  route: RouteResult
  routeIndex: number
  vehicleNumber: number
  color: string
  load: number
  capacity: number
  percentage: number
  stopCount: number
  driverName?: string
  status: "safe" | "optimal" | "critical"
  statusLabel: string
  badgeClass: string
  barClass: string
}

export function VehicleCapacityBalance({
  routes,
  capacity,
  drivers = [],
  allTrips = [],
  currentJobId,
  onFocusRoute,
}: VehicleCapacityBalanceProps) {
  const effectiveCapacity = capacity > 0 ? capacity : 100

  const items = useMemo<VehicleBalanceItem[]>(() => {
    return routes.map((route, routeIndex) => {
      const color = ROUTE_COLORS[routeIndex % ROUTE_COLORS.length]
      const load = Number(route.load || 0)
      const pct = (load / effectiveCapacity) * 100
      const stopCount = Math.max(0, (route.visits?.length || 0) - 2)

      // Find driver assigned to this route if any
      let driverName: string | undefined
      if (currentJobId) {
        const trip = allTrips.find(
          (t) =>
            t.job_id === currentJobId &&
            Number(t.route_index) === Number(routeIndex)
        )
        if (trip) {
          const matchedDriver = drivers.find((d) => d.id === trip.driver_id)
          driverName = driverDisplayName(
            trip.driver_username || matchedDriver?.username || "",
            trip.driver_full_name || matchedDriver?.full_name
          )
        }
      }

      let status: "safe" | "optimal" | "critical"
      let statusLabel: string
      let badgeClass: string
      let barClass: string

      if (pct > 95) {
        status = "critical"
        statusLabel = pct > 100 ? "Quá tải (> 100%)" : "Cận tải (> 95%)"
        badgeClass =
          "border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-300"
        barClass = "bg-rose-500 shadow-xs shadow-rose-500/50"
      } else if (pct >= 80) {
        status = "optimal"
        statusLabel = "Tối ưu (80 - 95%)"
        badgeClass =
          "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-300"
        barClass = "bg-amber-500 shadow-xs shadow-amber-500/50"
      } else {
        status = "safe"
        statusLabel = "Tải an toàn (< 80%)"
        badgeClass =
          "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300"
        barClass = "bg-emerald-500 shadow-xs shadow-emerald-500/50"
      }

      return {
        route,
        routeIndex,
        vehicleNumber: route.vehicle ?? routeIndex + 1,
        color,
        load,
        capacity: effectiveCapacity,
        percentage: Math.round(pct * 10) / 10,
        stopCount,
        driverName,
        status,
        statusLabel,
        badgeClass,
        barClass,
      }
    })
  }, [routes, effectiveCapacity, allTrips, drivers, currentJobId])

  // Fleet load balance KPIs
  const fleetStats = useMemo(() => {
    if (routes.length === 0) return null
    const totalLoad = routes.reduce((sum, r) => sum + Number(r.load || 0), 0)
    const totalCapacity = routes.length * effectiveCapacity
    const utilization = (totalLoad / totalCapacity) * 100
    const loads = routes.map((r) => Number(r.load || 0))
    const minLoad = Math.min(...loads)
    const maxLoad = Math.max(...loads)
    const diff = maxLoad - minLoad
    const avgLoad = totalLoad / routes.length

    return {
      totalLoad,
      totalCapacity,
      utilization: Math.round(utilization * 10) / 10,
      minLoad,
      maxLoad,
      diff,
      avgLoad: Math.round(avgLoad * 10) / 10,
    }
  }, [routes, effectiveCapacity])

  if (routes.length === 0) return null

  return (
    <Card size="sm" className="overflow-hidden">
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <ScaleIcon className="size-4 text-primary" />
            <CardTitle>Cân bằng tải trọng phương tiện</CardTitle>
            <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary">
              Vehicle Capacity Balance
            </span>
          </div>
          <CardDescription className="mt-1 text-xs">
            Đánh giá mức độ lấp đầy và cân đối hàng hóa giữa các xe theo tải trọng cho phép{" "}
            <strong>{effectiveCapacity} kg</strong>/xe.
          </CardDescription>
        </div>

        {/* Legend indicator */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5 font-medium text-emerald-700 dark:text-emerald-400">
            <span className="size-2.5 rounded-full bg-emerald-500" />
            An toàn (&lt; 80%)
          </span>
          <span className="flex items-center gap-1.5 font-medium text-amber-700 dark:text-amber-400">
            <span className="size-2.5 rounded-full bg-amber-500" />
            Tối ưu (80–95%)
          </span>
          <span className="flex items-center gap-1.5 font-medium text-rose-700 dark:text-rose-400">
            <span className="size-2.5 rounded-full bg-rose-500" />
            Cận tải (&gt; 95%)
          </span>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        {/* Fleet Summary Mini KPIs */}
        {fleetStats && (
          <div className="grid grid-cols-2 gap-2 rounded-lg border border-border/70 bg-muted/30 p-3 sm:grid-cols-4 sm:gap-3">
            <div>
              <span className="text-[11px] text-muted-foreground">Tổng tải toàn đoàn</span>
              <div className="font-mono text-base font-bold text-foreground">
                {fmt(fleetStats.totalLoad)} / {fmt(fleetStats.totalCapacity)} kg
              </div>
            </div>
            <div>
              <span className="text-[11px] text-muted-foreground">Tỷ lệ lấp đầy bình quân</span>
              <div className="flex items-center gap-1.5 font-mono text-base font-bold text-foreground">
                <GaugeIcon className="size-4 text-primary" />
                {fleetStats.utilization}%
              </div>
            </div>
            <div>
              <span className="text-[11px] text-muted-foreground">Tải trung bình mỗi xe</span>
              <div className="font-mono text-base font-bold text-foreground">
                {fmt(fleetStats.avgLoad)} kg
              </div>
            </div>
            <div>
              <span className="text-[11px] text-muted-foreground">Độ lệch tải (Min → Max)</span>
              <div className="font-mono text-base font-bold text-foreground">
                {fmt(fleetStats.minLoad)} → {fmt(fleetStats.maxLoad)} kg{" "}
                <span className="text-xs font-normal text-muted-foreground">
                  (Δ {fmt(fleetStats.diff)})
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Vehicle Progress Bars */}
        <div className="flex flex-col divide-y divide-border/60 rounded-lg border border-border bg-card">
          {items.map((item) => {
            const widthPct = Math.min(100, Math.max(0, item.percentage))
            return (
              <div
                key={`cap-item-${item.vehicleNumber}-${item.routeIndex}`}
                onClick={() => onFocusRoute?.(item.routeIndex)}
                className="group flex flex-col gap-2 p-3 transition hover:bg-muted/40 sm:flex-row sm:items-center sm:gap-4 cursor-pointer"
                title={`Bấm để xem tuyến Xe ${item.vehicleNumber} trên bản đồ`}
              >
                {/* Vehicle identification */}
                <div className="flex min-w-[150px] shrink-0 items-center gap-2.5">
                  <span
                    className="size-3 shrink-0 rounded-full ring-2 ring-background"
                    style={{ backgroundColor: item.color }}
                  />
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="font-bold text-foreground">
                        Xe {item.vehicleNumber}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        ({item.stopCount} điểm)
                      </span>
                    </div>
                    {item.driverName && (
                      <div className="text-[11px] text-muted-foreground truncate max-w-[140px]">
                        {item.driverName}
                      </div>
                    )}
                  </div>
                </div>

                {/* Progress bar container */}
                <div className="flex flex-1 flex-col justify-center">
                  <div className="relative h-4 w-full overflow-hidden rounded-full bg-muted">
                    {/* Visual markers at 80% and 95% */}
                    <div
                      className="absolute top-0 bottom-0 z-10 w-[1.5px] bg-foreground/25"
                      style={{ left: "80%" }}
                      title="Ngưỡng tải tối ưu 80%"
                    />
                    <div
                      className="absolute top-0 bottom-0 z-10 w-[1.5px] bg-foreground/35"
                      style={{ left: "95%" }}
                      title="Ngưỡng cận tải 95%"
                    />

                    {/* Progress fill */}
                    <div
                      className={`h-full transition-all duration-500 ease-out rounded-full ${item.barClass}`}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>

                  {/* Tick labels */}
                  <div className="relative mt-0.5 flex justify-between text-[10px] text-muted-foreground/75 font-mono">
                    <span>0%</span>
                    <span className="absolute" style={{ left: "80%", transform: "translateX(-50%)" }}>
                      80%
                    </span>
                    <span className="absolute" style={{ left: "95%", transform: "translateX(-50%)" }}>
                      95%
                    </span>
                    <span>100%</span>
                  </div>
                </div>

                {/* Numeric load & status badge */}
                <div className="flex shrink-0 items-center justify-between gap-3 sm:justify-end sm:min-w-[190px]">
                  <div className="text-right">
                    <span className="font-mono text-sm font-bold text-foreground">
                      {fmt(item.load)}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {" "}/ {item.capacity} kg
                    </span>
                  </div>

                  <Badge
                    variant="outline"
                    className={`gap-1 font-mono text-xs font-semibold ${item.badgeClass}`}
                  >
                    {item.status === "safe" && <ShieldCheckIcon className="size-3" />}
                    {item.status === "optimal" && <CheckCircle2Icon className="size-3" />}
                    {item.status === "critical" && <AlertTriangleIcon className="size-3" />}
                    {item.percentage}%
                  </Badge>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
