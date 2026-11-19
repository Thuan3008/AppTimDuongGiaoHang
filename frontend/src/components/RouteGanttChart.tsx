import { useMemo, useState } from "react"
import { CalendarClockIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ROUTE_COLORS, clock, fmt } from "@/lib/api"
import type { MapPoint, RouteResult } from "@/types/rapo"

interface RouteGanttChartProps {
  routes: RouteResult[]
  points?: MapPoint[]
  onFocusRoute?: (index: number) => void
}

interface TimelineSegment {
  id: string
  type: "travel" | "service" | "waiting"
  start: number
  end: number
  durationSec: number
  label: string
  detail: string
  stopIndex?: number
  customerId?: number
}

interface VehicleGanttData {
  route: RouteResult
  routeIndex: number
  vehicleNumber: number
  color: string
  departureTime: number
  endTime: number
  durationSec: number
  travelTimeSec: number
  serviceTimeSec: number
  stopCount: number
  segments: TimelineSegment[]
}

export function RouteGanttChart({
  routes,
  points = [],
  onFocusRoute,
}: RouteGanttChartProps) {
  const [selectedVehicleFilter, setSelectedVehicleFilter] = useState<number | "all">("all")

  const pointMap = useMemo(() => {
    return new Map(points.map((p) => [Number(p.id), p]))
  }, [points])

  const depotId = useMemo(() => {
    return Number(points.find((p) => p.depot)?.id ?? 0)
  }, [points])

  // Parse segments for each vehicle
  const vehicleData = useMemo<VehicleGanttData[]>(() => {
    return routes.map((route, routeIndex) => {
      const color = ROUTE_COLORS[routeIndex % ROUTE_COLORS.length]
      const vehicleNumber = route.vehicle ?? routeIndex + 1
      const departureTime = Number(route.departure_time || 0)
      const endTime = Number(route.end_time || departureTime + (route.duration || 0))
      const durationSec = Number(route.duration || endTime - departureTime)
      const travelTimeSec = Number(route.travel_time || 0)
      const serviceTimeSec = Number(route.service_time || 0)
      const stopCount = Math.max(0, (route.visits?.length || 0) - 2)

      const segments: TimelineSegment[] = []

      if (route.stops && route.stops.length >= 2) {
        let customerOrder = 1
        for (let i = 0; i < route.stops.length - 1; i++) {
          const cur = route.stops[i]
          const next = route.stops[i + 1]

          // 1. Travel segment between cur and next
          const tDepart = cur.departure_time
          const tArrive = next.arrival_time
          const tTravel = Math.max(0, tArrive - tDepart)

          if (tTravel > 0) {
            const fromName =
              cur.customer_id === depotId
                ? "Kho xuất phát"
                : pointMap.get(cur.customer_id)?.name || `Khách #${cur.customer_id}`
            const toName =
              next.customer_id === depotId
                ? "Kho về đích"
                : pointMap.get(next.customer_id)?.name || `Khách #${next.customer_id}`

            segments.push({
              id: `travel-${i}-${cur.customer_id}-${next.customer_id}`,
              type: "travel",
              start: tDepart,
              end: tArrive,
              durationSec: tTravel,
              label: `${Math.round(tTravel / 60)}p`,
              detail: `Di chuyển: ${fromName} → ${toName} | Thời gian: ${Math.round(
                tTravel / 60
              )} phút (${clock(tDepart)} → ${clock(tArrive)}) | Vận tốc TDVRP theo giờ`,
            })
          }

          // 2. Stop segment at next
          if (next.customer_id !== depotId) {
            const nextStart = next.service_start_time ?? next.arrival_time
            const tWait = Math.max(0, nextStart - next.arrival_time)

            // Waiting if any
            if (tWait > 0) {
              segments.push({
                id: `wait-${i}-${next.customer_id}`,
                type: "waiting",
                start: next.arrival_time,
                end: nextStart,
                durationSec: tWait,
                label: "Chờ",
                detail: `Chờ mở cửa: ${Math.round(tWait / 60)} phút (${clock(
                  next.arrival_time
                )} → ${clock(nextStart)}) tại Khách #${next.customer_id}`,
              })
            }

            // Service
            const tServ = Math.max(0, next.departure_time - nextStart)
            const pt = pointMap.get(next.customer_id)
            const cName = pt?.name || `Khách hàng #${next.customer_id}`

            segments.push({
              id: `serv-${i}-${next.customer_id}`,
              type: "service",
              start: nextStart,
              end: next.departure_time,
              durationSec: tServ,
              stopIndex: customerOrder,
              customerId: next.customer_id,
              label: `${customerOrder}`,
              detail: `Điểm giao #${customerOrder}: ${cName} | Phục vụ: ${Math.round(
                tServ / 60
              )} phút (${clock(nextStart)} → ${clock(next.departure_time)}) | Nhu cầu: ${fmt(
                next.demand ?? pt?.demand ?? 0
              )} kg`,
            })
            customerOrder++
          }
        }
      } else {
        // Fallback proportional representation if detailed stops trace is unavailable
        const halfTravel = travelTimeSec / 2
        segments.push({
          id: `fallback-travel-1-${routeIndex}`,
          type: "travel",
          start: departureTime,
          end: departureTime + halfTravel,
          durationSec: halfTravel,
          label: `${Math.round(halfTravel / 60)}p`,
          detail: `Di chuyển đi: ${Math.round(halfTravel / 60)} phút (${clock(
            departureTime
          )} → ${clock(departureTime + halfTravel)})`,
        })

        if (serviceTimeSec > 0) {
          segments.push({
            id: `fallback-service-${routeIndex}`,
            type: "service",
            start: departureTime + halfTravel,
            end: departureTime + halfTravel + serviceTimeSec,
            durationSec: serviceTimeSec,
            label: "Giao hàng",
            detail: `Tổng thời gian phục vụ ${stopCount} điểm: ${Math.round(
              serviceTimeSec / 60
            )} phút`,
          })
        }

        segments.push({
          id: `fallback-travel-2-${routeIndex}`,
          type: "travel",
          start: departureTime + halfTravel + serviceTimeSec,
          end: endTime,
          durationSec: halfTravel,
          label: `${Math.round(halfTravel / 60)}p`,
          detail: `Di chuyển về kho: ${Math.round(halfTravel / 60)} phút (${clock(
            departureTime + halfTravel + serviceTimeSec
          )} → ${clock(endTime)})`,
        })
      }

      return {
        route,
        routeIndex,
        vehicleNumber,
        color,
        departureTime,
        endTime,
        durationSec,
        travelTimeSec,
        serviceTimeSec,
        stopCount,
        segments,
      }
    })
  }, [routes, pointMap, depotId])

  // Global timeline boundaries
  const timelineBounds = useMemo(() => {
    if (vehicleData.length === 0) return { minTime: 28800, maxTime: 36000, span: 7200 }
    const starts = vehicleData.map((v) => v.departureTime)
    const ends = vehicleData.map((v) => v.endTime)
    const minTime = Math.min(...starts)
    const maxTime = Math.max(...ends)
    const span = Math.max(300, maxTime - minTime)
    return { minTime, maxTime, span }
  }, [vehicleData])

  // Time grid ticks (e.g. every 15, 30 or 60 min)
  const timeTicks = useMemo(() => {
    const { minTime, maxTime, span } = timelineBounds
    let interval = 1800 // 30 mins
    if (span <= 5400) interval = 900 // 15 mins
    else if (span > 18000) interval = 3600 // 60 mins

    const startTick = Math.floor(minTime / interval) * interval
    const ticks: { time: number; leftPct: number; label: string }[] = []

    for (let t = startTick; t <= maxTime + interval; t += interval) {
      if (t >= minTime - 60 && t <= maxTime + 60) {
        const leftPct = ((t - minTime) / span) * 100
        ticks.push({
          time: t,
          leftPct: Math.min(100, Math.max(0, leftPct)),
          label: clock(t),
        })
      }
    }
    return ticks
  }, [timelineBounds])

  // Aggregate totals
  const aggregates = useMemo(() => {
    const totalTravel = vehicleData.reduce((sum, v) => sum + v.travelTimeSec, 0)
    const totalService = vehicleData.reduce((sum, v) => sum + v.serviceTimeSec, 0)
    const totalDuration = vehicleData.reduce((sum, v) => sum + v.durationSec, 0)
    return {
      totalTravelMinutes: Math.round(totalTravel / 60),
      totalServiceMinutes: Math.round(totalService / 60),
      totalDurationMinutes: Math.round(totalDuration / 60),
    }
  }, [vehicleData])

  const visibleVehicles = useMemo(() => {
    if (selectedVehicleFilter === "all") return vehicleData
    return vehicleData.filter((v) => v.vehicleNumber === selectedVehicleFilter)
  }, [vehicleData, selectedVehicleFilter])

  if (routes.length === 0) return null

  const { minTime, span } = timelineBounds

  return (
    <Card size="sm" className="overflow-hidden">
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <CalendarClockIcon className="size-4 text-primary" />
            <CardTitle>Mini Gantt Chart · Lịch trình thời gian từng xe</CardTitle>
            <span className="rounded-full bg-blue-500/10 px-2 py-0.5 text-[11px] font-semibold text-blue-600 dark:text-blue-400">
              TDVRP Time Schedule
            </span>
          </div>
          <CardDescription className="mt-1 text-xs">
            Trục thời gian thể hiện rõ bản chất Time-Dependent: thời điểm rời kho, vận tốc biến thiên theo giờ, dừng bốc dỡ hàng và thời điểm về kho.
          </CardDescription>
        </div>

        {/* Filter controls */}
        <div className="flex flex-wrap items-center gap-1.5">
          <Button
            type="button"
            variant={selectedVehicleFilter === "all" ? "default" : "outline"}
            size="xs"
            onClick={() => setSelectedVehicleFilter("all")}
          >
            Tất cả xe ({vehicleData.length})
          </Button>
          {vehicleData.map((v) => (
            <Button
              key={`filter-btn-${v.vehicleNumber}`}
              type="button"
              variant={selectedVehicleFilter === v.vehicleNumber ? "default" : "outline"}
              size="xs"
              onClick={() => setSelectedVehicleFilter(v.vehicleNumber)}
              className="gap-1.5"
            >
              <span
                className="size-2 rounded-full"
                style={{ backgroundColor: v.color }}
              />
              Xe {v.vehicleNumber}
            </Button>
          ))}
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        {/* KPI Strip */}
        <div className="grid grid-cols-2 gap-2 rounded-lg border border-border/70 bg-muted/25 p-2.5 text-xs sm:grid-cols-4 sm:gap-3">
          <div>
            <span className="text-[11px] text-muted-foreground">Khoảng thời gian toàn đội</span>
            <div className="font-mono text-sm font-bold text-foreground">
              {clock(timelineBounds.minTime)} → {clock(timelineBounds.maxTime)}
            </div>
          </div>
          <div>
            <span className="text-[11px] text-muted-foreground">Tổng thời gian vận hành</span>
            <div className="font-mono text-sm font-bold text-foreground">
              {aggregates.totalDurationMinutes} phút
            </div>
          </div>
          <div>
            <span className="text-[11px] text-muted-foreground">Thời gian di chuyển trên đường</span>
            <div className="font-mono text-sm font-bold text-sky-600 dark:text-sky-400">
              {aggregates.totalTravelMinutes} phút
            </div>
          </div>
          <div>
            <span className="text-[11px] text-muted-foreground">Thời gian dừng bốc dỡ hàng</span>
            <div className="font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400">
              {aggregates.totalServiceMinutes} phút
            </div>
          </div>
        </div>

        {/* Gantt Viewport */}
        <div className="overflow-x-auto rounded-lg border border-border bg-card p-3">
          <div className="min-w-[620px]">
            {/* Time Axis Header */}
            <div className="relative mb-3 flex h-7 items-center border-b border-border pb-1">
              <div className="w-[170px] shrink-0 text-xs font-semibold text-muted-foreground">
                Tuyến xe / Lộ trình
              </div>
              <div className="relative flex-1 h-full">
                {timeTicks.map((tick, idx) => (
                  <div
                    key={`tick-${tick.time}-${idx}`}
                    className="absolute top-0 flex flex-col items-center -translate-x-1/2"
                    style={{ left: `${tick.leftPct}%` }}
                  >
                    <span className="font-mono text-[10px] font-medium text-muted-foreground">
                      {tick.label}
                    </span>
                    <div className="h-1.5 w-[1px] bg-border" />
                  </div>
                ))}
              </div>
            </div>

            {/* Vehicle Rows */}
            <div className="flex flex-col gap-3">
              {visibleVehicles.map((v) => {
                return (
                  <div
                    key={`gantt-row-${v.vehicleNumber}`}
                    className="group flex items-center rounded-md p-1 transition hover:bg-muted/40 cursor-pointer"
                    onClick={() => onFocusRoute?.(v.routeIndex)}
                    title={`Bấm để phóng to vào tuyến Xe ${v.vehicleNumber} trên bản đồ`}
                  >
                    {/* Vehicle Metadata Header */}
                    <div className="w-[170px] shrink-0 pr-3">
                      <div className="flex items-center gap-2">
                        <span
                          className="size-2.5 shrink-0 rounded-full"
                          style={{ backgroundColor: v.color }}
                        />
                        <span className="font-bold text-xs text-foreground">
                          Xe {v.vehicleNumber}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          ({v.stopCount} điểm)
                        </span>
                      </div>
                      <div className="font-mono text-[11px] text-muted-foreground/80 mt-0.5">
                        {clock(v.departureTime)} → {clock(v.endTime)}{" "}
                        <span className="text-foreground font-medium">
                          ({Math.round(v.durationSec / 60)}p)
                        </span>
                      </div>
                    </div>

                    {/* Gantt Track */}
                    <div className="relative flex-1 h-9 rounded-lg border border-border/60 bg-muted/20 overflow-hidden">
                      {/* Vertical guide lines */}
                      {timeTicks.map((tick, idx) => (
                        <div
                          key={`grid-${tick.time}-${idx}`}
                          className="absolute top-0 bottom-0 w-[1px] bg-border/40 pointer-events-none"
                          style={{ left: `${tick.leftPct}%` }}
                        />
                      ))}

                      {/* Segments */}
                      {v.segments.map((seg) => {
                        const leftPct = Math.max(
                          0,
                          Math.min(100, ((seg.start - minTime) / span) * 100)
                        )
                        const widthPct = Math.max(
                          0.8,
                          Math.min(100 - leftPct, ((seg.end - seg.start) / span) * 100)
                        )

                        if (seg.type === "travel") {
                          return (
                            <div
                              key={seg.id}
                              className="absolute top-1 bottom-1 rounded-sm bg-sky-500/80 hover:bg-sky-500 transition-colors flex items-center justify-center text-[10px] text-white font-medium shadow-2xs overflow-hidden"
                              style={{
                                left: `${leftPct}%`,
                                width: `${widthPct}%`,
                              }}
                              title={seg.detail}
                            >
                              {widthPct > 5 && (
                                <span className="truncate px-1 select-none">
                                  {seg.label}
                                </span>
                              )}
                            </div>
                          )
                        }

                        if (seg.type === "waiting") {
                          return (
                            <div
                              key={seg.id}
                              className="absolute top-1 bottom-1 rounded-sm bg-amber-400/90 hover:bg-amber-400 transition-colors flex items-center justify-center text-[9px] text-amber-950 font-bold shadow-2xs"
                              style={{
                                left: `${leftPct}%`,
                                width: `${widthPct}%`,
                              }}
                              title={seg.detail}
                            >
                              {widthPct > 4 && <span>W</span>}
                            </div>
                          )
                        }

                        // Service stop
                        return (
                          <div
                            key={seg.id}
                            className="absolute top-1 bottom-1 rounded-sm bg-emerald-500 hover:bg-emerald-400 transition-colors flex items-center justify-center text-[10px] text-white font-bold shadow-xs border border-emerald-600/30"
                            style={{
                              left: `${leftPct}%`,
                              width: `${widthPct}%`,
                            }}
                            title={seg.detail}
                          >
                            <span className="select-none truncate px-0.5">
                              {seg.label}
                            </span>
                          </div>
                        )
                      })}

                      {/* Start depot marker icon */}
                      <div
                        className="absolute top-0 bottom-0 flex items-center -translate-x-1/2 pointer-events-none z-10"
                        style={{
                          left: `${Math.max(
                            0,
                            Math.min(100, ((v.departureTime - minTime) / span) * 100)
                          )}%`,
                        }}
                      >
                        <span
                          className="size-3.5 rounded-full bg-red-600 text-white font-bold text-[8px] flex items-center justify-center shadow-xs ring-1 ring-background"
                          title={`Xuất phát từ kho: ${clock(v.departureTime)}`}
                        >
                          K
                        </span>
                      </div>

                      {/* Return depot marker icon */}
                      <div
                        className="absolute top-0 bottom-0 flex items-center -translate-x-1/2 pointer-events-none z-10"
                        style={{
                          left: `${Math.max(
                            0,
                            Math.min(100, ((v.endTime - minTime) / span) * 100)
                          )}%`,
                        }}
                      >
                        <span
                          className="size-3.5 rounded-full bg-slate-800 text-white font-bold text-[8px] flex items-center justify-center shadow-xs ring-1 ring-background"
                          title={`Về đến kho: ${clock(v.endTime)}`}
                        >
                          ✓
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border/70 bg-muted/20 px-3 py-2 text-xs">
          <div className="flex flex-wrap items-center gap-4">
            <span className="font-semibold text-muted-foreground">Chú giải Gantt:</span>
            <div className="flex items-center gap-1.5 font-medium">
              <span className="size-3 rounded-xs bg-sky-500" />
              <span>Di chuyển trên đường (Vận tốc TDVRP theo giờ)</span>
            </div>
            <div className="flex items-center gap-1.5 font-medium">
              <span className="size-3 rounded-xs bg-emerald-500" />
              <span>Dừng phục vụ khách (Số hiển thị: Thứ tự ghé thăm)</span>
            </div>
            <div className="flex items-center gap-1.5 font-medium">
              <span className="size-3 rounded-xs bg-amber-400" />
              <span>Chờ khung giờ mở cửa (Waiting)</span>
            </div>
          </div>

          <div className="flex items-center gap-3 text-muted-foreground font-mono text-[11px]">
            <span>K = Xuất phát từ kho</span>
            <span>✓ = Về kho hoàn tất</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
