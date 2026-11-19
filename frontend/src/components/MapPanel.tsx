import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react"
import L from "leaflet"
import "leaflet/dist/leaflet.css"
import {
  AlertTriangleIcon,
  CrosshairIcon,
  Maximize2Icon,
  XIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ROUTE_COLORS, algorithmLabel, clock, fmt } from "@/lib/api"
import type { MapPickTarget, MapPoint, RouteResult } from "@/types/rapo"

export interface MapPanelHandle {
  fit: () => void
  focusRoute: (index: number) => void
  invalidateSize: () => void
}

interface MapPanelProps {
  points: MapPoint[]
  routes: RouteResult[]
  algorithmNames: string[]
  selectedAlgorithm: string
  pendingPick: MapPickTarget | null
  onAlgorithmChange: (name: string) => void
  onCancelMapPick: () => void
  onMapPick: (lat: number, lon: number) => Promise<boolean>
}

export const MapPanel = forwardRef<MapPanelHandle, MapPanelProps>(
  function MapPanel(
    {
      points,
      routes,
      algorithmNames,
      selectedAlgorithm,
      pendingPick,
      onAlgorithmChange,
      onCancelMapPick,
      onMapPick,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement | null>(null)
    const mapRef = useRef<L.Map | null>(null)
    const markersRef = useRef<L.LayerGroup | null>(null)
    const linesRef = useRef<L.LayerGroup | null>(null)
    const routeLayersRef = useRef<L.GeoJSON[]>([])
    const pointsRef = useRef<MapPoint[]>(points)
    const pendingPickRef = useRef<MapPickTarget | null>(pendingPick)
    const onMapPickRef = useRef(onMapPick)

    const [tileWarning, setTileWarning] = useState<string | null>(null)
    const pointsCountRef = useRef<number>(0)

    useEffect(() => {
      pointsRef.current = points
      pendingPickRef.current = pendingPick
      onMapPickRef.current = onMapPick
    }, [points, pendingPick, onMapPick])

    const fitPoints = () => {
      const map = mapRef.current
      const currentPoints = pointsRef.current
      if (map && currentPoints.length > 0) {
        map.fitBounds(
          currentPoints.map((p) => [p.lat, p.lon] as [number, number]),
          {
            padding: [40, 40],
            maxZoom: 15,
          }
        )
      }
    }

    const focusRoute = (index: number) => {
      const map = mapRef.current
      const layer = routeLayersRef.current[index]
      if (map && layer) {
        const bounds = layer.getBounds()
        if (bounds.isValid()) {
          map.fitBounds(bounds, { padding: [40, 40] })
        }
        layer.openPopup()
      }
    }

    useImperativeHandle(ref, () => ({
      fit: fitPoints,
      focusRoute,
      invalidateSize: () => mapRef.current?.invalidateSize(),
    }))

    // Initialize Leaflet map once
    useEffect(() => {
      const el = containerRef.current
      if (!el || mapRef.current) return

      const map = L.map(el, { zoomControl: true }).setView([10.79, 106.66], 12)
      mapRef.current = map

      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      })
        .addTo(map)
        .on("tileerror", () => {
          setTileWarning(
            "Không tải được một số ô nền OpenStreetMap. Các điểm và đường tuyến vẫn hiển thị."
          )
        })

      markersRef.current = L.layerGroup().addTo(map)
      linesRef.current = L.layerGroup().addTo(map)

      map.on("click", async (event: L.LeafletMouseEvent) => {
        if (!pendingPickRef.current) return
        const accepted = await onMapPickRef.current(
          event.latlng.lat,
          event.latlng.lng
        )
        if (accepted && markersRef.current) {
          L.circleMarker(event.latlng, {
            radius: 8,
            color: "#dc2626",
            fillColor: "#ef4444",
            fillOpacity: 0.85,
            weight: 2,
          }).addTo(markersRef.current)
        }
      })

      const resizeObserver = new ResizeObserver(() => {
        map.invalidateSize()
      })
      resizeObserver.observe(el)

      const timer = window.setTimeout(() => map.invalidateSize(), 200)

      return () => {
        window.clearTimeout(timer)
        resizeObserver.disconnect()
        map.remove()
        mapRef.current = null
        markersRef.current = null
        linesRef.current = null
        routeLayersRef.current = []
      }
    }, [])

    // Update map cursor for map-picking mode
    useEffect(() => {
      const map = mapRef.current
      if (!map) return
      map.getContainer().style.cursor = pendingPick ? "crosshair" : ""
    }, [pendingPick])

    // Render markers whenever points or routes change
    useEffect(() => {
      const markers = markersRef.current
      if (!markers) return
      markers.clearLayers()

      const depotPoint = points.find((p) => Boolean(p.depot))
      const depotId = depotPoint ? String(depotPoint.id) : null

      // Build map of customer assignments across routes
      const assignmentMap = new Map<
        string,
        {
          vehicle: number
          routeIndex: number
          stopOrder: number
          totalStops: number
          color: string
        }
      >()

      routes.forEach((r, routeIndex) => {
        const color = ROUTE_COLORS[routeIndex % ROUTE_COLORS.length]
        const customerVisits = (r.visits || []).filter((id, idx) => {
          if (depotId != null) {
            return String(id) !== depotId
          }
          return idx > 0 && idx < r.visits.length - 1
        })

        customerVisits.forEach((customerId, stopIdx) => {
          assignmentMap.set(String(customerId), {
            vehicle: r.vehicle ?? routeIndex + 1,
            routeIndex,
            stopOrder: stopIdx + 1,
            totalStops: customerVisits.length,
            color,
          })
        })
      })

      for (const p of points) {
        const isDepot = Boolean(p.depot)

        if (isDepot) {
          const icon = L.divIcon({
            className: "depot-leaflet-icon",
            html: `
              <div class="depot-marker-wrapper">
                <div class="depot-radar-ping"></div>
                <div class="depot-radar-ping delay"></div>
                <div class="rapo-pin depot" title="Kho trung tâm (Depot)">K</div>
              </div>
            `,
            iconSize: [44, 44],
            iconAnchor: [22, 22],
            popupAnchor: [0, -22],
          })

          const addressLine = p.address
            ? `<div style="color: #475569; margin-bottom: 2px;">📍 ${p.address}</div>`
            : ""
          const snapLine = p.snap_distance_m
            ? `<div style="color: #059669; font-size: 11px; margin-top: 2px;">✓ Ghép đường OSRM: ${fmt(p.snap_distance_m)} m</div>`
            : ""

          const popupHtml = `
            <div style="font-family: inherit; font-size: 12px; line-height: 1.5; min-width: 190px;">
              <div style="font-weight: 700; font-size: 13px; color: #dc2626; margin-bottom: 4px; display: flex; align-items: center; gap: 4px;">
                🏢 ${p.name || "Kho trung tâm (Depot)"}
              </div>
              ${addressLine}
              <div style="color: #64748b;">Tọa độ: ${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}</div>
              ${snapLine}
              <div style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed #e2e8f0; font-size: 11px; color: #94a3b8;">
                Trung tâm xuất phát &amp; tập kết hàng
              </div>
            </div>
          `

          L.marker([p.lat, p.lon], {
            icon,
            zIndexOffset: 1000,
          })
            .addTo(markers)
            .bindPopup(popupHtml)
        } else {
          const assign = assignmentMap.get(String(p.id))
          const pinBg = assign ? assign.color : "#1e3a8a"
          const pinLabel = assign ? String(assign.stopOrder) : String(p.id)
          const titleAttr = assign
            ? `Xe ${assign.vehicle} · Điểm dừng số ${assign.stopOrder}/${assign.totalStops} (Khách #${p.id})`
            : `Khách hàng #${p.id} (Chưa phân tuyến)`

          const icon = L.divIcon({
            className: "customer-leaflet-icon",
            html: `<div class="rapo-pin" style="background-color: ${pinBg};" title="${titleAttr}">${pinLabel}</div>`,
            iconSize: [26, 26],
            iconAnchor: [13, 13],
            popupAnchor: [0, -13],
          })

          const badgeHtml = assign
            ? `<div style="display: inline-block; background-color: ${assign.color}; color: #ffffff; font-weight: 700; font-size: 11px; padding: 2px 8px; border-radius: 9999px; margin-bottom: 6px;">
                Xe ${assign.vehicle} · Điểm ghé #${assign.stopOrder} / ${assign.totalStops}
              </div>`
            : `<div style="display: inline-block; background-color: #64748b; color: #ffffff; font-weight: 600; font-size: 10px; padding: 1px 6px; border-radius: 9999px; margin-bottom: 4px;">
                Chưa phân tuyến
              </div>`

          const addressLine = p.address
            ? `<div style="color: #475569; margin-bottom: 2px;">📍 ${p.address}</div>`
            : ""
          const snapLine = p.snap_distance_m
            ? `<div style="color: #059669; font-size: 11px; margin-top: 2px;">✓ Ghép đường: ${fmt(p.snap_distance_m)} m</div>`
            : ""

          const popupHtml = `
            <div style="font-family: inherit; font-size: 12px; line-height: 1.5; min-width: 200px;">
              ${badgeHtml}
              <div style="font-weight: 700; font-size: 13px; color: #1e293b; margin-bottom: 3px;">
                ${p.name || `Khách hàng #${p.id}`}
              </div>
              <div style="color: #334155; margin-bottom: 2px;">
                <strong>Nhu cầu:</strong> ${fmt(p.demand)} kiện / kg
              </div>
              ${addressLine}
              <div style="color: #64748b; font-size: 11px;">
                Tọa độ: ${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}
              </div>
              ${snapLine}
            </div>
          `

          L.marker([p.lat, p.lon], {
            icon,
            zIndexOffset: assign ? 500 : 0,
          })
            .addTo(markers)
            .bindPopup(popupHtml)
        }
      }

      if (points.length > 0 && pointsCountRef.current !== points.length) {
        pointsCountRef.current = points.length
        fitPoints()
      }
    }, [points, routes])

    // Render route polylines whenever routes change
    useEffect(() => {
      const lines = linesRef.current
      if (!lines) return
      lines.clearLayers()
      routeLayersRef.current = []

      routes.forEach((r, i) => {
        const color = ROUTE_COLORS[i % ROUTE_COLORS.length]
        const layer = L.geoJSON(r.geometry, {
          style: {
            color,
            weight: 5,
            opacity: 0.85,
            dashArray: r.geometry_source === "straight-line" ? "6 8" : undefined,
          },
        }).addTo(lines)

        const stopCount = Math.max(0, (r.visits?.length || 0) - 2)
        const popupHtml = `
          <div style="font-family: inherit; font-size: 12px; line-height: 1.5; min-width: 180px;">
            <div style="font-weight: 700; font-size: 13px; color: ${color}; margin-bottom: 4px;">
              🚛 Tuyến Xe ${r.vehicle}
            </div>
            <div><strong>Số điểm giao:</strong> ${stopCount} điểm</div>
            <div><strong>Tải trọng:</strong> ${fmt(r.load)} kiện</div>
            <div><strong>Thời gian chạy:</strong> ${fmt(r.duration ? r.duration / 60 : 0)} phút</div>
            <div><strong>Khởi hành:</strong> ${clock(r.departure_time)}</div>
            <div style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed #e2e8f0; font-size: 11px; color: #64748b;">
              Lộ trình: ${r.visits.join(" → ")}
            </div>
          </div>
        `
        layer.bindPopup(popupHtml)

        routeLayersRef.current.push(layer)
      })
    }, [routes])

    const fallbackRoutes = routes.filter((r) => r.geometry_source !== "osrm")

    return (
      <Card size="sm" className="overflow-hidden">
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 pb-2">
          <div className="flex items-center gap-2">
            <CardTitle>Lộ trình trên OpenStreetMap</CardTitle>
            {pendingPick && (
              <Badge variant="default" className="gap-1">
                <CrosshairIcon className="size-3" />
                Chọn vị trí: {pendingPick.label}
              </Badge>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {pendingPick && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={onCancelMapPick}
              >
                <XIcon data-icon="inline-start" />
                Hủy chọn
              </Button>
            )}

            {algorithmNames.length > 0 && (
              <select
                aria-label="Thuật toán hiển thị"
                value={selectedAlgorithm}
                onChange={(e) => onAlgorithmChange(e.target.value)}
                className="h-8 rounded-lg border border-input bg-background px-2.5 text-xs font-medium text-foreground outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
              >
                {algorithmNames.map((name) => (
                  <option key={name} value={name}>
                    {algorithmLabel(name)}
                  </option>
                ))}
              </select>
            )}

            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={fitPoints}
            >
              <Maximize2Icon data-icon="inline-start" />
              Xem toàn bộ điểm
            </Button>
          </div>
        </CardHeader>

        <CardContent className="flex flex-col gap-2">
          <div
            ref={containerRef}
            className="h-[360px] w-full overflow-hidden rounded-lg border border-border sm:h-[460px] xl:h-[520px]"
          />

          {/* Route Legend Strip */}
          {routes.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border/70 bg-muted/30 px-3 py-2 text-xs">
              <span className="font-semibold text-muted-foreground">
                Chú giải bản đồ:
              </span>
              <div className="flex items-center gap-1.5 font-medium">
                <span className="relative flex size-3 items-center justify-center">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex size-2 rounded-full bg-red-600" />
                </span>
                <span>Kho xuất phát (K)</span>
              </div>

              <span className="text-muted-foreground/40">|</span>

              {routes.map((r, i) => {
                const color = ROUTE_COLORS[i % ROUTE_COLORS.length]
                const stopCount = Math.max(0, (r.visits?.length || 0) - 2)
                return (
                  <button
                    key={`route-legend-${r.vehicle}-${i}`}
                    type="button"
                    onClick={() => focusRoute(i)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-background px-2.5 py-1 font-medium transition hover:border-foreground/30 hover:shadow-xs cursor-pointer"
                    title={`Bấm để phóng to vào tuyến Xe ${r.vehicle}`}
                  >
                    <span
                      className="size-2.5 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-foreground">Xe {r.vehicle}</span>
                    <span className="text-[11px] text-muted-foreground">
                      ({stopCount} điểm)
                    </span>
                  </button>
                )
              })}
            </div>
          )}

          {tileWarning && (
            <div
              role="status"
              className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
            >
              <AlertTriangleIcon className="mt-0.5 size-4 shrink-0" />
              <span>{tileWarning}</span>
            </div>
          )}

          {fallbackRoutes.length > 0 && (
            <div
              role="status"
              className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
            >
              <AlertTriangleIcon className="mt-0.5 size-4 shrink-0" />
              <span>
                {fallbackRoutes.length} tuyến dùng đường dự phòng vì OSRM không
                khả dụng. {fallbackRoutes[0].warning || ""}
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }
)
