import { useMemo, useState, type FormEvent } from "react"
import {
  ArchiveRestoreIcon,
  ArrowLeftIcon,
  BanIcon,
  CheckCircle2Icon,
  ChevronDownIcon,
  ChevronUpIcon,
  EyeIcon,
  LayersIcon,
  Loader2Icon,
  MapIcon,
  PrinterIcon,
  RefreshCwIcon,
  RotateCcwIcon,
  RouteIcon,
  SearchIcon,
  SendIcon,
  Trash2Icon,
  TruckIcon,
  UserCogIcon,
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
import { Input } from "@/components/ui/input"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { toast } from "@/components/ui/sonner"
import {
  DispatchManifestModal,
  type ManifestTarget,
} from "@/components/DispatchManifestModal"
import {
  algorithmLabel,
  api,
  clock,
  deliveryStatusLabel,
  driverDisplayName,
  fmt,
  statusLabel,
} from "@/lib/api"
import type {
  AdminJobGroup,
  Driver,
  OptimizationPayload,
  Trip,
  TripStop,
} from "@/types/rapo"

type RouteFilter =
  | "all"
  | "unassigned"
  | "assigned"
  | "in_progress"
  | "completed"
  | "cancelled"

export interface RouteRowItem {
  key: string
  jobId: string
  tripName: string
  dataset: string
  algorithm: string
  routeIndex: number
  vehicle: number
  stopsCount: number
  load: number | null
  duration: number | null
  assignedTrip: Trip | null
  status: RouteFilter
  deletedAt?: string | null
}

export interface JobGroupItem {
  jobId: string
  tripName: string
  dataset: string
  algorithm: string
  departureTime?: number
  latestDeparture?: number
  totalVehicles: number
  totalStopsCount: number
  assignedCount: number
  inProgressCount: number
  completedCount: number
  cancelledCount: number
  unassignedCount: number
  stopsCompleted: number
  stopsTotal: number
  deletedAt?: string | null
  jobDeleted?: boolean
  routes: RouteRowItem[]
}

type ActionResult = { ok: boolean; error?: string }

interface TripsManagementPageProps {
  adminJobs: AdminJobGroup[]
  deletedJobs: AdminJobGroup[]
  activeDrivers: Driver[]
  loading: boolean
  onBackToMap: () => void
  onRefresh: () => Promise<void>
  onOpenTripOnMap: (trip: Trip) => Promise<void>
  onViewRouteOnMap: (jobId: string, routeIndex?: number) => Promise<void>
  onAssignRoute: (
    jobId: string,
    algorithm: string,
    routeIndex: number,
    driverId: number
  ) => Promise<ActionResult>
  onReassignDriver: (
    tripId: string,
    driverId: number,
    row?: RouteRowItem
  ) => Promise<ActionResult>
  onCancelTrip: (tripId: string) => Promise<ActionResult>
  onSoftDeleteRoute: (row: RouteRowItem) => Promise<ActionResult>
  onRestoreRoute: (row: RouteRowItem) => Promise<ActionResult>
  onSoftDeleteJob: (jobId: string) => Promise<ActionResult>
  onRestoreJob: (jobId: string) => Promise<ActionResult>
  onSoftDeleteAll: () => Promise<ActionResult>
  onRestoreAll: () => Promise<ActionResult>
}

function formatDeletedAt(iso?: string | null): string {
  if (!iso) return ""
  const parsed = new Date(iso)
  if (Number.isNaN(parsed.getTime())) return iso
  return parsed.toLocaleString("vi-VN")
}

function buildJobGroups(groups: AdminJobGroup[]): JobGroupItem[] {
  return groups.map((group) => {
    const { job_id: jobId, plan, trips, deleted_at: deletedAt, job_deleted: jobDeleted } = group
    const tripName =
      plan.trip_name ||
      `Chuyến ${plan.instance || plan.dataset || jobId.slice(0, 8)}`
    const dataset = plan.dataset || plan.instance || "—"
    const algorithm = plan.algorithm || "RAPO-2709"
    const summaryList = Array.isArray(plan.routes_summary)
      ? plan.routes_summary
      : []

    const routes: RouteRowItem[] = []
    if (summaryList.length > 0) {
      for (const r of summaryList) {
        const assignedTrip =
          trips.find((t) => Number(t.route_index) === Number(r.route_index)) ||
          null
        const status = (assignedTrip?.status ?? "unassigned") as RouteFilter
        routes.push({
          key: `${jobId}-${r.route_index}`,
          jobId,
          tripName,
          dataset,
          algorithm: assignedTrip?.algorithm || algorithm,
          routeIndex: r.route_index,
          vehicle: r.vehicle,
          stopsCount: assignedTrip?.stops_total ?? r.stops_count,
          load: r.load,
          duration: r.duration,
          assignedTrip,
          status,
          deletedAt: r.deleted_at ?? assignedTrip?.deleted_at ?? deletedAt,
        })
      }
    } else {
      for (const trip of trips) {
        routes.push({
          key: `${jobId}-${trip.route_index}-${trip.id}`,
          jobId,
          tripName,
          dataset,
          algorithm: trip.algorithm,
          routeIndex: trip.route_index,
          vehicle: trip.route_index + 1,
          stopsCount: trip.stops_total,
          load: null,
          duration: null,
          assignedTrip: trip,
          status: trip.status as RouteFilter,
          deletedAt: trip.deleted_at ?? deletedAt,
        })
      }
    }

    const totalVehicles = plan.total_routes || routes.length || trips.length || 0
    const totalStopsCount = routes.reduce(
      (sum, r) => sum + (r.stopsCount || 0),
      0
    )
    const assignedCount = routes.filter((r) => r.status === "assigned").length
    const inProgressCount = routes.filter(
      (r) => r.status === "in_progress"
    ).length
    const completedCount = routes.filter(
      (r) => r.status === "completed"
    ).length
    const cancelledCount = routes.filter(
      (r) => r.status === "cancelled"
    ).length
    const unassignedCount = routes.filter(
      (r) => r.status === "unassigned"
    ).length
    const stopsCompleted = trips.reduce(
      (sum, t) => sum + (t.stops_completed || 0),
      0
    )
    const stopsTotal = trips.reduce(
      (sum, t) => sum + (t.stops_total || 0),
      0
    )

    return {
      jobId,
      tripName,
      dataset,
      algorithm,
      departureTime: plan.departure_time,
      latestDeparture: plan.latest_departure,
      totalVehicles,
      totalStopsCount,
      assignedCount,
      inProgressCount,
      completedCount,
      cancelledCount,
      unassignedCount,
      stopsCompleted,
      stopsTotal,
      deletedAt,
      jobDeleted,
      routes,
    }
  })
}

function statusBadgeVariant(
  status: RouteFilter
): "default" | "secondary" | "destructive" | "outline" {
  if (status === "completed") return "default"
  if (status === "in_progress") return "secondary"
  if (status === "cancelled") return "destructive"
  return "outline"
}

function routeStatusText(status: RouteFilter): string {
  if (status === "unassigned") return "Chưa phân công"
  return statusLabel(status)
}

export function TripsManagementPage({
  adminJobs,
  deletedJobs,
  activeDrivers,
  loading,
  onBackToMap,
  onRefresh,
  onOpenTripOnMap,
  onViewRouteOnMap,
  onAssignRoute,
  onReassignDriver,
  onCancelTrip,
  onSoftDeleteRoute,
  onRestoreRoute,
  onSoftDeleteJob,
  onRestoreJob,
  onSoftDeleteAll,
  onRestoreAll,
}: TripsManagementPageProps) {
  const [viewScope, setViewScope] = useState<"active" | "trash">("active")
  const [searchQuery, setSearchQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<RouteFilter>("all")
  const [feedback, setFeedback] = useState<{
    text: string
    error: boolean
  } | null>(null)
  const [busyKey, setBusyKey] = useState<string | null>(null)

  // Expanded job cards state: set of jobIds
  const [expandedJobs, setExpandedJobs] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {}
    if (adminJobs.length > 0) {
      init[adminJobs[0].job_id] = true
    }
    return init
  })

  // Assign / Reassign Sheet state
  const [assignTarget, setAssignTarget] = useState<RouteRowItem | null>(null)
  const [selectedDriverId, setSelectedDriverId] = useState<string>("")
  const [assignSaving, setAssignSaving] = useState(false)
  const [assignError, setAssignError] = useState<string | null>(null)

  // Trip stops detail Sheet state
  const [detailTrip, setDetailTrip] = useState<Trip | null>(null)
  const [detailStops, setDetailStops] = useState<TripStop[]>([])
  const [detailPointNames, setDetailPointNames] = useState<Map<number, string>>(
    () => new Map()
  )
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  // Manifest print modal target state
  const [manifestTarget, setManifestTarget] = useState<ManifestTarget | null>(
    null
  )

  const activeJobGroups = useMemo(() => buildJobGroups(adminJobs), [adminJobs])
  const trashJobGroups = useMemo(
    () => buildJobGroups(deletedJobs),
    [deletedJobs]
  )
  const jobGroups = viewScope === "trash" ? trashJobGroups : activeJobGroups

  // Overall KPI statistics (always reflects active fleet, plus trash totals for tabs)
  const stats = useMemo(() => {
    let totalRoutes = 0
    let unassigned = 0
    let assigned = 0
    let inProgress = 0
    let completed = 0

    for (const job of activeJobGroups) {
      totalRoutes += job.totalVehicles
      unassigned += job.unassignedCount
      assigned += job.assignedCount
      inProgress += job.inProgressCount
      completed += job.completedCount
    }

    const trashRoutes = trashJobGroups.reduce(
      (sum, job) => sum + job.totalVehicles,
      0
    )

    return {
      totalJobs: activeJobGroups.length,
      totalRoutes,
      unassigned,
      assigned,
      inProgress,
      completed,
      trashJobs: trashJobGroups.length,
      trashRoutes,
    }
  }, [activeJobGroups, trashJobGroups])

  // Filtered job groups according to search & status filter
  const filteredJobs = useMemo(() => {
    const q = searchQuery.trim().toLowerCase()

    return jobGroups
      .map((job) => {
        const matchesJobName =
          job.tripName.toLowerCase().includes(q) ||
          job.jobId.toLowerCase().includes(q) ||
          job.dataset.toLowerCase().includes(q) ||
          algorithmLabel(job.algorithm).toLowerCase().includes(q)

        const matchingRoutes = job.routes.filter((route) => {
          if (statusFilter !== "all" && route.status !== statusFilter) {
            return false
          }
          if (!q) return true
          if (matchesJobName) return true
          const driverText = route.assignedTrip
            ? `${route.assignedTrip.driver_full_name || ""} ${route.assignedTrip.driver_username}`.toLowerCase()
            : ""
          return (
            `xe ${route.vehicle}`.includes(q) ||
            driverText.includes(q) ||
            algorithmLabel(route.algorithm).toLowerCase().includes(q)
          )
        })

        return {
          ...job,
          routes: matchingRoutes,
          hasMatch: matchingRoutes.length > 0 || (matchesJobName && statusFilter === "all"),
        }
      })
      .filter((job) => job.hasMatch)
  }, [jobGroups, searchQuery, statusFilter])

  const toggleJob = (jobId: string) => {
    setExpandedJobs((prev) => ({
      ...prev,
      [jobId]: !prev[jobId],
    }))
  }

  const expandAll = () => {
    const next: Record<string, boolean> = {}
    for (const job of jobGroups) next[job.jobId] = true
    setExpandedJobs(next)
  }

  const collapseAll = () => {
    setExpandedJobs({})
  }

  const openAssignSheet = (row: RouteRowItem) => {
    setAssignTarget(row)
    const currentId = row.assignedTrip?.driver_id
    const defaultId =
      currentId && activeDrivers.some((d) => d.id === currentId)
        ? String(currentId)
        : activeDrivers[0]
          ? String(activeDrivers[0].id)
          : ""
    setSelectedDriverId(defaultId)
    setAssignError(null)
  }

  const handleAssignSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!assignTarget) return
    const driverId = Number(selectedDriverId)
    if (!driverId) {
      setAssignError("Vui lòng chọn một tài xế đang hoạt động")
      return
    }

    setAssignSaving(true)
    setAssignError(null)
    try {
      const res = assignTarget.assignedTrip
        ? await onReassignDriver(
            assignTarget.assignedTrip.id,
            driverId,
            assignTarget
          )
        : await onAssignRoute(
            assignTarget.jobId,
            assignTarget.algorithm,
            assignTarget.routeIndex,
            driverId
          )

      if (res.ok) {
        const picked = activeDrivers.find((d) => d.id === driverId)
        const label = picked
          ? driverDisplayName(picked.username, picked.full_name)
          : `ID #${driverId}`
        const toastMsg = assignTarget.assignedTrip
          ? `Đã chuyển Xe ${assignTarget.vehicle} sang tài xế ${label}`
          : `Đã giao Xe ${assignTarget.vehicle} cho tài xế ${label}`
        toast.success(toastMsg, {
          description: `Chuyến: ${assignTarget.tripName}`,
        })
        setFeedback({
          text: assignTarget.assignedTrip
            ? `Đã chuyển Xe ${assignTarget.vehicle} (${assignTarget.tripName}) sang tài xế ${label}`
            : `Đã giao Xe ${assignTarget.vehicle} (${assignTarget.tripName}) cho tài xế ${label}`,
          error: false,
        })
        setAssignTarget(null)
      } else {
        setAssignError(res.error || "Không thể phân công tài xế")
        toast.error(res.error || "Không thể phân công tài xế")
      }
    } finally {
      setAssignSaving(false)
    }
  }

  const handleCancel = async (row: RouteRowItem) => {
    const trip = row.assignedTrip
    if (!trip) return
    const confirmed = window.confirm(
      `Bạn có chắc muốn huỷ tuyến Xe ${row.vehicle} (${row.tripName}) đang giao cho ${driverDisplayName(
        trip.driver_username,
        trip.driver_full_name
      )}?`
    )
    if (!confirmed) return

    setBusyKey(row.key)
    setFeedback(null)
    try {
      const res = await onCancelTrip(trip.id)
      if (res.ok) {
        toast.info(`Đã huỷ tuyến Xe ${row.vehicle}`, {
          description: `Chuyến: ${row.tripName}`,
        })
        setFeedback({
          text: `Đã huỷ tuyến Xe ${row.vehicle} (${row.tripName})`,
          error: false,
        })
      } else {
        toast.error(res.error || "Không thể huỷ tuyến")
        setFeedback({
          text: res.error || "Không thể huỷ tuyến",
          error: true,
        })
      }
    } finally {
      setBusyKey(null)
    }
  }

  const runBusyAction = async (
    key: string,
    action: () => Promise<ActionResult>,
    successMsg: string,
    description?: string
  ) => {
    setBusyKey(key)
    setFeedback(null)
    try {
      const res = await action()
      if (res.ok) {
        toast.success(successMsg, description ? { description } : undefined)
        setFeedback({ text: successMsg, error: false })
      } else {
        const msg = res.error || "Thao tác không thành công"
        toast.error(msg)
        setFeedback({ text: msg, error: true })
      }
    } finally {
      setBusyKey(null)
    }
  }

  const handleSoftDeleteRoute = async (row: RouteRowItem) => {
    const trip = row.assignedTrip
    const extraWarning =
      trip?.status === "in_progress"
        ? `\nLưu ý: Tuyến này đang ĐANG GIAO cho ${driverDisplayName(
            trip.driver_username,
            trip.driver_full_name
          )}.`
        : ""
    if (
      !window.confirm(
        `Xoá mềm tuyến Xe ${row.vehicle} thuộc chuyến "${row.tripName}"?${extraWarning}\nTuyến sẽ chuyển vào mục Đã xoá mềm và có thể khôi phục bất cứ lúc nào.`
      )
    ) {
      return
    }
    await runBusyAction(
      row.key,
      () => onSoftDeleteRoute(row),
      `Đã xoá mềm tuyến Xe ${row.vehicle}`,
      `Chuyến: ${row.tripName}`
    )
  }

  const handleRestoreRoute = async (row: RouteRowItem) => {
    await runBusyAction(
      row.key,
      () => onRestoreRoute(row),
      `Đã khôi phục tuyến Xe ${row.vehicle}`,
      `Chuyến: ${row.tripName}`
    )
  }

  const handleSoftDeleteJob = async (job: JobGroupItem) => {
    if (
      !window.confirm(
        `Xoá mềm toàn bộ chuyến "${job.tripName}" (${job.totalVehicles} tuyến xe)?\nDữ liệu không bị xoá vĩnh viễn và có thể khôi phục trong mục Đã xoá mềm.`
      )
    ) {
      return
    }
    await runBusyAction(
      `job-${job.jobId}`,
      () => onSoftDeleteJob(job.jobId),
      `Đã xoá mềm chuyến "${job.tripName}"`
    )
  }

  const handleRestoreJob = async (job: JobGroupItem) => {
    await runBusyAction(
      `job-${job.jobId}`,
      () => onRestoreJob(job.jobId),
      `Đã khôi phục chuyến "${job.tripName}"`
    )
  }

  const handleSoftDeleteAll = async () => {
    if (
      !window.confirm(
        `Xoá mềm toàn bộ ${stats.totalJobs} chuyến đang hoạt động (${stats.totalRoutes} tuyến xe)?\nTất cả sẽ chuyển vào mục Đã xoá mềm và có thể khôi phục lại bất cứ lúc nào.`
      )
    ) {
      return
    }
    await runBusyAction(
      "all-jobs",
      onSoftDeleteAll,
      "Đã xoá mềm toàn bộ các chuyến đang hoạt động"
    )
  }

  const handleRestoreAll = async () => {
    if (
      !window.confirm(
        `Khôi phục toàn bộ ${stats.trashJobs} chuyến (${stats.trashRoutes} tuyến xe) từ mục Đã xoá mềm trở lại danh sách hoạt động?`
      )
    ) {
      return
    }
    await runBusyAction(
      "all-jobs",
      onRestoreAll,
      "Đã khôi phục toàn bộ chuyến từ mục Đã xoá mềm"
    )
    setViewScope("active")
  }

  const handleOpenDetail = async (trip: Trip) => {
    setDetailTrip(trip)
    setDetailStops([])
    setDetailPointNames(new Map())
    setDetailError(null)
    setDetailLoading(true)
    try {
      const data = await api<{
        trip: Trip
        stops: TripStop[]
        payload: OptimizationPayload
      }>(`/api/trips/${trip.id}`)
      setDetailTrip(data.trip)
      setDetailStops(data.stops || [])
      const nameMap = new Map<number, string>()
      for (const p of data.payload?.points || []) {
        if (p.name) nameMap.set(Number(p.id), p.name)
      }
      setDetailPointNames(nameMap)
    } catch (err) {
      setDetailError(
        err instanceof Error ? err.message : "Không tải được chi tiết điểm giao"
      )
    } finally {
      setDetailLoading(false)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-4 p-4 sm:p-6">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <RouteIcon className="size-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold tracking-tight sm:text-xl">
              Quản lý các chuyến & tuyến giao hàng
            </h2>
            <p className="text-xs text-muted-foreground sm:text-sm">
              Theo dõi danh sách các chuyến đã tạo. Bấm vào từng chuyến để mở
              toàn bộ danh sách tuyến xe, phân công hoặc điều chuyển tài xế.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg border border-border bg-muted/40 p-0.5">
            <Button
              type="button"
              size="xs"
              variant={viewScope === "active" ? "default" : "ghost"}
              onClick={() => setViewScope("active")}
            >
              Đang hoạt động ({stats.totalJobs})
            </Button>
            <Button
              type="button"
              size="xs"
              variant={viewScope === "trash" ? "default" : "ghost"}
              onClick={() => setViewScope("trash")}
            >
              <Trash2Icon data-icon="inline-start" />
              Đã xoá mềm ({stats.trashRoutes})
            </Button>
          </div>

          {viewScope === "active" ? (
            <Button
              type="button"
              variant="destructive"
              size="sm"
              disabled={loading || Boolean(busyKey) || stats.totalJobs === 0}
              onClick={() => void handleSoftDeleteAll()}
              title="Xoá mềm tất cả chuyến và tuyến đang hoạt động"
            >
              {busyKey === "all-jobs" ? (
                <Loader2Icon
                  data-icon="inline-start"
                  className="animate-spin"
                />
              ) : (
                <Trash2Icon data-icon="inline-start" />
              )}
              Xoá tất cả
            </Button>
          ) : (
            <Button
              type="button"
              variant="default"
              size="sm"
              disabled={loading || Boolean(busyKey) || stats.trashJobs === 0}
              onClick={() => void handleRestoreAll()}
            >
              {busyKey === "all-jobs" ? (
                <Loader2Icon
                  data-icon="inline-start"
                  className="animate-spin"
                />
              ) : (
                <ArchiveRestoreIcon data-icon="inline-start" />
              )}
              Khôi phục tất cả
            </Button>
          )}

          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={loading || Boolean(busyKey)}
            onClick={() => void onRefresh()}
          >
            <RefreshCwIcon
              data-icon="inline-start"
              className={loading ? "animate-spin" : undefined}
            />
            Làm mới
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={onBackToMap}
          >
            <ArrowLeftIcon data-icon="inline-start" />
            Quay lại bản đồ
          </Button>
        </div>
      </div>

      {/* Feedback Banner */}
      {feedback && (
        <div
          role="status"
          className={`flex items-center gap-2 rounded-lg border px-3.5 py-2.5 text-sm ${
            feedback.error
              ? "border-destructive/40 bg-destructive/10 text-destructive"
              : "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
          }`}
        >
          {!feedback.error && <CheckCircle2Icon className="size-4 shrink-0" />}
          <span>{feedback.text}</span>
        </div>
      )}

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Đợt tối ưu</CardDescription>
            <CardTitle className="font-mono text-2xl font-bold">
              {stats.totalJobs}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Tổng tuyến xe</CardDescription>
            <CardTitle className="font-mono text-2xl font-bold">
              {stats.totalRoutes}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Chưa phân công</CardDescription>
            <CardTitle className="font-mono text-2xl font-bold text-amber-600 dark:text-amber-400">
              {stats.unassigned}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">
              Đã giao / Đang giao
            </CardDescription>
            <CardTitle className="font-mono text-2xl font-bold text-blue-600 dark:text-blue-400">
              {stats.assigned} / {stats.inProgress}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card size="sm" className="col-span-2 lg:col-span-1">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Hoàn thành</CardDescription>
            <CardTitle className="font-mono text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              {stats.completed}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Filter and Control Bar */}
      <Card size="sm">
        <CardContent className="flex flex-col gap-3 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-foreground">
                Bộ lọc & Tìm kiếm
              </span>
              <Badge variant="secondary">{filteredJobs.length} chuyến</Badge>
            </div>
            <div className="flex items-center gap-1.5">
              <Button
                type="button"
                variant="ghost"
                size="xs"
                onClick={expandAll}
                className="text-xs"
              >
                <ChevronDownIcon data-icon="inline-start" />
                Mở tất cả
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                onClick={collapseAll}
                className="text-xs"
              >
                <ChevronUpIcon data-icon="inline-start" />
                Thu gọn tất cả
              </Button>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="relative min-w-[220px] flex-1">
              <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Tìm theo tên chuyến, mã job, tài xế, thuật toán…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8"
              />
            </div>

            <div className="flex flex-wrap items-center gap-1">
              {(
                [
                  ["all", "Tất cả"],
                  ["unassigned", "Chưa phân công"],
                  ["assigned", "Đã giao"],
                  ["in_progress", "Đang giao"],
                  ["completed", "Hoàn thành"],
                  ["cancelled", "Đã huỷ"],
                ] as const
              ).map(([key, label]) => (
                <Button
                  key={key}
                  type="button"
                  size="xs"
                  variant={statusFilter === key ? "default" : "outline"}
                  onClick={() => setStatusFilter(key)}
                >
                  {label}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {viewScope === "trash" && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3.5 py-2.5 text-xs text-amber-900 dark:text-amber-200">
          <span>
            Các tuyến và chuyến tại đây đã được <strong>xoá mềm</strong> (được ẩn
            khỏi điều phối và tài xế nhưng vẫn giữ nguyên lịch sử giao hàng). Bạn
            có thể khôi phục lại từng tuyến hoặc cả chuyến bất cứ lúc nào.
          </span>
        </div>
      )}

      {/* Grouped Jobs List: 1 Chuyến = 1 Khối, bấm vào mới hiện các tuyến */}
      <div className="flex flex-col gap-3">
        {filteredJobs.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
              <LayersIcon className="size-8 text-muted-foreground/60" />
              <p className="mt-2 text-sm font-medium">
                {jobGroups.length === 0
                  ? viewScope === "trash"
                    ? "Thùng rác xoá mềm đang trống."
                    : stats.trashJobs > 0
                      ? `Không còn chuyến nào đang hoạt động (${stats.trashJobs} chuyến đang nằm trong mục Đã xoá mềm).`
                      : "Chưa có chuyến giao hàng nào. Hãy quay lại Bản đồ điều phối để tạo và tối ưu chuyến mới."
                  : "Không tìm thấy chuyến nào khớp với bộ lọc hiện tại."}
              </p>
              {viewScope === "active" &&
                jobGroups.length === 0 &&
                stats.trashJobs > 0 && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="mt-3"
                    onClick={() => setViewScope("trash")}
                  >
                    <Trash2Icon data-icon="inline-start" />
                    Mở mục Đã xoá mềm ({stats.trashRoutes} tuyến)
                  </Button>
                )}
            </CardContent>
          </Card>
        ) : (
          filteredJobs.map((job) => {
            const isExpanded = Boolean(
              expandedJobs[job.jobId] ||
                (searchQuery && job.routes.length > 0) ||
                (statusFilter !== "all" && job.routes.length > 0)
            )
            const assignedCount = job.assignedCount + job.inProgressCount + job.completedCount

            return (
              <Card
                key={job.jobId}
                className="overflow-hidden transition-shadow hover:shadow-sm"
              >
                {/* Header của Chuyến: click vào tiêu đề để toggle mở/đóng danh sách tuyến */}
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => toggleJob(job.jobId)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault()
                      toggleJob(job.jobId)
                    }
                  }}
                  className="flex cursor-pointer flex-wrap items-center justify-between gap-3 bg-muted/20 px-4 py-3 transition-colors hover:bg-muted/40 select-none sm:px-5"
                >
                  <div className="flex min-w-0 items-start gap-3">
                    <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <TruckIcon className="size-4" />
                    </div>

                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-bold text-foreground sm:text-base">
                          {job.tripName}
                        </span>
                        <Badge variant="outline" className="font-mono text-xs">
                          {job.totalVehicles} xe
                        </Badge>
                        <Badge variant="secondary" className="text-xs">
                          {algorithmLabel(job.algorithm)}
                        </Badge>
                        {viewScope === "trash" && (
                          <Badge variant="destructive" className="text-xs">
                            {job.jobDeleted ? "Đã xoá cả chuyến" : "Có tuyến đã xoá mềm"}
                          </Badge>
                        )}
                      </div>

                      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                        <span className="font-mono">Mã {job.jobId.slice(0, 8)}</span>
                        <span>•</span>
                        <span>{job.dataset}</span>
                        {job.departureTime != null && (
                          <>
                            <span>•</span>
                            <span>
                              Khung giờ xuất phát: {clock(job.departureTime)}
                              {job.latestDeparture != null ? ` – ${clock(job.latestDeparture)}` : ""}
                            </span>
                          </>
                        )}
                        <span>•</span>
                        <span className="font-medium text-foreground">
                          {assignedCount}/{job.totalVehicles} xe đã giao ·{" "}
                          {job.completedCount}/{job.totalVehicles} hoàn thành
                        </span>
                        {viewScope === "trash" && job.deletedAt && (
                          <>
                            <span>•</span>
                            <span>Xoá lúc {formatDeletedAt(job.deletedAt)}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Thao tác cấp chuyến */}
                  <div
                    className="flex shrink-0 items-center gap-2"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Button
                      type="button"
                      size="sm"
                      disabled={Boolean(busyKey)}
                      onClick={() => void onViewRouteOnMap(job.jobId)}
                      title="Mở toàn bộ các xe của chuyến này trên bản đồ"
                    >
                      <MapIcon data-icon="inline-start" />
                      <span className="hidden sm:inline">Toàn cảnh chuyến</span>
                      <span className="sm:hidden">Bản đồ</span>
                    </Button>

                    {viewScope === "active" ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={Boolean(busyKey)}
                        className="border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
                        onClick={() => void handleSoftDeleteJob(job)}
                        title="Xoá mềm toàn bộ chuyến này"
                      >
                        {busyKey === `job-${job.jobId}` ? (
                          <Loader2Icon
                            data-icon="inline-start"
                            className="animate-spin"
                          />
                        ) : (
                          <Trash2Icon data-icon="inline-start" />
                        )}
                        <span className="hidden sm:inline">Xoá chuyến</span>
                      </Button>
                    ) : (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={Boolean(busyKey)}
                        onClick={() => void handleRestoreJob(job)}
                        title="Khôi phục toàn bộ chuyến này"
                      >
                        {busyKey === `job-${job.jobId}` ? (
                          <Loader2Icon
                            data-icon="inline-start"
                            className="animate-spin"
                          />
                        ) : (
                          <RotateCcwIcon data-icon="inline-start" />
                        )}
                        <span>Khôi phục chuyến</span>
                      </Button>
                    )}

                    <Button
                      type="button"
                      variant="outline"
                      size="icon-sm"
                      onClick={() => toggleJob(job.jobId)}
                      aria-label={isExpanded ? "Thu gọn tuyến xe" : "Xem toàn bộ tuyến xe"}
                    >
                      {isExpanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
                    </Button>
                  </div>
                </div>

                {/* Danh sách các tuyến xe của Chuyến: Chỉ hiện khi bấm vào */}
                {isExpanded && (
                  <CardContent className="border-t border-border p-0">
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse text-left text-xs">
                        <thead>
                          <tr className="border-b border-border bg-muted/40 text-muted-foreground">
                            <th className="px-4 py-2 font-medium whitespace-nowrap sm:px-5">
                              Tuyến xe
                            </th>
                            <th className="px-3 py-2 font-medium whitespace-nowrap">
                              Thông số tuyến
                            </th>
                            <th className="px-3 py-2 font-medium">Tài xế phụ trách</th>
                            <th className="px-3 py-2 font-medium whitespace-nowrap">
                              Trạng thái
                            </th>
                            <th className="px-3 py-2 font-medium whitespace-nowrap">
                              Tiến độ giao
                            </th>
                            <th className="px-4 py-2 text-right font-medium whitespace-nowrap sm:px-5">
                              Thao tác
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {job.routes.length === 0 ? (
                            <tr>
                              <td
                                colSpan={6}
                                className="px-4 py-6 text-center text-muted-foreground"
                              >
                                Không có tuyến xe nào thỏa điều kiện lọc trong chuyến này.
                              </td>
                            </tr>
                          ) : (
                            job.routes.map((row) => {
                              const trip = row.assignedTrip
                              const isRowBusy = busyKey === row.key
                              const canAssignOrReassign =
                                row.stopsCount > 0 &&
                                row.status !== "completed" &&
                                row.status !== "cancelled"
                              const canCancel =
                                Boolean(trip) &&
                                (row.status === "assigned" ||
                                  row.status === "in_progress")
                              const progressPct =
                                trip && trip.stops_total > 0
                                  ? Math.round(
                                      (trip.stops_completed / trip.stops_total) * 100
                                    )
                                  : 0

                              return (
                                <tr
                                  key={row.key}
                                  className="transition-colors hover:bg-muted/20"
                                >
                                  <td className="px-4 py-2.5 sm:px-5">
                                    <div className="flex items-center gap-2">
                                      <span className="font-semibold text-foreground">
                                        Xe {row.vehicle}
                                      </span>
                                      <Badge variant="outline" className="font-mono text-[10px]">
                                        Tuyến #{row.routeIndex + 1}
                                      </Badge>
                                    </div>
                                    <div className="font-mono text-[11px] text-muted-foreground">
                                      {algorithmLabel(row.algorithm)}
                                    </div>
                                  </td>

                                  <td className="px-3 py-2.5 whitespace-nowrap">
                                    <div className="font-medium text-foreground">
                                      {row.stopsCount} điểm giao
                                    </div>
                                    <div className="font-mono text-[11px] text-muted-foreground">
                                      {row.load != null ? `Tải ${fmt(row.load)}` : "—"}
                                      {row.duration != null
                                        ? ` · ${clock(row.duration)}`
                                        : ""}
                                    </div>
                                  </td>

                                  <td className="px-3 py-2.5">
                                    {trip ? (
                                      <div>
                                        <div className="font-medium text-foreground">
                                          {trip.driver_full_name?.trim() ||
                                            trip.driver_username}
                                        </div>
                                        <div className="font-mono text-[11px] text-muted-foreground">
                                          @{trip.driver_username}
                                        </div>
                                      </div>
                                    ) : (
                                      <span className="text-muted-foreground italic">
                                        Chưa gán tài xế
                                      </span>
                                    )}
                                  </td>

                                  <td className="px-3 py-2.5 whitespace-nowrap">
                                    <Badge variant={statusBadgeVariant(row.status)}>
                                      {routeStatusText(row.status)}
                                    </Badge>
                                  </td>

                                  <td className="px-3 py-2.5 whitespace-nowrap">
                                    {trip ? (
                                      <div className="flex w-28 flex-col gap-1">
                                        <div className="flex items-center justify-between font-mono text-[11px]">
                                          <span>
                                            {trip.stops_completed}/{trip.stops_total} điểm
                                          </span>
                                          <span className="text-muted-foreground">
                                            {progressPct}%
                                          </span>
                                        </div>
                                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                                          <div
                                            className="h-full rounded-full bg-primary transition-all"
                                            style={{ width: `${progressPct}%` }}
                                          />
                                        </div>
                                      </div>
                                    ) : (
                                      <span className="text-muted-foreground">—</span>
                                    )}
                                  </td>

                                  <td className="px-4 py-2.5 text-right whitespace-nowrap sm:px-5">
                                    <div className="flex items-center justify-end gap-1.5">
                                      {viewScope === "active" ? (
                                        <>
                                          <Button
                                            type="button"
                                            variant="outline"
                                            size="xs"
                                            disabled={isRowBusy || Boolean(busyKey)}
                                            onClick={() =>
                                              trip
                                                ? void onOpenTripOnMap(trip)
                                                : void onViewRouteOnMap(
                                                    row.jobId,
                                                    row.routeIndex
                                                  )
                                            }
                                          >
                                            <MapIcon data-icon="inline-start" />
                                            Bản đồ
                                          </Button>

                                          <Button
                                            type="button"
                                            variant="outline"
                                            size="xs"
                                            disabled={isRowBusy || Boolean(busyKey)}
                                            title="In lệnh điều phối & phiếu bàn giao khổ A4"
                                            onClick={() =>
                                              setManifestTarget({
                                                jobId: row.jobId,
                                                routeIndex: row.routeIndex,
                                                vehicle: row.vehicle,
                                                algorithm: row.algorithm,
                                                tripName: row.tripName,
                                                dataset: row.dataset,
                                                trip: trip,
                                              })
                                            }
                                          >
                                            <PrinterIcon data-icon="inline-start" />
                                            In lệnh
                                          </Button>

                                          {trip && (
                                            <Button
                                              type="button"
                                              variant="ghost"
                                              size="xs"
                                              disabled={isRowBusy || Boolean(busyKey)}
                                              onClick={() => void handleOpenDetail(trip)}
                                            >
                                              <EyeIcon data-icon="inline-start" />
                                              Điểm giao
                                            </Button>
                                          )}

                                          {canAssignOrReassign && (
                                            <Button
                                              type="button"
                                              variant={trip ? "secondary" : "default"}
                                              size="xs"
                                              disabled={
                                                isRowBusy ||
                                                Boolean(busyKey) ||
                                                activeDrivers.length === 0
                                              }
                                              onClick={() => openAssignSheet(row)}
                                            >
                                              {trip ? (
                                                <UserCogIcon data-icon="inline-start" />
                                              ) : (
                                                <SendIcon data-icon="inline-start" />
                                              )}
                                              {trip ? "Đổi tài xế" : "Giao Driver"}
                                            </Button>
                                          )}

                                          {canCancel && (
                                            <Button
                                              type="button"
                                              variant="outline"
                                              size="xs"
                                              disabled={isRowBusy || Boolean(busyKey)}
                                              onClick={() => void handleCancel(row)}
                                            >
                                              <BanIcon data-icon="inline-start" />
                                              Huỷ
                                            </Button>
                                          )}

                                          <Button
                                            type="button"
                                            variant="destructive"
                                            size="xs"
                                            disabled={isRowBusy || Boolean(busyKey)}
                                            title="Xoá mềm tuyến này khỏi danh sách hoạt động"
                                            onClick={() =>
                                              void handleSoftDeleteRoute(row)
                                            }
                                          >
                                            {isRowBusy ? (
                                              <Loader2Icon
                                                data-icon="inline-start"
                                                className="animate-spin"
                                              />
                                            ) : (
                                              <Trash2Icon data-icon="inline-start" />
                                            )}
                                            Xoá mềm
                                          </Button>
                                        </>
                                      ) : (
                                        <>
                                          <Button
                                            type="button"
                                            variant="outline"
                                            size="xs"
                                            disabled={isRowBusy || Boolean(busyKey)}
                                            onClick={() =>
                                              trip
                                                ? void onOpenTripOnMap(trip)
                                                : void onViewRouteOnMap(
                                                    row.jobId,
                                                    row.routeIndex
                                                  )
                                            }
                                          >
                                            <MapIcon data-icon="inline-start" />
                                            Bản đồ
                                          </Button>

                                          {trip && (
                                            <Button
                                              type="button"
                                              variant="ghost"
                                              size="xs"
                                              disabled={isRowBusy || Boolean(busyKey)}
                                              onClick={() => void handleOpenDetail(trip)}
                                            >
                                              <EyeIcon data-icon="inline-start" />
                                              Điểm giao
                                            </Button>
                                          )}

                                          <Button
                                            type="button"
                                            variant="default"
                                            size="xs"
                                            disabled={isRowBusy || Boolean(busyKey)}
                                            onClick={() => void handleRestoreRoute(row)}
                                          >
                                            {isRowBusy ? (
                                              <Loader2Icon
                                                data-icon="inline-start"
                                                className="animate-spin"
                                              />
                                            ) : (
                                              <RotateCcwIcon data-icon="inline-start" />
                                            )}
                                            Khôi phục
                                          </Button>
                                        </>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              )
                            })
                          )}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                )}
              </Card>
            )
          })
        )}
      </div>

      {/* Assign / Reassign Driver Sheet */}
      <Sheet
        open={Boolean(assignTarget)}
        onOpenChange={(open) => {
          if (!open) setAssignTarget(null)
        }}
      >
        <SheetContent side="right" className="w-[92vw] max-w-md">
          <SheetHeader>
            <SheetTitle>
              {assignTarget?.assignedTrip
                ? "Điều chuyển tài xế cho tuyến"
                : "Phân công tài xế cho tuyến"}
            </SheetTitle>
            <SheetDescription>
              {assignTarget
                ? `${assignTarget.tripName} · Xe ${assignTarget.vehicle} (${assignTarget.stopsCount} điểm giao)`
                : ""}
            </SheetDescription>
          </SheetHeader>

          {assignTarget && (
            <form
              onSubmit={handleAssignSubmit}
              className="flex flex-1 flex-col justify-between gap-4 px-4 pb-4"
            >
              <div className="flex flex-col gap-3">
                {assignTarget.assignedTrip && (
                  <div className="rounded-lg border border-border bg-muted/40 p-3 text-xs">
                    <div className="text-muted-foreground">
                      Tài xế hiện tại:
                    </div>
                    <div className="mt-0.5 font-semibold text-foreground">
                      {driverDisplayName(
                        assignTarget.assignedTrip.driver_username,
                        assignTarget.assignedTrip.driver_full_name
                      )}
                    </div>
                  </div>
                )}

                <div className="flex flex-col gap-1.5">
                  <label
                    htmlFor="assign-route-driver"
                    className="text-xs font-medium"
                  >
                    Chọn tài xế đang hoạt động
                  </label>
                  <select
                    id="assign-route-driver"
                    value={selectedDriverId}
                    disabled={assignSaving || activeDrivers.length === 0}
                    onChange={(e) => setSelectedDriverId(e.target.value)}
                    className="h-9 w-full rounded-lg border border-input bg-background px-2.5 text-sm text-foreground outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
                  >
                    {activeDrivers.map((d) => (
                      <option key={d.id} value={d.id}>
                        {driverDisplayName(d.username, d.full_name)}
                      </option>
                    ))}
                  </select>
                </div>

                {assignError && (
                  <div
                    role="alert"
                    className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive"
                  >
                    {assignError}
                  </div>
                )}
              </div>

              <SheetFooter className="flex flex-row justify-end gap-2 p-0">
                <Button
                  type="button"
                  variant="outline"
                  disabled={assignSaving}
                  onClick={() => setAssignTarget(null)}
                >
                  Hủy
                </Button>
                <Button
                  type="submit"
                  disabled={assignSaving || !selectedDriverId}
                >
                  {assignSaving && (
                    <Loader2Icon
                      data-icon="inline-start"
                      className="animate-spin"
                    />
                  )}
                  {assignTarget.assignedTrip
                    ? "Xác nhận đổi tài xế"
                    : "Giao tuyến"}
                </Button>
              </SheetFooter>
            </form>
          )}
        </SheetContent>
      </Sheet>

      {/* Stops Detail Sheet */}
      <Sheet
        open={Boolean(detailTrip)}
        onOpenChange={(open) => {
          if (!open) setDetailTrip(null)
        }}
      >
        <SheetContent side="right" className="w-[92vw] max-w-lg">
          <SheetHeader>
            <SheetTitle>Chi tiết tiến độ các điểm giao</SheetTitle>
            <SheetDescription>
              {detailTrip
                ? `Xe ${detailTrip.route_index + 1} · Driver: ${driverDisplayName(
                    detailTrip.driver_username,
                    detailTrip.driver_full_name
                  )} · ${statusLabel(detailTrip.status)}`
                : ""}
            </SheetDescription>
          </SheetHeader>

          <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 pb-4">
            {detailLoading ? (
              <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
                <Loader2Icon className="size-4 animate-spin" />
                Đang tải danh sách điểm giao…
              </div>
            ) : detailError ? (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {detailError}
              </div>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full border-collapse text-left text-xs">
                  <thead>
                    <tr className="border-b border-border bg-muted/50 text-muted-foreground">
                      <th className="px-2.5 py-2 font-medium">STT</th>
                      <th className="px-2.5 py-2 font-medium">Điểm giao</th>
                      <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                        Trạng thái
                      </th>
                      <th className="px-2.5 py-2 font-medium whitespace-nowrap">
                        Hoàn thành lúc
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {detailStops.map((stop) => (
                      <tr key={stop.stop_index}>
                        <td className="px-2.5 py-2 font-mono">
                          {stop.stop_index + 1}
                        </td>
                        <td className="px-2.5 py-2 font-medium">
                          {detailPointNames.get(Number(stop.customer_id)) ||
                            `Khách hàng ${stop.customer_id}`}
                        </td>
                        <td className="px-2.5 py-2 whitespace-nowrap">
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
                        <td className="px-2.5 py-2 font-mono text-muted-foreground whitespace-nowrap">
                          {stop.completed_at || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {detailTrip && (
              <div className="mt-auto flex flex-wrap justify-end gap-2 pt-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setManifestTarget({
                      jobId: detailTrip.job_id,
                      routeIndex: detailTrip.route_index,
                      vehicle: detailTrip.route_index + 1,
                      algorithm: detailTrip.algorithm,
                      tripName:
                        detailTrip.plan?.trip_name ||
                        `Chuyến ${detailTrip.job_id.slice(0, 8)}`,
                      dataset: detailTrip.plan?.dataset,
                      trip: detailTrip,
                    })
                  }}
                >
                  <PrinterIcon data-icon="inline-start" />
                  In lệnh giao hàng
                </Button>
                <Button
                  type="button"
                  size="sm"
                  onClick={() => {
                    const target = detailTrip
                    setDetailTrip(null)
                    void onOpenTripOnMap(target)
                  }}
                >
                  <MapIcon data-icon="inline-start" />
                  Mở tuyến trên bản đồ
                </Button>
              </div>
            )}
          </div>
        </SheetContent>
      </Sheet>

      {/* A4 Dispatch Manifest Modal */}
      <DispatchManifestModal
        open={Boolean(manifestTarget)}
        onClose={() => setManifestTarget(null)}
        target={manifestTarget}
      />
    </div>
  )
}
