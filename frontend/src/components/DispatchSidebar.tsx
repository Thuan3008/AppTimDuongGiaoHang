// import { useState, type FormEvent } from "react"
import {
  ArrowLeftIcon,
  CheckCircle2Icon,
  ChevronRightIcon,
  Loader2Icon,
  MapIcon,
  PlayIcon,
  RouteIcon,
  // UserPlusIcon,
  // UsersIcon,
} from "lucide-react"

import { ManualPointsEditor } from "@/components/ManualPointsEditor"
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
  algorithmLabel,
  clock,
  driverDisplayName,
  fmt,
  statusLabel,
} from "@/lib/api"
import type {
  AdminJobGroup,
  CustomPointDraft,
  DatasetEntry,
  DepotDraft,
  Driver,
  GeocodeCandidate,
  MapPickTarget,
  Trip,
  User,
} from "@/types/rapo"

export interface StatusMessage {
  text: string
  error: boolean
}

interface DispatchSidebarProps {
  user: User
  busy: boolean
  drivers: Driver[]
  catalog: DatasetEntry[]
  tripName: string
  dataset: string
  manualMode: boolean
  manualVehicles: string
  manualCapacity: string
  manualServiceMinutes: string
  depot: DepotDraft
  customPoints: CustomPointDraft[]
  pendingPick: MapPickTarget | null
  availableAlgorithms: string[]
  selectedAlgorithms: string[]
  populationSize: string
  budget: string
  seed: string
  numSeeds: string
  earliestDeparture: string
  latestDeparture: string
  allTrips: Trip[]
  adminJobs: AdminJobGroup[]
  currentJobId: string | null
  currentTrip: Trip | null
  statusMsg: StatusMessage
  onTripNameChange: (value: string) => void
  onDatasetChange: (value: string) => void
  onManualModeChange: (enabled: boolean) => void
  onManualVehiclesChange: (value: string) => void
  onManualCapacityChange: (value: string) => void
  onManualServiceMinutesChange: (value: string) => void
  onDepotChange: (patch: Partial<DepotDraft>) => void
  onDepotAddressInput: (address: string) => void
  onDepotCoordInput: (field: "lat" | "lon", value: string) => void
  onResolveDepot: () => void
  onSelectDepotCandidate: (candidate: GeocodeCandidate) => void
  onPickDepotOnMap: () => void
  onAddPoint: () => void
  onRemovePoint: (uid: string) => void
  onPointChange: (uid: string, patch: Partial<CustomPointDraft>) => void
  onPointAddressInput: (uid: string, address: string) => void
  onPointCoordInput: (uid: string, field: "lat" | "lon", value: string) => void
  onResolvePoint: (uid: string) => void
  onSelectPointCandidate: (uid: string, candidate: GeocodeCandidate) => void
  onPickPointOnMap: (uid: string, index: number) => void
  onPreviewPoints: () => void
  onToggleAlgorithm: (name: string) => void
  onPopulationChange: (value: string) => void
  onBudgetChange: (value: string) => void
  onSeedChange: (value: string) => void
  onNumSeedsChange: (value: string) => void
  onEarliestDepartureChange: (value: string) => void
  onLatestDepartureChange: (value: string) => void
  onStartOptimization: () => void
  onCreateDriver: (
    fullName: string,
    username: string,
    password: string
  ) => Promise<boolean>
  onOpenDriversPage: () => void
  onOpenTripsPage: () => void
  onOpenTrip: (trip: Trip) => void
  onStartTrip: (trip: Trip) => void
  onViewAllJobRoutes: (jobId: string, focusIndex?: number) => void
}

function tripStatusBadgeVariant(
  status: string
): "default" | "secondary" | "destructive" | "outline" {
  if (status === "completed") return "default"
  if (status === "in_progress") return "secondary"
  if (status === "cancelled") return "destructive"
  return "outline"
}

export function DispatchSidebar({
  user,
  busy,
  drivers,
  catalog,
  tripName,
  dataset,
  manualMode,
  manualVehicles,
  manualCapacity,
  manualServiceMinutes,
  depot,
  customPoints,
  pendingPick,
  availableAlgorithms,
  selectedAlgorithms,
  populationSize,
  budget,
  seed,
  numSeeds,
  earliestDeparture,
  latestDeparture,
  allTrips,
  adminJobs,
  currentJobId,
  currentTrip,
  statusMsg,
  onTripNameChange,
  onDatasetChange,
  onManualModeChange,
  onManualVehiclesChange,
  onManualCapacityChange,
  onManualServiceMinutesChange,
  onDepotChange,
  onDepotAddressInput,
  onDepotCoordInput,
  onResolveDepot,
  onSelectDepotCandidate,
  onPickDepotOnMap,
  onAddPoint,
  onRemovePoint,
  onPointChange,
  onPointAddressInput,
  onPointCoordInput,
  onResolvePoint,
  onSelectPointCandidate,
  onPickPointOnMap,
  onPreviewPoints,
  onToggleAlgorithm,
  onPopulationChange,
  onBudgetChange,
  onSeedChange,
  onNumSeedsChange,
  onEarliestDepartureChange,
  onLatestDepartureChange,
  onStartOptimization,
  onOpenTripsPage,
  onOpenTrip,
  onStartTrip,
  onViewAllJobRoutes,
}: DispatchSidebarProps) {
  const isAdmin = user.role === "admin"
  const seedNum = Number(seed) || 0
  const numSeedsNum = Math.max(1, Number(numSeeds) || 1)
  const seedPlanText =
    numSeedsNum > 1
      ? `Chạy ${numSeedsNum} seed liên tiếp: ${seedNum} → ${seedNum + numSeedsNum - 1}. Hiển thị nghiệm tốt nhất.`
      : `Chạy 1 seed (seed=${seedNum}).`

  // Build grouped list matching web_map/static/js/trips.js
  const groups: AdminJobGroup[] = (() => {
    if (isAdmin && adminJobs.length > 0) {
      return adminJobs
    }
    const grouped = new Map<string, Trip[]>()
    for (const trip of allTrips) {
      const list = grouped.get(trip.job_id) ?? []
      list.push(trip)
      grouped.set(trip.job_id, list)
    }
    return [...grouped.entries()].map(([job_id, trips]) => ({
      job_id,
      plan: trips[0]?.plan || {},
      trips,
    }))
  })()

  const isSuccessStatus =
    !statusMsg.error && statusMsg.text.includes("Hoàn tất")

  return (
    <div className="flex flex-col gap-4 p-3 sm:p-4">
      {/* Status Banner */}
      <div
        role="status"
        aria-live="polite"
        className={`rounded-lg border px-3 py-2 text-xs font-medium leading-relaxed ${
          statusMsg.error
            ? "border-destructive/40 bg-destructive/10 text-destructive"
            : isSuccessStatus
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
              : "border-border bg-muted/50 text-foreground"
        }`}
      >
        {statusMsg.text}
      </div>

      {/* Admin Controls */}
      {isAdmin ? (
        <>
          <Card size="sm">
            <CardHeader>
              <CardTitle>Tạo chuyến giao hàng</CardTitle>
              <CardDescription>
                Cấu hình bộ dữ liệu, thuật toán và tham số tối ưu TDVRP.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <label htmlFor="trip-name" className="text-xs font-medium">
                  Tên chuyến
                </label>
                <Input
                  id="trip-name"
                  type="text"
                  maxLength={100}
                  placeholder="Ví dụ: Giao hàng ca sáng"
                  value={tripName}
                  onChange={(e) => onTripNameChange(e.target.value)}
                  disabled={busy}
                />
              </div>

              <div className="flex flex-col gap-1">
                <label htmlFor="dataset" className="text-xs font-medium">
                  Dataset / nền giao thông
                </label>
                <select
                  id="dataset"
                  value={dataset}
                  disabled={busy}
                  onChange={(e) => onDatasetChange(e.target.value)}
                  className="h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm text-foreground outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {catalog.map((ds) => {
                    const isDisabled =
                      ds.missing.length > 0 ||
                      (manualMode && !ds.id.startsWith("hcmc"))
                    return (
                      <option key={ds.id} value={ds.id} disabled={isDisabled}>
                        {ds.name}
                        {ds.missing.length ? " · thiếu sidecar" : ""}
                      </option>
                    )
                  })}
                </select>
              </div>

              <label className="flex cursor-pointer items-center gap-2 text-xs font-medium select-none">
                <input
                  id="manual-toggle"
                  type="checkbox"
                  checked={manualMode}
                  disabled={busy}
                  onChange={(e) => onManualModeChange(e.target.checked)}
                  className="size-4 rounded border-input accent-primary"
                />
                Tự nhập kho và các điểm giao
              </label>

              {manualMode && (
                <ManualPointsEditor
                  disabled={busy}
                  manualVehicles={manualVehicles}
                  manualCapacity={manualCapacity}
                  manualServiceMinutes={manualServiceMinutes}
                  depot={depot}
                  points={customPoints}
                  pendingPick={pendingPick}
                  onVehiclesChange={onManualVehiclesChange}
                  onCapacityChange={onManualCapacityChange}
                  onServiceMinutesChange={onManualServiceMinutesChange}
                  onDepotChange={onDepotChange}
                  onDepotAddressInput={onDepotAddressInput}
                  onDepotCoordInput={onDepotCoordInput}
                  onResolveDepot={onResolveDepot}
                  onSelectDepotCandidate={onSelectDepotCandidate}
                  onPickDepotOnMap={onPickDepotOnMap}
                  onAddPoint={onAddPoint}
                  onRemovePoint={onRemovePoint}
                  onPointChange={onPointChange}
                  onPointAddressInput={onPointAddressInput}
                  onPointCoordInput={onPointCoordInput}
                  onResolvePoint={onResolvePoint}
                  onSelectPointCandidate={onSelectPointCandidate}
                  onPickPointOnMap={onPickPointOnMap}
                  onPreviewPoints={onPreviewPoints}
                />
              )}

              {/* Algorithms */}
              <div className="flex flex-col gap-1.5">
                <span className="text-xs font-medium">Thuật toán tối ưu</span>
                <div className="flex flex-wrap gap-2">
                  {availableAlgorithms.map((name) => {
                    const checked = selectedAlgorithms.includes(name)
                    return (
                      <label
                        key={name}
                        className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors select-none ${
                          checked
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border bg-background text-muted-foreground hover:bg-muted"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={busy}
                          onChange={() => onToggleAlgorithm(name)}
                          className="size-3.5 accent-primary"
                        />
                        {algorithmLabel(name)}
                      </label>
                    )
                  })}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="flex flex-col gap-1">
                  <label htmlFor="population" className="text-xs font-medium">
                    Quần thể (P)
                  </label>
                  <Input
                    id="population"
                    type="number"
                    min={2}
                    max={50}
                    value={populationSize}
                    onChange={(e) => onPopulationChange(e.target.value)}
                    disabled={busy}
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label htmlFor="budget" className="text-xs font-medium">
                    NFE / seed
                  </label>
                  <Input
                    id="budget"
                    type="number"
                    min={2}
                    max={10000}
                    value={budget}
                    onChange={(e) => onBudgetChange(e.target.value)}
                    disabled={busy}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="flex flex-col gap-1">
                  <label htmlFor="seed" className="text-xs font-medium">
                    Seed bắt đầu
                  </label>
                  <Input
                    id="seed"
                    type="number"
                    min={0}
                    max={999999}
                    value={seed}
                    onChange={(e) => onSeedChange(e.target.value)}
                    disabled={busy}
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label htmlFor="num-seeds" className="text-xs font-medium">
                    Số seed
                  </label>
                  <Input
                    id="num-seeds"
                    type="number"
                    min={1}
                    max={100}
                    value={numSeeds}
                    onChange={(e) => onNumSeedsChange(e.target.value)}
                    disabled={busy}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div className="flex flex-col gap-1">
                  <label
                    htmlFor="earliest-departure"
                    className="text-xs font-medium"
                  >
                    Giờ xuất phát sớm nhất
                  </label>
                  <Input
                    id="earliest-departure"
                    type="time"
                    value={earliestDeparture}
                    onChange={(e) => onEarliestDepartureChange(e.target.value)}
                    disabled={busy}
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label
                    htmlFor="latest-departure"
                    className="text-xs font-medium"
                  >
                    Giờ xuất phát muộn nhất
                  </label>
                  <Input
                    id="latest-departure"
                    type="time"
                    value={latestDeparture}
                    onChange={(e) => onLatestDepartureChange(e.target.value)}
                    disabled={busy}
                  />
                </div>
              </div>

              <p className="text-[11px] text-muted-foreground">{seedPlanText}</p>

              <Button
                type="button"
                disabled={busy || !dataset || drivers.length === 0}
                onClick={onStartOptimization}
                className="w-full"
              >
                {busy ? (
                  <Loader2Icon
                    data-icon="inline-start"
                    className="animate-spin"
                  />
                ) : (
                  <PlayIcon data-icon="inline-start" />
                )}
                {busy ? "Đang tối ưu…" : "Tạo và tối ưu chuyến"}
              </Button>
            </CardContent>
          </Card>

          {/* Add Driver Card */}
          {/* <Card size="sm">
            <CardHeader>
              <div className="flex items-center justify-between gap-2">
                <div>
                  <CardTitle>Thêm tài xế mới</CardTitle>
                  <CardDescription>
                    Hiện có {drivers.length} tài khoản Driver hoạt động.
                  </CardDescription>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  onClick={onOpenDriversPage}
                >
                  <UsersIcon data-icon="inline-start" />
                  Quản lý
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <form
                onSubmit={handleDriverSubmit}
                className="flex flex-col gap-2.5"
              >
                <div className="flex flex-col gap-1">
                  <label
                    htmlFor="new-driver-fullname"
                    className="text-xs font-medium"
                  >
                    Tên tài xế
                  </label>
                  <Input
                    id="new-driver-fullname"
                    type="text"
                    maxLength={100}
                    placeholder="Ví dụ: Nguyễn Văn A"
                    value={newDriverFullName}
                    onChange={(e) => setNewDriverFullName(e.target.value)}
                    disabled={creatingDriver}
                    required
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label
                    htmlFor="new-driver-username"
                    className="text-xs font-medium"
                  >
                    Tài khoản
                  </label>
                  <Input
                    id="new-driver-username"
                    type="text"
                    placeholder="Tên đăng nhập"
                    value={newDriverUsername}
                    onChange={(e) => setNewDriverUsername(e.target.value)}
                    disabled={creatingDriver}
                    required
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label
                    htmlFor="new-driver-password"
                    className="text-xs font-medium"
                  >
                    Mật khẩu
                  </label>
                  <Input
                    id="new-driver-password"
                    type="password"
                    minLength={8}
                    placeholder="Tối thiểu 8 ký tự"
                    value={newDriverPassword}
                    onChange={(e) => setNewDriverPassword(e.target.value)}
                    disabled={creatingDriver}
                    required
                  />
                </div>
                <Button
                  type="submit"
                  variant="secondary"
                  size="sm"
                  disabled={creatingDriver}
                  className="w-full"
                >
                  {creatingDriver ? (
                    <Loader2Icon
                      data-icon="inline-start"
                      className="animate-spin"
                    />
                  ) : (
                    <UserPlusIcon data-icon="inline-start" />
                  )}
                  Tạo tài khoản Driver
                </Button>
              </form>
            </CardContent>
          </Card> */}
        </>
      ) : (
        <Card size="sm">
          <CardHeader>
            <CardTitle>Chế độ Tài xế</CardTitle>
            <CardDescription>
              Bạn đang xem danh sách các chuyến được giao riêng cho tài khoản của
              bạn.
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      {/* Trips / Jobs List */}
      <Card size="sm">
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <CardTitle>
                {isAdmin ? "Các chuyến đã tạo" : "Chuyến được giao"}
              </CardTitle>
              <Badge variant="secondary">{groups.length}</Badge>
            </div>
            {isAdmin && (
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={onOpenTripsPage}
              >
                <RouteIcon data-icon="inline-start" />
                Quản lý
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-2.5">
          {isAdmin && currentTrip && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onViewAllJobRoutes(currentTrip.job_id)}
              className="w-full"
            >
              <ArrowLeftIcon data-icon="inline-start" />
              Quay lại xem toàn bộ các xe
            </Button>
          )}

          {groups.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              {isAdmin
                ? "Chưa có chuyến tối ưu nào được tạo."
                : "Chưa có chuyến được giao cho tài khoản này."}
            </p>
          ) : (
            groups.map(({ job_id: jobId, plan, trips }) => {
              const completed = trips.filter(
                (t) => t.status === "completed"
              ).length
              const totalRoutes = plan.total_routes || trips.length || 0
              const isActiveGroup =
                (currentTrip && currentTrip.job_id === jobId) ||
                currentJobId === jobId

              const metaText =
                totalRoutes > 0 && totalRoutes !== trips.length
                  ? `Mã ${jobId.slice(0, 8)} · ${trips.length}/${totalRoutes} xe đã giao · ${completed}/${trips.length} hoàn thành`
                  : `Mã ${jobId.slice(0, 8)} · ${trips.length} xe · ${completed}/${trips.length} hoàn thành`

              return (
                <details
                  key={jobId}
                  open={Boolean(isActiveGroup)}
                  className="group rounded-lg border border-border bg-muted/20"
                >
                  <summary className="flex cursor-pointer list-none items-start gap-2 p-2.5 select-none">
                    <ChevronRightIcon className="mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-90" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-semibold text-foreground">
                        {plan.trip_name ||
                          `Chuyến ${plan.instance || plan.dataset || jobId.slice(0, 8)}`}
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        {metaText}
                      </div>
                    </div>
                  </summary>

                  <div className="flex flex-col gap-2 border-t border-border p-2.5">
                    {isAdmin && (
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => onViewAllJobRoutes(jobId)}
                        className="w-full"
                      >
                        <MapIcon data-icon="inline-start" />
                        Xem toàn bộ chuyến ({totalRoutes || trips.length} xe)
                      </Button>
                    )}

                    {isAdmin &&
                    Array.isArray(plan.routes_summary) &&
                    plan.routes_summary.length > 0
                      ? plan.routes_summary.map((r) => {
                          const assignedTrip = trips.find(
                            (t) => Number(t.route_index) === Number(r.route_index)
                          )

                          if (assignedTrip) {
                            const selected = currentTrip?.id === assignedTrip.id
                            return (
                              <div
                                key={r.route_index}
                                className={`flex flex-col gap-1.5 rounded-md border p-2 ${
                                  selected
                                    ? "border-primary bg-primary/5"
                                    : "border-border bg-card"
                                }`}
                              >
                                <div className="flex items-center justify-between gap-1.5">
                                  <span className="text-xs font-semibold">
                                    Xe {r.vehicle} ·{" "}
                                    {algorithmLabel(assignedTrip.algorithm)}
                                  </span>
                                  <Badge
                                    variant={tripStatusBadgeVariant(
                                      assignedTrip.status
                                    )}
                                  >
                                    {statusLabel(assignedTrip.status)}
                                  </Badge>
                                </div>
                                <div className="text-[11px] text-muted-foreground">
                                  Driver:{" "}
                                  {driverDisplayName(
                                    assignedTrip.driver_username,
                                    assignedTrip.driver_full_name
                                  )}{" "}
                                  · Đã giao: {assignedTrip.stops_completed}/
                                  {assignedTrip.stops_total} điểm
                                </div>
                                <Button
                                  type="button"
                                  variant="secondary"
                                  size="xs"
                                  onClick={() => onOpenTrip(assignedTrip)}
                                  className="w-fit"
                                >
                                  Mở riêng xe này
                                </Button>
                              </div>
                            )
                          }

                          return (
                            <div
                              key={r.route_index}
                              className="flex flex-col gap-1.5 rounded-md border border-dashed border-border bg-card/60 p-2"
                            >
                              <div className="flex items-center justify-between gap-1.5">
                                <span className="text-xs font-semibold">
                                  Xe {r.vehicle} ·{" "}
                                  {algorithmLabel(plan.algorithm || "RAPO")}
                                </span>
                                <Badge variant="outline">Chưa phân công</Badge>
                              </div>
                              <div className="text-[11px] text-muted-foreground">
                                {r.stops_count} điểm giao · Tải: {fmt(r.load)} ·{" "}
                                {clock(r.duration)}
                              </div>
                              <Button
                                type="button"
                                variant="outline"
                                size="xs"
                                onClick={() =>
                                  onViewAllJobRoutes(jobId, r.route_index)
                                }
                                className="w-fit"
                              >
                                Xem xe này
                              </Button>
                            </div>
                          )
                        })
                      : trips.map((trip) => {
                          const selected = currentTrip?.id === trip.id
                          return (
                            <div
                              key={trip.id}
                              className={`flex flex-col gap-1.5 rounded-md border p-2 ${
                                selected
                                  ? "border-primary bg-primary/5"
                                  : "border-border bg-card"
                              }`}
                            >
                              <div className="flex items-center justify-between gap-1.5">
                                <span className="text-xs font-semibold">
                                  Xe {trip.route_index + 1} ·{" "}
                                  {algorithmLabel(trip.algorithm)}
                                </span>
                                <Badge
                                  variant={tripStatusBadgeVariant(trip.status)}
                                >
                                  {statusLabel(trip.status)}
                                </Badge>
                              </div>
                              <div className="text-[11px] text-muted-foreground">
                                Driver:{" "}
                                {driverDisplayName(
                                  trip.driver_username,
                                  trip.driver_full_name
                                )}{" "}
                                · Đã giao: {trip.stops_completed}/
                                {trip.stops_total} điểm
                              </div>
                              <div className="flex flex-wrap gap-1.5">
                                <Button
                                  type="button"
                                  variant="secondary"
                                  size="xs"
                                  onClick={() => onOpenTrip(trip)}
                                >
                                  Mở tuyến
                                </Button>
                                {!isAdmin && trip.status === "assigned" && (
                                  <Button
                                    type="button"
                                    size="xs"
                                    onClick={() => onStartTrip(trip)}
                                  >
                                    <CheckCircle2Icon data-icon="inline-start" />
                                    Bắt đầu chuyến
                                  </Button>
                                )}
                              </div>
                            </div>
                          )
                        })}
                  </div>
                </details>
              )
            })
          )}
        </CardContent>
      </Card>

      {isAdmin && (
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Các thuật toán dùng chung quần thể ban đầu và ngân sách NFE. Chuyến được
          phân bổ trực tiếp cho Driver.
        </p>
      )}
    </div>
  )
}
