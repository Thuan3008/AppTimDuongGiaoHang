import { useMemo, useState, type FormEvent } from "react"
import {
  ArrowLeftIcon,
  CheckCircle2Icon,
  KeyRoundIcon,
  Loader2Icon,
  LockIcon,
  PencilIcon,
  RefreshCwIcon,
  SearchIcon,
  Trash2Icon,
  UnlockIcon,
  UserPlusIcon,
  UsersIcon,
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
import type { Driver } from "@/types/rapo"

export interface DriverUpdateInput {
  full_name?: string
  active?: boolean
  password?: string
}

interface DriversManagementPageProps {
  drivers: Driver[]
  loading: boolean
  onBackToMap: () => void
  onRefresh: () => Promise<void>
  onCreateDriver: (
    fullName: string,
    username: string,
    password: string
  ) => Promise<{ ok: boolean; error?: string }>
  onUpdateDriver: (
    driverId: number,
    patch: DriverUpdateInput
  ) => Promise<{ ok: boolean; error?: string }>
  onDeleteDriver: (driverId: number) => Promise<{ ok: boolean; error?: string }>
}

type StatusFilter = "all" | "active" | "inactive"

function formatDate(iso?: string): string {
  if (!iso) return "—"
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function DriversManagementPage({
  drivers,
  loading,
  onBackToMap,
  onRefresh,
  onCreateDriver,
  onUpdateDriver,
  onDeleteDriver,
}: DriversManagementPageProps) {
  // Create form state
  const [fullName, setFullName] = useState("")
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [creating, setCreating] = useState(false)

  // Search & filter state
  const [searchQuery, setSearchQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all")

  // Feedback banner
  const [feedback, setFeedback] = useState<{
    text: string
    error: boolean
  } | null>(null)

  // Action busy state
  const [busyDriverId, setBusyDriverId] = useState<number | null>(null)

  // Edit sheet state
  const [editingDriver, setEditingDriver] = useState<Driver | null>(null)
  const [editFullName, setEditFullName] = useState("")
  const [editPassword, setEditPassword] = useState("")
  const [editActive, setEditActive] = useState(true)
  const [savingEdit, setSavingEdit] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)

  const stats = useMemo(() => {
    const total = drivers.length
    const active = drivers.filter((d) => d.active).length
    const inactive = total - active
    const tripsTotal = drivers.reduce((sum, d) => sum + (d.trips_total || 0), 0)
    const tripsActive = drivers.reduce(
      (sum, d) => sum + (d.trips_active || 0),
      0
    )
    return { total, active, inactive, tripsTotal, tripsActive }
  }, [drivers])

  const filteredDrivers = useMemo(() => {
    const q = searchQuery.trim().toLowerCase()
    return drivers.filter((driver) => {
      if (statusFilter === "active" && !driver.active) return false
      if (statusFilter === "inactive" && driver.active) return false
      if (!q) return true
      return (
        driver.username.toLowerCase().includes(q) ||
        (driver.full_name || "").toLowerCase().includes(q)
      )
    })
  }, [drivers, searchQuery, statusFilter])

  const handleCreateSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const trimmedName = fullName.trim()
    const trimmedUsername = username.trim()
    if (!trimmedName) {
      setFeedback({ text: "Vui lòng nhập tên tài xế", error: true })
      return
    }
    if (!trimmedUsername) {
      setFeedback({ text: "Vui lòng nhập tên đăng nhập", error: true })
      return
    }
    if (password.length < 8) {
      setFeedback({
        text: "Mật khẩu phải có ít nhất 8 ký tự",
        error: true,
      })
      return
    }

    setCreating(true)
    setFeedback(null)
    try {
      const res = await onCreateDriver(trimmedName, trimmedUsername, password)
      if (res.ok) {
        setFullName("")
        setUsername("")
        setPassword("")
        toast.success(`Đã tạo tài xế ${trimmedName} (@${trimmedUsername}) thành công`)
        setFeedback({
          text: `Đã tạo tài khoản tài xế ${trimmedName} (@${trimmedUsername})`,
          error: false,
        })
      } else {
        toast.error(res.error || "Không thể tạo tài khoản tài xế")
        setFeedback({
          text: res.error || "Không thể tạo tài khoản tài xế",
          error: true,
        })
      }
    } finally {
      setCreating(false)
    }
  }

  const openEditSheet = (driver: Driver) => {
    setEditingDriver(driver)
    setEditFullName(driver.full_name || "")
    setEditPassword("")
    setEditActive(Boolean(driver.active))
    setEditError(null)
  }

  const handleSaveEdit = async (e: FormEvent) => {
    e.preventDefault()
    if (!editingDriver) return
    const trimmedName = editFullName.trim()
    if (!trimmedName) {
      setEditError("Tên tài xế không được để trống")
      return
    }
    if (editPassword && editPassword.length < 8) {
      setEditError("Mật khẩu mới phải có ít nhất 8 ký tự")
      return
    }

    setSavingEdit(true)
    setEditError(null)
    try {
      const patch: DriverUpdateInput = {
        full_name: trimmedName,
        active: editActive,
      }
      if (editPassword) {
        patch.password = editPassword
      }
      const res = await onUpdateDriver(editingDriver.id, patch)
      if (res.ok) {
        toast.success(`Đã cập nhật tài xế ${trimmedName} (@${editingDriver.username})`)
        setFeedback({
          text: `Đã cập nhật tài khoản ${trimmedName} (@${editingDriver.username})`,
          error: false,
        })
        setEditingDriver(null)
      } else {
        setEditError(res.error || "Không thể cập nhật tài khoản tài xế")
        toast.error(res.error || "Không thể cập nhật tài khoản tài xế")
      }
    } finally {
      setSavingEdit(false)
    }
  }

  const handleToggleActive = async (driver: Driver) => {
    setBusyDriverId(driver.id)
    setFeedback(null)
    try {
      const nextActive = !driver.active
      const res = await onUpdateDriver(driver.id, { active: nextActive })
      if (res.ok) {
        toast.info(
          nextActive
            ? `Đã mở khóa tài khoản @${driver.username}`
            : `Đã tạm khóa tài khoản @${driver.username}`
        )
        setFeedback({
          text: nextActive
            ? `Đã mở khóa tài khoản @${driver.username}`
            : `Đã tạm khóa tài khoản @${driver.username}`,
          error: false,
        })
      } else {
        toast.error(res.error || "Không thể thay đổi trạng thái tài khoản")
        setFeedback({
          text: res.error || "Không thể thay đổi trạng thái tài khoản",
          error: true,
        })
      }
    } finally {
      setBusyDriverId(null)
    }
  }

  const handleDelete = async (driver: Driver) => {
    const label = driver.full_name
      ? `${driver.full_name} (@${driver.username})`
      : `@${driver.username}`
    const confirmed = window.confirm(
      `Bạn có chắc muốn xóa vĩnh viễn tài khoản tài xế ${label}?`
    )
    if (!confirmed) return

    setBusyDriverId(driver.id)
    setFeedback(null)
    try {
      const res = await onDeleteDriver(driver.id)
      if (res.ok) {
        toast.success(`Đã xóa vĩnh viễn tài xế ${label}`)
        setFeedback({
          text: `Đã xóa tài khoản tài xế ${label}`,
          error: false,
        })
      } else {
        toast.error(res.error || "Không thể xóa tài khoản tài xế")
        setFeedback({
          text: res.error || "Không thể xóa tài khoản tài xế",
          error: true,
        })
      }
    } finally {
      setBusyDriverId(null)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-4 p-4 sm:p-6">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <UsersIcon className="size-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold tracking-tight sm:text-xl">
              Quản lý tài khoản tài xế
            </h2>
            <p className="text-xs text-muted-foreground sm:text-sm">
              Tạo mới, cập nhật họ tên, đặt lại mật khẩu, tạm khóa hoặc xóa tài
              khoản Driver.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={loading}
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
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Tổng tài xế</CardDescription>
            <CardTitle className="font-mono text-2xl font-bold">
              {stats.total}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Đang hoạt động</CardDescription>
            <CardTitle className="font-mono text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              {stats.active}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">Đã tạm khóa</CardDescription>
            <CardTitle className="font-mono text-2xl font-bold text-muted-foreground">
              {stats.inactive}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card size="sm">
          <CardHeader className="pb-1">
            <CardDescription className="text-xs">
              Chuyến đang giao / Tổng chuyến
            </CardDescription>
            <CardTitle className="font-mono text-2xl font-bold">
              {stats.tripsActive} / {stats.tripsTotal}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Main Content: Create Form + Drivers Table */}
      <div className="grid gap-4 lg:grid-cols-12">
        {/* Create Driver Card */}
        <Card className="lg:col-span-4">
          <CardHeader>
            <CardTitle>Tạo tài khoản tài xế</CardTitle>
            <CardDescription>
              Tài khoản mới có thể đăng nhập ngay để nhận tuyến và cập nhật tiến
              độ giao hàng.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form
              onSubmit={handleCreateSubmit}
              className="flex flex-col gap-3.5"
            >
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="mgmt-driver-fullname"
                  className="text-xs font-medium"
                >
                  Tên tài xế <span className="text-destructive">*</span>
                </label>
                <Input
                  id="mgmt-driver-fullname"
                  type="text"
                  maxLength={100}
                  placeholder="Ví dụ: Nguyễn Văn A"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  disabled={creating}
                  required
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="mgmt-driver-username"
                  className="text-xs font-medium"
                >
                  Tên đăng nhập <span className="text-destructive">*</span>
                </label>
                <Input
                  id="mgmt-driver-username"
                  type="text"
                  minLength={3}
                  maxLength={50}
                  placeholder="Ví dụ: driver_nam"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={creating}
                  required
                />
                <span className="text-[11px] text-muted-foreground">
                  3–50 ký tự chữ, số, dấu gạch ngang (-) hoặc gạch dưới (_).
                </span>
              </div>

              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="mgmt-driver-password"
                  className="text-xs font-medium"
                >
                  Mật khẩu <span className="text-destructive">*</span>
                </label>
                <Input
                  id="mgmt-driver-password"
                  type="password"
                  minLength={8}
                  placeholder="Tối thiểu 8 ký tự"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={creating}
                  required
                />
              </div>

              <Button type="submit" disabled={creating} className="mt-1 w-full">
                {creating ? (
                  <Loader2Icon
                    data-icon="inline-start"
                    className="animate-spin"
                  />
                ) : (
                  <UserPlusIcon data-icon="inline-start" />
                )}
                {creating ? "Đang tạo tài khoản…" : "Tạo tài khoản Driver"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Drivers List Table Card */}
        <Card className="lg:col-span-8">
          <CardHeader className="gap-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle>Danh sách tài xế ({filteredDrivers.length})</CardTitle>
                <CardDescription>
                  Tài khoản đã khóa không thể đăng nhập và không hiện trong danh
                  sách phân công tuyến.
                </CardDescription>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="relative min-w-[200px] flex-1">
                <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  type="search"
                  placeholder="Tìm theo tên tài xế hoặc tài khoản…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8"
                />
              </div>

              <div className="flex items-center gap-1">
                {(
                  [
                    ["all", "Tất cả"],
                    ["active", "Hoạt động"],
                    ["inactive", "Đã khóa"],
                  ] as const
                ).map(([key, label]) => (
                  <Button
                    key={key}
                    type="button"
                    size="sm"
                    variant={statusFilter === key ? "default" : "outline"}
                    onClick={() => setStatusFilter(key)}
                  >
                    {label}
                  </Button>
                ))}
              </div>
            </div>
          </CardHeader>

          <CardContent>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/50 text-muted-foreground">
                    <th className="px-3 py-2.5 font-medium">Tài xế</th>
                    <th className="px-3 py-2.5 font-medium whitespace-nowrap">
                      Trạng thái
                    </th>
                    <th className="px-3 py-2.5 font-medium whitespace-nowrap">
                      Chuyến giao
                    </th>
                    <th className="px-3 py-2.5 font-medium whitespace-nowrap">
                      Ngày tạo
                    </th>
                    <th className="px-3 py-2.5 text-right font-medium whitespace-nowrap">
                      Thao tác
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {filteredDrivers.length === 0 ? (
                    <tr>
                      <td
                        colSpan={5}
                        className="px-3 py-8 text-center text-muted-foreground"
                      >
                        Không tìm thấy tài khoản tài xế nào phù hợp.
                      </td>
                    </tr>
                  ) : (
                    filteredDrivers.map((driver) => {
                      const isRowBusy = busyDriverId === driver.id
                      const hasTrips = (driver.trips_total || 0) > 0
                      return (
                        <tr
                          key={driver.id}
                          className="transition-colors hover:bg-muted/30"
                        >
                          <td className="px-3 py-2.5">
                            <div className="font-semibold text-foreground">
                              {driver.full_name?.trim() || (
                                <span className="font-normal text-muted-foreground italic">
                                  Chưa đặt tên
                                </span>
                              )}
                            </div>
                            <div className="font-mono text-[11px] text-muted-foreground">
                              @{driver.username} · ID #{driver.id}
                            </div>
                          </td>

                          <td className="px-3 py-2.5 whitespace-nowrap">
                            <Badge
                              variant={driver.active ? "default" : "secondary"}
                            >
                              {driver.active ? "Hoạt động" : "Đã khóa"}
                            </Badge>
                          </td>

                          <td className="px-3 py-2.5 whitespace-nowrap">
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono">
                                {driver.trips_completed || 0}/
                                {driver.trips_total || 0} hoàn thành
                              </span>
                              {(driver.trips_active || 0) > 0 && (
                                <Badge variant="outline">
                                  {driver.trips_active} đang giao
                                </Badge>
                              )}
                            </div>
                          </td>

                          <td className="px-3 py-2.5 font-mono text-muted-foreground whitespace-nowrap">
                            {formatDate(driver.created_at)}
                          </td>

                          <td className="px-3 py-2.5 whitespace-nowrap">
                            <div className="flex items-center justify-end gap-1.5">
                              <Button
                                type="button"
                                variant="outline"
                                size="xs"
                                disabled={isRowBusy}
                                onClick={() => openEditSheet(driver)}
                              >
                                <PencilIcon data-icon="inline-start" />
                                Sửa
                              </Button>

                              <Button
                                type="button"
                                variant="secondary"
                                size="xs"
                                disabled={isRowBusy}
                                onClick={() => void handleToggleActive(driver)}
                              >
                                {isRowBusy ? (
                                  <Loader2Icon
                                    data-icon="inline-start"
                                    className="animate-spin"
                                  />
                                ) : driver.active ? (
                                  <LockIcon data-icon="inline-start" />
                                ) : (
                                  <UnlockIcon data-icon="inline-start" />
                                )}
                                {driver.active ? "Khóa" : "Mở khóa"}
                              </Button>

                              <Button
                                type="button"
                                variant="destructive"
                                size="xs"
                                disabled={isRowBusy || hasTrips}
                                title={
                                  hasTrips
                                    ? "Tài xế đã có lịch sử chuyến giao, chỉ có thể tạm khóa"
                                    : "Xóa tài khoản tài xế"
                                }
                                onClick={() => void handleDelete(driver)}
                              >
                                <Trash2Icon data-icon="inline-start" />
                                Xóa
                              </Button>
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
        </Card>
      </div>

      {/* Edit Driver Sheet */}
      <Sheet
        open={Boolean(editingDriver)}
        onOpenChange={(open) => {
          if (!open) setEditingDriver(null)
        }}
      >
        <SheetContent side="right" className="w-[92vw] max-w-md">
          <SheetHeader>
            <SheetTitle>Chỉnh sửa tài khoản tài xế</SheetTitle>
            <SheetDescription>
              {editingDriver
                ? `Tài khoản @${editingDriver.username} (ID #${editingDriver.id})`
                : ""}
            </SheetDescription>
          </SheetHeader>

          {editingDriver && (
            <form
              onSubmit={handleSaveEdit}
              className="flex flex-1 flex-col justify-between gap-4 px-4 pb-4"
            >
              <div className="flex flex-col gap-4">
                <div className="flex flex-col gap-1.5">
                  <label
                    htmlFor="edit-driver-fullname"
                    className="text-xs font-medium"
                  >
                    Tên tài xế <span className="text-destructive">*</span>
                  </label>
                  <Input
                    id="edit-driver-fullname"
                    type="text"
                    maxLength={100}
                    placeholder="Ví dụ: Nguyễn Văn A"
                    value={editFullName}
                    onChange={(e) => setEditFullName(e.target.value)}
                    disabled={savingEdit}
                    required
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label
                    htmlFor="edit-driver-password"
                    className="flex items-center gap-1.5 text-xs font-medium"
                  >
                    <KeyRoundIcon className="size-3.5 text-muted-foreground" />
                    Đặt lại mật khẩu mới (tùy chọn)
                  </label>
                  <Input
                    id="edit-driver-password"
                    type="password"
                    minLength={8}
                    placeholder="Để trống nếu giữ nguyên mật khẩu cũ"
                    value={editPassword}
                    onChange={(e) => setEditPassword(e.target.value)}
                    disabled={savingEdit}
                  />
                </div>

                <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-border bg-muted/30 p-3 text-xs font-medium select-none">
                  <input
                    type="checkbox"
                    checked={editActive}
                    disabled={savingEdit}
                    onChange={(e) => setEditActive(e.target.checked)}
                    className="size-4 accent-primary"
                  />
                  <span>Cho phép tài khoản hoạt động & đăng nhập</span>
                </label>

                {editError && (
                  <div
                    role="alert"
                    className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive"
                  >
                    {editError}
                  </div>
                )}
              </div>

              <SheetFooter className="flex flex-row justify-end gap-2 p-0">
                <Button
                  type="button"
                  variant="outline"
                  disabled={savingEdit}
                  onClick={() => setEditingDriver(null)}
                >
                  Hủy
                </Button>
                <Button type="submit" disabled={savingEdit}>
                  {savingEdit && (
                    <Loader2Icon
                      data-icon="inline-start"
                      className="animate-spin"
                    />
                  )}
                  Lưu thay đổi
                </Button>
              </SheetFooter>
            </form>
          )}
        </SheetContent>
      </Sheet>
    </div>
  )
}
