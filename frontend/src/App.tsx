import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { DispatchSidebar, type StatusMessage } from "@/components/DispatchSidebar"
import {
  DriversManagementPage,
  type DriverUpdateInput,
} from "@/components/DriversManagementPage"
import { HeaderBar } from "@/components/HeaderBar"
import { LoginScreen } from "@/components/LoginScreen"
import { MapPanel, type MapPanelHandle } from "@/components/MapPanel"
import { ResultsWorkspace } from "@/components/ResultsWorkspace"
import { TripsManagementPage } from "@/components/TripsManagementPage"
import { useTheme } from "@/components/theme-provider"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Toaster, toast } from "@/components/ui/sonner"
import {
  ApiError,
  algorithmLabel,
  api,
  driverDisplayName,
  exportAlgorithmCSV,
  exportPayloadJSON,
  fmt,
  inputTime,
  timeSeconds,
} from "@/lib/api"
import type {
  AdminJobGroup,
  AppView,
  CustomCustomerPayload,
  CustomPointDraft,
  CustomPointsPayload,
  DatasetEntry,
  DepotDraft,
  Driver,
  GeocodeCandidate,
  GeocodeResponse,
  MapPickTarget,
  MapPoint,
  OptimizationConfig,
  OptimizationPayload,
  RunJobResponse,
  SnapPointResponse,
  Trip,
  TripStop,
  TripsResponse,
  User,
} from "@/types/rapo"

const DEFAULT_DEPOT_NOTE = "Nhập địa chỉ rồi bấm Tìm hoặc Xem điểm."

function createDefaultDepot(): DepotDraft {
  return {
    name: "Kho",
    address: "",
    lat: "",
    lon: "",
    snapConfirmed: false,
    note: DEFAULT_DEPOT_NOTE,
    warning: null,
    candidates: [],
    selectedCandidate: null,
  }
}

function createPointDraft(
  value: Partial<CustomCustomerPayload> = {}
): CustomPointDraft {
  const uid =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
  const latStr = value.lat != null ? String(value.lat) : ""
  const lonStr = value.lon != null ? String(value.lon) : ""
  const hasSnap = Boolean(value.lat != null && value.lon != null && value.snap_status)

  return {
    uid,
    name: value.name || "",
    address: value.address || "",
    lat: latStr,
    lon: lonStr,
    demand: value.demand != null ? String(value.demand) : "1",
    serviceMinutes:
      value.service_time == null ? "5" : String(Number(value.service_time) / 60),
    windowStart: value.time_window ? inputTime(value.time_window[0]) : "",
    windowEnd: value.time_window ? inputTime(value.time_window[1]) : "",
    resolvedAddress:
      value.address && value.lat != null && value.lon != null
        ? value.address
        : undefined,
    snapConfirmed: Boolean(
      value.snap_confirmed || value.snap_status === "confirm"
    ),
    snapCheckedLat: hasSnap ? latStr : undefined,
    snapCheckedLon: hasSnap ? lonStr : undefined,
    note: DEFAULT_DEPOT_NOTE,
    warning: null,
    candidates: [],
    selectedCandidate: null,
  }
}

function parseRequiredNumber(raw: string, label: string): number {
  if (raw.trim() === "") {
    throw new Error(`Cần nhập ${label}`)
  }
  const value = Number(raw)
  if (!Number.isFinite(value)) {
    throw new Error(`Cần nhập ${label}`)
  }
  return value
}

function isSnapCheckCurrent(
  lat: string,
  lon: string,
  checkedLat?: string,
  checkedLon?: string
): boolean {
  const latNum = Number(lat)
  const lonNum = Number(lon)
  if (lat.trim() === "" || lon.trim() === "") return false
  if (!Number.isFinite(latNum) || !Number.isFinite(lonNum)) return false
  return checkedLat === String(latNum) && checkedLon === String(lonNum)
}

export function App() {
  const { theme, setTheme } = useTheme()
  const dark =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches)
  const toggleTheme = useCallback(() => {
    setTheme(dark ? "light" : "dark")
  }, [dark, setTheme])

  // Auth state
  const [authChecked, setAuthChecked] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const [user, setUser] = useState<User | null>(null)

  // Layout state
  const [activeView, setActiveView] = useState<AppView>("map")
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [driversLoading, setDriversLoading] = useState(false)
  const [tripsLoading, setTripsLoading] = useState(false)
  const [statusMsg, setStatusMsg] = useState<StatusMessage>({
    text: "Đang nạp dữ liệu…",
    error: false,
  })
  const [clockText, setClockText] = useState("Sẵn sàng")

  // Admin catalog & config state
  const [catalog, setCatalog] = useState<DatasetEntry[]>([])
  const [drivers, setDrivers] = useState<Driver[]>([])
  const activeDrivers = useMemo(
    () => drivers.filter((d) => d.active),
    [drivers]
  )
  const [tripName, setTripName] = useState("")
  const [dataset, setDataset] = useState("")
  const [manualMode, setManualMode] = useState(false)
  const [manualVehicles, setManualVehicles] = useState("1")
  const [manualCapacity, setManualCapacity] = useState("100")
  const [datasetCapacity, setDatasetCapacity] = useState<number>(100)
  const [depot, setDepot] = useState<DepotDraft>(createDefaultDepot)
  const [customPoints, setCustomPoints] = useState<CustomPointDraft[]>([])
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([])
  const [populationSize, setPopulationSize] = useState("10")
  const [budget, setBudget] = useState("290")
  const [seed, setSeed] = useState("0")
  const [numSeeds, setNumSeeds] = useState("1")
  const [earliestDeparture, setEarliestDeparture] = useState("08:00")
  const [latestDeparture, setLatestDeparture] = useState("09:00")

  // Optimization, Map & Trips state
  const [busy, setBusy] = useState(false)
  const [startedAt, setStartedAt] = useState(0)
  const [points, setPoints] = useState<MapPoint[]>([])
  const [payload, setPayload] = useState<OptimizationPayload | null>(null)
  const [selectedResultAlg, setSelectedResultAlg] = useState("")
  const [allTrips, setAllTrips] = useState<Trip[]>([])
  const [adminJobs, setAdminJobs] = useState<AdminJobGroup[]>([])
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [currentTrip, setCurrentTrip] = useState<Trip | null>(null)
  const [deliveryStatuses, setDeliveryStatuses] = useState<TripStop[]>([])
  const [showStopsPanel, setShowStopsPanel] = useState(false)
  const [pendingPick, setPendingPick] = useState<MapPickTarget | null>(null)

  const inProgressTripsCount = useMemo(() => {
    if (user?.role === "driver") {
      return currentTrip?.status === "in_progress" ? 1 : 0
    }
    return allTrips.filter((t) => t.status === "in_progress").length
  }, [user, currentTrip, allTrips])

  // Refs for fresh async reads
  const mapRef = useRef<MapPanelHandle | null>(null)
  const pollTimerRef = useRef<number | null>(null)
  const userRef = useRef<User | null>(user)
  const busyRef = useRef(busy)
  const datasetRef = useRef(dataset)
  const depotRef = useRef(depot)
  const customPointsRef = useRef(customPoints)
  const currentJobIdRef = useRef(currentJobId)
  const currentTripRef = useRef(currentTrip)
  const pendingPickRef = useRef(pendingPick)

  useEffect(() => {
    userRef.current = user
    busyRef.current = busy
    datasetRef.current = dataset
    depotRef.current = depot
    customPointsRef.current = customPoints
    currentJobIdRef.current = currentJobId
    currentTripRef.current = currentTrip
    pendingPickRef.current = pendingPick
  }, [
    user,
    busy,
    dataset,
    depot,
    customPoints,
    currentJobId,
    currentTrip,
    pendingPick,
  ])

  const setStatus = useCallback((text: string, error = false) => {
    setStatusMsg({ text, error })
  }, [])

  const clearPollTimer = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const clearResult = useCallback(() => {
    setPayload(null)
    setSelectedResultAlg("")
    setDeliveryStatuses([])
    setCurrentTrip(null)
    setCurrentJobId(null)
    setShowStopsPanel(false)
  }, [])

  const applyResultPayload = useCallback(
    (data: OptimizationPayload, stops: TripStop[] = []) => {
      setPayload(data)
      setPoints(data.points || [])
      setDeliveryStatuses(Array.isArray(stops) ? stops : [])

      const names = Object.keys(data.results || {})
      setSelectedResultAlg(names[0] || "")

      const errorText = Object.entries(data.errors || {})
        .map(([alg, msg]) => `${algorithmLabel(alg)}: ${msg}`)
        .join(" | ")
      setStatus(
        errorText || "Hoàn tất · Kết quả đã lưu trên máy",
        Boolean(errorText)
      )
    },
    [setStatus]
  )

  const reloadTrips = useCallback(async (overrideUser?: User | null) => {
    const activeUser = overrideUser ?? userRef.current
    if (!activeUser) return

    try {
      const data = await api<TripsResponse>("/api/trips")
      setAllTrips(data.trips || [])
      setAdminJobs(Array.isArray(data.jobs) ? data.jobs : [])

      const activeJobId = currentJobIdRef.current
      const activeTrip = currentTripRef.current
      if (activeUser.role === "admin" && activeJobId && !activeTrip) {
        try {
          const stopData = await api<{ stops: TripStop[] }>(
            `/api/runs/${activeJobId}/stops`
          )
          if (Array.isArray(stopData.stops)) {
            setDeliveryStatuses(stopData.stops)
          }
        } catch {
          // ignore transient sync error
        }
      }
    } catch {
      // ignore background poll errors
    }
  }, [])

  const loadDrivers = useCallback(async () => {
    setDriversLoading(true)
    try {
      const data = await api<{ drivers: Driver[] }>(
        "/api/drivers?include_inactive=1"
      )
      setDrivers(data.drivers || [])
      const activeCount = (data.drivers || []).filter((d) => d.active).length
      if (!activeCount) {
        setStatus("Hãy tạo ít nhất một tài khoản Driver hoạt động", true)
      }
    } finally {
      setDriversLoading(false)
    }
  }, [setStatus])

  const previewDataset = useCallback(
    async (
      datasetId: string,
      catalogList: DatasetEntry[],
      shouldFillDepot = false
    ) => {
      clearResult()
      const entry = catalogList.find((ds) => ds.id === datasetId)
      const defaultAlgs = entry?.algorithms?.length ? [entry.algorithms[0]] : []
      setSelectedAlgorithms(defaultAlgs)

      setBusy(true)
      try {
        const data = await api<{
          points: MapPoint[]
          customers: number
          capacity: number
        }>(`/api/datasets/${datasetId}`)
        setPoints(data.points || [])
        if (data.capacity) {
          setDatasetCapacity(data.capacity)
        }
        if (shouldFillDepot && data.points.length > 0) {
          setDepot((prev) => ({
            ...prev,
            lat: prev.lat || String(data.points[0].lat),
            lon: prev.lon || String(data.points[0].lon),
          }))
        }
        setStatus(
          `Đã nạp ${data.customers} khách hàng · Sức chứa ${data.capacity}`
        )
      } finally {
        setBusy(false)
      }
    },
    [clearResult, setStatus]
  )

  const applySavedConfig = useCallback(
    (config: OptimizationConfig, catalogList: DatasetEntry[]) => {
      setDataset(config.dataset)
      setTripName(config.trip_name || `Chuyến ${config.dataset}`)
      setPopulationSize(String(config.population_size))
      setBudget(String(config.budget))
      setSeed(String(config.seed))
      setNumSeeds(String(config.num_seeds || 1))

      const depSeconds =
        config.earliest_departure == null ? 8 * 3600 : config.earliest_departure
      setEarliestDeparture(inputTime(depSeconds) || "08:00")
      if (config.latest_departure != null) {
        setLatestDeparture(inputTime(config.latest_departure) || "09:00")
      }

      const custom = config.custom_points
      setManualMode(Boolean(custom))
      if (custom) {
        setManualVehicles(String(custom.num_vehicles))
        setManualCapacity(String(custom.vehicle_capacity))
        const latStr = String(custom.depot.lat)
        const lonStr = String(custom.depot.lon)
        setDepot({
          name: custom.depot.name || "Kho",
          address: custom.depot.address || "",
          lat: latStr,
          lon: lonStr,
          resolvedAddress: custom.depot.address || undefined,
          snapConfirmed: Boolean(custom.depot.snap_confirmed),
          snapCheckedLat: custom.depot.snap_status ? latStr : undefined,
          snapCheckedLon: custom.depot.snap_status ? lonStr : undefined,
          note: DEFAULT_DEPOT_NOTE,
          warning: null,
          candidates: [],
          selectedCandidate: null,
        })
        setCustomPoints(custom.customers.map((pt) => createPointDraft(pt)))
      } else {
        setCustomPoints([])
      }

      const available =
        catalogList.find((ds) => ds.id === config.dataset)?.algorithms || []
      const filteredAlgs = (config.algorithms || []).filter((alg) =>
        available.includes(alg)
      )
      setSelectedAlgorithms(
        filteredAlgs.length
          ? filteredAlgs
          : available.length
            ? [available[0]]
            : []
      )
    },
    []
  )

  const pollJob = useCallback(
    async (jobId: string) => {
      const step = async () => {
        clearPollTimer()
        try {
          const job = await api<RunJobResponse>(`/api/runs/${jobId}`)
          setStatus(job.stage)

          if (job.payload) {
            setCurrentJobId(jobId)
            setCurrentTrip(null)
            setShowStopsPanel(false)
            applyResultPayload(job.payload, job.stops || [])
            setBusy(false)
            setStartedAt(0)
            await reloadTrips()

            const algNames = Object.keys(job.payload.results || {})
            const firstAlg = algNames[0]
            const firstResult = firstAlg ? job.payload.results[firstAlg] : null
            const rCount = firstResult?.routes?.length || 0
            const fitnessStr = firstResult ? fmt(firstResult.fitness) : "—"

            toast.success(`Đã tối ưu thành công ${rCount} tuyến`, {
              description: `Thuật toán: ${firstAlg || "RAPO"} · Fitness: ${fitnessStr}s · Seed: ${firstResult?.seed ?? 0}`,
            })
            return
          }

          if (job.status === "failed") {
            setBusy(false)
            setStartedAt(0)
            setStatus(job.stage, true)
            toast.error("Tối ưu chuyến thất bại", {
              description: job.stage || "Đã xảy ra lỗi trong quá trình tối ưu",
            })
            return
          }

          pollTimerRef.current = window.setTimeout(() => {
            void step()
          }, 1500)
        } catch (err) {
          setBusy(false)
          setStartedAt(0)
          setStatus(
            `${
              err instanceof Error ? err.message : "Lỗi kết nối"
            }. Tải lại trang để kết nối lại lượt chạy.`,
            true
          )
        }
      }
      await step()
    },
    [applyResultPayload, clearPollTimer, reloadTrips, setStatus]
  )

  const bootApp = useCallback(
    async (loggedInUser: User) => {
      await reloadTrips(loggedInUser)

      if (loggedInUser.role === "driver") {
        setStatus("Chọn một chuyến được giao để xem lộ trình")
        return
      }

      const datasetResp = await api<{ datasets: DatasetEntry[] }>(
        "/api/datasets"
      )
      const catalogList = datasetResp.datasets || []
      setCatalog(catalogList)

      const ready = catalogList.find((ds) => ds.missing.length === 0)
      if (!ready) {
        setStatus("Chưa có dataset đủ sidecar.", true)
        return
      }

      setDataset(ready.id)
      setTripName(`Chuyến giao ${new Date().toLocaleDateString("vi-VN")}`)

      await loadDrivers()
      await previewDataset(ready.id, catalogList)

      let savedJobId: string | null
      try {
        savedJobId = localStorage.getItem("rapo-map-job")
      } catch {
        savedJobId = null
      }

      if (savedJobId) {
        try {
          const job = await api<RunJobResponse>(`/api/runs/${savedJobId}`)
          if (job.config) {
            applySavedConfig(job.config, catalogList)
          }
          if (job.payload) {
            setCurrentJobId(savedJobId)
            setCurrentTrip(null)
            setShowStopsPanel(false)
            applyResultPayload(job.payload, job.stops || [])
            setStatus("Đã nạp kết quả gần nhất · có thể giao từng route")
            return
          }
          if (job.status === "queued" || job.status === "running") {
            setBusy(true)
            setStartedAt(Date.now())
            await pollJob(savedJobId)
            return
          }
        } catch {
          try {
            localStorage.removeItem("rapo-map-job")
          } catch {
            // ignore
          }
        }
      }

      setStatus("Sẵn sàng chạy tối ưu")
    },
    [
      applyResultPayload,
      applySavedConfig,
      loadDrivers,
      pollJob,
      previewDataset,
      reloadTrips,
      setStatus,
    ]
  )

  // Initial auth check
  useEffect(() => {
    let active = true
    ;(async () => {
      try {
        const data = await api<{ user: User }>("/api/auth/me")
        if (!active) return
        setUser(data.user)
        setAuthChecked(true)
        await bootApp(data.user)
      } catch (err) {
        if (!active) return
        if (err instanceof ApiError && err.status !== 401) {
          setAuthError(err.message)
        }
        setAuthChecked(true)
      }
    })()

    return () => {
      active = false
      clearPollTimer()
    }
  }, [bootApp, clearPollTimer])

  // Running elapsed clock (1s)
  useEffect(() => {
    const id = window.setInterval(() => {
      if (busy && startedAt) {
        setClockText(`Đang chạy · ${Math.floor((Date.now() - startedAt) / 1000)}s`)
      } else {
        setClockText("RAPO / MAP")
      }
    }, 1000)
    return () => window.clearInterval(id)
  }, [busy, startedAt])

  // Periodic 10s sync of trips & stop statuses
  useEffect(() => {
    if (!user) return
    const id = window.setInterval(() => {
      if (!busyRef.current) {
        void reloadTrips()
      }
    }, 10000)
    return () => window.clearInterval(id)
  }, [user, reloadTrips])

  // ---------------------------------------------------------------------------
  // Auth Handlers
  // ---------------------------------------------------------------------------
  const handleLogin = async (username: string, password: string) => {
    const data = await api<{ user: User }>("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    })
    setAuthError(null)
    setUser(data.user)
    await bootApp(data.user)
  }

  const handleLogout = async () => {
    clearPollTimer()
    try {
      await api("/api/auth/logout", { method: "POST" })
    } finally {
      try {
        localStorage.removeItem("rapo-map-job")
      } catch {
        // ignore
      }
      setUser(null)
      setActiveView("map")
      setPayload(null)
      setPoints([])
      setAllTrips([])
      setAdminJobs([])
      setCurrentJobId(null)
      setCurrentTrip(null)
      setDeliveryStatuses([])
      setShowStopsPanel(false)
      setPendingPick(null)
      setMobileMenuOpen(false)
    }
  }

  const handleCreateDriver = async (
    fullName: string,
    username: string,
    password: string
  ): Promise<boolean> => {
    try {
      await api("/api/drivers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name: fullName, username, password }),
      })
      await loadDrivers()
      setStatus(
        `Đã tạo Driver: ${fullName || username} (@${username})`,
        false
      )
      return true
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Lỗi tạo Driver", true)
      return false
    }
  }

  const handleCreateDriverWithFeedback = async (
    fullName: string,
    username: string,
    password: string
  ): Promise<{ ok: boolean; error?: string }> => {
    try {
      await api("/api/drivers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name: fullName, username, password }),
      })
      await loadDrivers()
      return { ok: true }
    } catch (err) {
      return {
        ok: false,
        error: err instanceof Error ? err.message : "Lỗi tạo tài khoản tài xế",
      }
    }
  }

  const handleUpdateDriver = async (
    driverId: number,
    patch: DriverUpdateInput
  ): Promise<{ ok: boolean; error?: string }> => {
    try {
      await api(`/api/drivers/${driverId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      })
      await Promise.all([loadDrivers(), reloadTrips()])
      return { ok: true }
    } catch (err) {
      return {
        ok: false,
        error:
          err instanceof Error
            ? err.message
            : "Lỗi cập nhật tài khoản tài xế",
      }
    }
  }

  const handleDeleteDriver = async (
    driverId: number
  ): Promise<{ ok: boolean; error?: string }> => {
    try {
      await api(`/api/drivers/${driverId}`, {
        method: "DELETE",
      })
      await loadDrivers()
      return { ok: true }
    } catch (err) {
      return {
        ok: false,
        error:
          err instanceof Error ? err.message : "Lỗi xóa tài khoản tài xế",
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Geocoding, Snap-Point & Custom Delivery Points
  // ---------------------------------------------------------------------------
  const checkPointSnap = async (lat: number, lon: number) => {
    return api<SnapPointResponse>("/api/snap-point", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: datasetRef.current, lat, lon }),
    })
  }

  const applyCandidateToDraft = <T extends DepotDraft | CustomPointDraft>(
    draft: T,
    candidate: GeocodeCandidate,
    address: string,
    candidates: GeocodeCandidate[]
  ): T | null => {
    if (candidate.snap_status === "rejected") {
      setStatus(
        `Điểm này cách đường gần nhất ${fmt(
          candidate.snap_distance_m
        )} m, vượt giới hạn 300 m. Hãy chọn lại vị trí.`,
        true
      )
      return null
    }
    if (candidate.snap_status === "confirm") {
      const accepted = window.confirm(
        `Điểm này cách đường gần nhất ${fmt(
          candidate.snap_distance_m
        )} m. Bạn có chắc muốn dùng vị trí này không?`
      )
      if (!accepted) return null
    }
    return {
      ...draft,
      lat: String(candidate.lat),
      lon: String(candidate.lon),
      resolvedAddress: address,
      snapConfirmed: candidate.snap_status === "confirm",
      snapCheckedLat: String(candidate.lat),
      snapCheckedLon: String(candidate.lon),
      warning: null,
      candidates,
      selectedCandidate: candidate,
      note: `Đã chọn: ${candidate.display_name} · ${Number(
        candidate.lat
      ).toFixed(6)}, ${Number(candidate.lon).toFixed(6)} · cách đường ${fmt(
        candidate.snap_distance_m
      )} m`,
      resolving: false,
    }
  }

  const resolveSingleDraft = async <T extends DepotDraft | CustomPointDraft>(
    draft: T,
    updateDraft: (updater: (prev: T) => T) => void
  ): Promise<T> => {
    const address = draft.address.trim()
    const latNum = Number(draft.lat)
    const lonNum = Number(draft.lon)
    const hasCoords =
      draft.lat.trim() !== "" &&
      draft.lon.trim() !== "" &&
      Number.isFinite(latNum) &&
      Number.isFinite(lonNum)

    if (!address) {
      if (hasCoords) {
        if (
          isSnapCheckCurrent(
            draft.lat,
            draft.lon,
            draft.snapCheckedLat,
            draft.snapCheckedLon
          )
        ) {
          return draft
        }
        updateDraft((prev) => ({
          ...prev,
          resolving: true,
          note: "Đang kiểm tra khoảng cách tới đường…",
        }))
        try {
          const snap = await checkPointSnap(latNum, lonNum)
          const candidate: GeocodeCandidate = {
            ...snap,
            display_name: "Tọa độ nhập thủ công",
            provider: "Tọa độ",
            house_match: "not_requested",
          }
          const updated = applyCandidateToDraft(draft, candidate, "", [
            candidate,
          ])
          if (!updated) {
            updateDraft((prev) => ({ ...prev, resolving: false }))
            throw new Error("Vị trí chưa được xác nhận")
          }
          updateDraft(() => updated)
          return updated
        } catch (err) {
          updateDraft((prev) => ({ ...prev, resolving: false }))
          throw err
        }
      }
      throw new Error("Nhập địa chỉ hoặc tọa độ nâng cao")
    }

    if (
      draft.resolvedAddress === address &&
      hasCoords &&
      isSnapCheckCurrent(
        draft.lat,
        draft.lon,
        draft.snapCheckedLat,
        draft.snapCheckedLon
      )
    ) {
      return draft
    }

    updateDraft((prev) => ({
      ...prev,
      resolving: true,
      note: "Đang tìm địa chỉ…",
    }))

    try {
      const data = await api<GeocodeResponse>("/api/geocode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: datasetRef.current, address }),
      })

      if (data.requires_confirmation) {
        updateDraft((prev) => ({
          ...prev,
          resolvedAddress: undefined,
          lat: "",
          lon: "",
          snapConfirmed: false,
          snapCheckedLat: undefined,
          snapCheckedLon: undefined,
          warning: data.warning || "Địa chỉ cần xác nhận thủ công.",
          candidates: data.candidates || [],
          selectedCandidate: null,
          resolving: false,
        }))
        throw new Error(
          `${data.warning || "Địa chỉ chưa rõ ràng."} Chưa tự động chọn tọa độ.`
        )
      }

      const updated = applyCandidateToDraft(
        draft,
        data.result,
        address,
        data.candidates || []
      )
      if (!updated) {
        updateDraft((prev) => ({ ...prev, resolving: false }))
        throw new Error("Vị trí chưa được xác nhận")
      }
      updateDraft(() => updated)
      return updated
    } catch (err) {
      updateDraft((prev) => ({ ...prev, resolving: false }))
      throw err
    }
  }

  const resolveDepot = async (): Promise<DepotDraft> => {
    return resolveSingleDraft(depotRef.current, (updater) => {
      setDepot((prev) => {
        const next = updater(prev)
        depotRef.current = next
        return next
      })
    })
  }

  const resolvePointByUid = async (uid: string): Promise<CustomPointDraft> => {
    const target = customPointsRef.current.find((p) => p.uid === uid)
    if (!target) throw new Error("Không tìm thấy điểm giao")
    return resolveSingleDraft(target, (updater) => {
      setCustomPoints((prev) => {
        const next = prev.map((item) =>
          item.uid === uid ? updater(item) : item
        )
        customPointsRef.current = next
        return next
      })
    })
  }

  const resolveAllAddresses = async (): Promise<{
    depot: DepotDraft
    customers: CustomPointDraft[]
  }> => {
    const resolvedDepot = await resolveDepot()
    const resolvedCustomers: CustomPointDraft[] = []
    for (const pt of customPointsRef.current) {
      const resolved = await resolvePointByUid(pt.uid)
      resolvedCustomers.push(resolved)
    }
    return { depot: resolvedDepot, customers: resolvedCustomers }
  }

  const collectCustomPoints = (
    depotDraft: DepotDraft,
    pointDrafts: CustomPointDraft[]
  ): CustomPointsPayload => {
    const customers: CustomCustomerPayload[] = pointDrafts.map((pt, index) => {
      const hasStart = Boolean(pt.windowStart)
      const hasEnd = Boolean(pt.windowEnd)
      if (hasStart !== hasEnd) {
        throw new Error(
          `Điểm giao ${index + 1} phải nhập đủ hai đầu khung giờ`
        )
      }
      let timeWindow: [number, number] | null = null
      if (hasStart && hasEnd) {
        const startSec = timeSeconds(pt.windowStart)
        const endSec = timeSeconds(pt.windowEnd)
        if (startSec == null || endSec == null || startSec >= endSec) {
          throw new Error(`Khung giờ điểm giao ${index + 1} không hợp lệ`)
        }
        timeWindow = [startSec, endSec]
      }

      return {
        name: pt.name.trim() || `Điểm giao ${index + 1}`,
        address: pt.address.trim(),
        lat: parseRequiredNumber(pt.lat, `vĩ độ điểm giao ${index + 1}`),
        lon: parseRequiredNumber(pt.lon, `kinh độ điểm giao ${index + 1}`),
        demand: parseRequiredNumber(pt.demand, `nhu cầu điểm giao ${index + 1}`),
        service_time:
          parseRequiredNumber(
            pt.serviceMinutes,
            `thời gian phục vụ điểm giao ${index + 1}`
          ) * 60,
        time_window: timeWindow,
        snap_confirmed: pt.snapConfirmed,
      }
    })

    if (!customers.length) {
      throw new Error("Cần thêm ít nhất một điểm giao")
    }

    return {
      depot: {
        name: depotDraft.name.trim() || "Kho",
        address: depotDraft.address.trim(),
        lat: parseRequiredNumber(depotDraft.lat, "vĩ độ kho"),
        lon: parseRequiredNumber(depotDraft.lon, "kinh độ kho"),
        snap_confirmed: depotDraft.snapConfirmed,
      },
      num_vehicles: Number(manualVehicles),
      vehicle_capacity: Number(manualCapacity),
      customers,
    }
  }

  const handleManualModeChange = async (enabled: boolean) => {
    setManualMode(enabled)
    let nextDataset = dataset

    if (enabled && !nextDataset.startsWith("hcmc")) {
      const hcmc = catalog.find(
        (item) => item.id.startsWith("hcmc") && item.missing.length === 0
      )
      if (hcmc) {
        nextDataset = hcmc.id
        setDataset(hcmc.id)
      }
    }

    if (enabled && customPoints.length === 0) {
      setCustomPoints([createPointDraft()])
    }

    if (nextDataset) {
      try {
        await previewDataset(nextDataset, catalog, enabled)
      } catch (err) {
        setStatus(err instanceof Error ? err.message : "Lỗi nạp dataset", true)
      }
    }
  }

  const handleDatasetChange = async (nextDataset: string) => {
    setDataset(nextDataset)
    try {
      await previewDataset(nextDataset, catalog, manualMode)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Lỗi nạp dataset", true)
    }
  }

  const handleDepotAddressInput = (address: string) => {
    setDepot((prev) => {
      if (prev.resolvedAddress && address.trim() !== prev.resolvedAddress) {
        return {
          ...prev,
          address,
          resolvedAddress: undefined,
          snapConfirmed: false,
          snapCheckedLat: undefined,
          snapCheckedLon: undefined,
          lat: "",
          lon: "",
          note: "Địa chỉ đã thay đổi, cần tìm lại.",
        }
      }
      return { ...prev, address }
    })
  }

  const handleDepotCoordInput = (field: "lat" | "lon", value: string) => {
    setDepot((prev) => ({
      ...prev,
      [field]: value,
      snapConfirmed: false,
      snapCheckedLat: undefined,
      snapCheckedLon: undefined,
    }))
  }

  const handlePointAddressInput = (uid: string, address: string) => {
    setCustomPoints((prev) =>
      prev.map((pt) => {
        if (pt.uid !== uid) return pt
        if (pt.resolvedAddress && address.trim() !== pt.resolvedAddress) {
          return {
            ...pt,
            address,
            resolvedAddress: undefined,
            snapConfirmed: false,
            snapCheckedLat: undefined,
            snapCheckedLon: undefined,
            lat: "",
            lon: "",
            note: "Địa chỉ đã thay đổi, cần tìm lại.",
          }
        }
        return { ...pt, address }
      })
    )
  }

  const handlePointCoordInput = (
    uid: string,
    field: "lat" | "lon",
    value: string
  ) => {
    setCustomPoints((prev) =>
      prev.map((pt) =>
        pt.uid === uid
          ? {
              ...pt,
              [field]: value,
              snapConfirmed: false,
              snapCheckedLat: undefined,
              snapCheckedLon: undefined,
            }
          : pt
      )
    )
  }

  const handleMapPick = async (lat: number, lon: number): Promise<boolean> => {
    const target = pendingPickRef.current
    if (!target) return false

    try {
      setStatus("Đang kiểm tra vị trí đã chọn với road graph…")
      const snap = await checkPointSnap(lat, lon)
      const candidate: GeocodeCandidate = {
        ...snap,
        lat,
        lon,
        display_name: "Vị trí chọn trên bản đồ",
        provider: "Bản đồ",
        house_match: "exact",
      }

      if (target.kind === "depot") {
        const updated = applyCandidateToDraft(
          depotRef.current,
          candidate,
          depotRef.current.address.trim(),
          depotRef.current.candidates
        )
        if (!updated) return false
        setDepot(updated)
      } else if (target.uid) {
        const currentPt = customPointsRef.current.find(
          (p) => p.uid === target.uid
        )
        if (!currentPt) return false
        const updated = applyCandidateToDraft(
          currentPt,
          candidate,
          currentPt.address.trim(),
          currentPt.candidates
        )
        if (!updated) return false
        setCustomPoints((prev) =>
          prev.map((item) => (item.uid === target.uid ? updated : item))
        )
      }

      setPendingPick(null)
      setStatus(
        `Đã chọn vị trí · cách đường gần nhất ${fmt(snap.snap_distance_m)} m`
      )
      return true
    } catch (err) {
      setStatus(
        err instanceof Error ? err.message : "Không kiểm tra được vị trí",
        true
      )
      return false
    }
  }

  const handlePreviewCustomPoints = async () => {
    try {
      setStatus("Đang tìm địa chỉ…")
      const resolved = await resolveAllAddresses()
      const custom = collectCustomPoints(resolved.depot, resolved.customers)
      const data = await api<{
        points: MapPoint[]
        customers: number
        vehicles: number
        capacity: number
      }>("/api/custom-points/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          trip_name: tripName.trim(),
          custom_points: custom,
        }),
      })
      setPoints(data.points || [])
      setStatus(
        `Đã kiểm tra ${data.customers} điểm giao · ${data.vehicles} xe · tải trọng ${data.capacity}`
      )
      setMobileMenuOpen(false)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Lỗi xem điểm", true)
    }
  }

  // ---------------------------------------------------------------------------
  // Optimization, Trips & Driver Actions
  // ---------------------------------------------------------------------------
  const handleStartOptimization = async () => {
    const depSeconds = timeSeconds(earliestDeparture || "08:00") ?? 8 * 3600
    const latestDepSeconds = latestDeparture ? (timeSeconds(latestDeparture) ?? undefined) : undefined
    if (latestDepSeconds !== undefined && latestDepSeconds < depSeconds) {
      setStatus("Giờ xuất phát muộn nhất phải lớn hơn hoặc bằng giờ sớm nhất", true)
      toast.error("Khung giờ xuất phát không hợp lệ", {
        description: "Giờ xuất phát muộn nhất phải sau hoặc bằng giờ xuất phát sớm nhất.",
      })
      return
    }

    const config: OptimizationConfig = {
      trip_name: tripName.trim(),
      dataset,
      algorithms: selectedAlgorithms,
      population_size: Number(populationSize),
      budget: Number(budget),
      seed: Number(seed),
      num_seeds: Number(numSeeds),
      earliest_departure: depSeconds,
      ...(latestDepSeconds !== undefined ? { latest_departure: latestDepSeconds } : {}),
    }

    try {
      if (manualMode) {
        setStatus("Đang tìm địa chỉ…")
        const resolved = await resolveAllAddresses()
        config.custom_points = collectCustomPoints(
          resolved.depot,
          resolved.customers
        )
      }
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Lỗi địa chỉ", true)
      return
    }

    if (!config.trip_name) {
      setStatus("Nhập tên chuyến", true)
      return
    }
    if (!config.algorithms.length) {
      setStatus("Chọn ít nhất một thuật toán", true)
      return
    }
    if (config.budget < config.population_size) {
      setStatus("NFE phải lớn hơn hoặc bằng P", true)
      return
    }

    clearResult()
    setBusy(true)
    setStartedAt(Date.now())
    setStatus("Đang tối ưu các route…")
    setMobileMenuOpen(false)
    toast.info("Bắt đầu tối ưu chuyến...", {
      description: `Ngân sách ${config.budget} NFE · Quần thể ${config.population_size} cá thể`,
    })

    try {
      const job = await api<{ id: string }>("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      })

      setCurrentJobId(job.id)
      try {
        localStorage.setItem("rapo-map-job", job.id)
      } catch {
        // ignore
      }

      await pollJob(job.id)
    } catch (err) {
      setBusy(false)
      setStartedAt(0)
      const msg = err instanceof Error ? err.message : "Lỗi khởi chạy tối ưu"
      setStatus(msg, true)
      toast.error("Không thể khởi chạy tối ưu", { description: msg })
    }
  }

  const handleAssignRoute = async (
    algorithm: string,
    routeIndex: number,
    driverId: number
  ) => {
    if (!driverId || !currentJobId) {
      setStatus("Chọn Driver cho route", true)
      toast.error("Vui lòng chọn tài xế cho tuyến")
      return
    }
    try {
      await api("/api/trips", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: currentJobId,
          algorithm,
          route_index: routeIndex,
          driver_id: driverId,
        }),
      })
      await reloadTrips()
      const assignedDriver = activeDrivers.find((d) => d.id === driverId)
      const dName = assignedDriver
        ? driverDisplayName(assignedDriver.username, assignedDriver.full_name)
        : `Tài xế #${driverId}`
      toast.success(`Đã giao Xe ${routeIndex + 1} cho ${dName}`, {
        description: `Chuyến: ${tripName || "Lộ trình tối ưu"}`,
      })
      setStatus(`Đã giao route xe ${routeIndex + 1} cho ${dName}`, false)
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Lỗi giao tuyến"
      setStatus(msg, true)
      toast.error("Không thể giao tuyến", { description: msg })
    }
  }

  const handleOpenTrip = async (trip: Trip) => {
    if (!trip || !trip.id) {
      setStatus("Không tìm thấy thông tin chuyến hợp lệ để mở", true)
      return
    }
    try {
      const data = await api<{
        trip: Trip
        stops: TripStop[]
        payload: OptimizationPayload
      }>(`/api/trips/${trip.id}`)
      setCurrentJobId(trip.job_id)
      setCurrentTrip(data.trip)
      setShowStopsPanel(true)
      applyResultPayload(data.payload, data.stops || [])
      setStatus(`Đang xem route xe ${trip.route_index + 1} · ${trip.driver_username}`)
      setMobileMenuOpen(false)
      await reloadTrips()
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Lỗi mở chuyến", true)
    }
  }

  const handleStartTrip = async (trip: Trip) => {
    if (!trip || !trip.id) {
      setStatus("Không tìm thấy thông tin chuyến để cập nhật", true)
      return
    }
    try {
      await api(`/api/trips/${trip.id}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "in_progress" }),
      })
      await handleOpenTrip({ ...trip, status: "in_progress" })
      setStatus("Đã bắt đầu chuyến", false)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Lỗi bắt đầu chuyến", true)
    }
  }

  const handleCompleteStop = async (stopIndex: number): Promise<boolean> => {
    const activeTrip = currentTripRef.current
    if (!activeTrip || !activeTrip.id) return false

    try {
      const data = await api<{ trip: Trip; stops: TripStop[] }>(
        `/api/trips/${activeTrip.id}/stops/${stopIndex}`,
        {
          method: "PATCH",
          timeoutMs: 10000,
        }
      )
      setCurrentTrip(data.trip)
      setDeliveryStatuses(data.stops || [])
      if (data.trip.status === "completed") {
        toast.success("Đã hoàn thành toàn bộ chuyến giao hàng! 🎉", {
          description: "Tất cả các điểm dừng đã được giao thành công.",
        })
      } else {
        toast.success(`Đã hoàn thành điểm giao #${stopIndex + 1}`, {
          description: "Tiến độ đã được đồng bộ trực tiếp lên hệ thống.",
        })
      }
      setStatus(
        data.trip.status === "completed"
          ? "Đã hoàn thành toàn bộ chuyến"
          : "Đã hoàn thành điểm giao",
        false
      )
      await reloadTrips()
      return true
    } catch (err) {
      try {
        const latest = await api<{ trip: Trip; stops: TripStop[] }>(
          `/api/trips/${activeTrip.id}`,
          { timeoutMs: 5000 }
        )
        const target = latest.stops.find((s) => s.stop_index === stopIndex)
        if (target?.status === "completed") {
          setCurrentTrip(latest.trip)
          setDeliveryStatuses(latest.stops || [])
          setStatus("Điểm giao đã được ghi nhận hoàn thành", false)
          await reloadTrips()
          return true
        }
      } catch {
        // ignore secondary check error
      }
      setStatus(
        err instanceof Error ? err.message : "Lỗi cập nhật điểm giao",
        true
      )
      return false
    }
  }

  const handleViewAllJobRoutes = async (jobId: string, focusIndex?: number) => {
    try {
      const data = await api<RunJobResponse>(`/api/runs/${jobId}`)
      if (!data.payload) {
        throw new Error("Không tìm thấy dữ liệu toàn bộ chuyến")
      }
      setCurrentJobId(jobId)
      setCurrentTrip(null)
      setShowStopsPanel(false)
      applyResultPayload(data.payload, data.stops || [])
      const primaryAlg = Object.keys(data.payload.results || {})[0]
      const totalVehicles =
        data.payload.results?.[primaryAlg]?.routes?.length || ""
      setStatus(
        `Đang xem toàn bộ chuyến (${totalVehicles} xe) · Mã ${jobId.slice(0, 8)}`
      )
      setMobileMenuOpen(false)
      if (focusIndex != null) {
        window.setTimeout(() => mapRef.current?.focusRoute(focusIndex), 150)
      }
      await reloadTrips()
    } catch (err) {
      setStatus(
        err instanceof Error ? err.message : "Lỗi tải dữ liệu toàn bộ chuyến",
        true
      )
    }
  }

  // Derived data for components
  const availableAlgorithms = useMemo(
    () => catalog.find((ds) => ds.id === dataset)?.algorithms || [],
    [catalog, dataset]
  )

  const resultAlgorithmNames = useMemo(
    () => (payload ? Object.keys(payload.results || {}) : []),
    [payload]
  )

  const activeRoutes = useMemo(() => {
    if (!payload || !selectedResultAlg) return []
    return payload.results[selectedResultAlg]?.routes ?? []
  }, [payload, selectedResultAlg])

  const totalRoutesCount = useMemo(() => {
    let count = 0
    for (const job of adminJobs) {
      if (
        Array.isArray(job.plan.routes_summary) &&
        job.plan.routes_summary.length > 0
      ) {
        count += job.plan.routes_summary.length
      } else if (job.plan.total_routes) {
        count += job.plan.total_routes
      } else {
        count += job.trips.length
      }
    }
    return count || allTrips.length
  }, [adminJobs, allTrips])

  if (!authChecked) {
    return (
      <div className="flex min-h-svh items-center justify-center bg-background text-sm text-muted-foreground">
        Đang kiểm tra phiên đăng nhập…
      </div>
    )
  }

  if (!user) {
    return (
      <>
        <LoginScreen
          initialError={authError}
          dark={dark}
          onToggleTheme={toggleTheme}
          onSubmit={handleLogin}
        />
        <Toaster />
      </>
    )
  }

  const sidebarContent = (
    <DispatchSidebar
      user={user}
      busy={busy}
      drivers={activeDrivers}
      catalog={catalog}
      tripName={tripName}
      dataset={dataset}
      manualMode={manualMode}
      manualVehicles={manualVehicles}
      manualCapacity={manualCapacity}
      depot={depot}
      customPoints={customPoints}
      pendingPick={pendingPick}
      availableAlgorithms={availableAlgorithms}
      selectedAlgorithms={selectedAlgorithms}
      populationSize={populationSize}
      budget={budget}
      seed={seed}
      numSeeds={numSeeds}
      earliestDeparture={earliestDeparture}
      latestDeparture={latestDeparture}
      allTrips={allTrips}
      adminJobs={adminJobs}
      currentJobId={currentJobId}
      currentTrip={currentTrip}
      statusMsg={statusMsg}
      onTripNameChange={setTripName}
      onDatasetChange={(val) => void handleDatasetChange(val)}
      onManualModeChange={(enabled) => void handleManualModeChange(enabled)}
      onManualVehiclesChange={setManualVehicles}
      onManualCapacityChange={setManualCapacity}
      onDepotChange={(patch) => setDepot((prev) => ({ ...prev, ...patch }))}
      onDepotAddressInput={handleDepotAddressInput}
      onDepotCoordInput={handleDepotCoordInput}
      onResolveDepot={() =>
        void resolveDepot().catch((err) =>
          setStatus(err instanceof Error ? err.message : "Lỗi tìm địa chỉ", true)
        )
      }
      onSelectDepotCandidate={(candidate) => {
        const updated = applyCandidateToDraft(
          depotRef.current,
          candidate,
          depotRef.current.address.trim(),
          depotRef.current.candidates
        )
        if (updated) setDepot(updated)
      }}
      onPickDepotOnMap={() => {
        setPendingPick({ kind: "depot", label: "Kho xuất phát" })
        setStatus("Bấm chuột vào đúng vị trí kho trên bản đồ")
        setMobileMenuOpen(false)
      }}
      onAddPoint={() => setCustomPoints((prev) => [...prev, createPointDraft()])}
      onRemovePoint={(uid) =>
        setCustomPoints((prev) => prev.filter((item) => item.uid !== uid))
      }
      onPointChange={(uid, patch) =>
        setCustomPoints((prev) =>
          prev.map((item) => (item.uid === uid ? { ...item, ...patch } : item))
        )
      }
      onPointAddressInput={handlePointAddressInput}
      onPointCoordInput={handlePointCoordInput}
      onResolvePoint={(uid) =>
        void resolvePointByUid(uid).catch((err) =>
          setStatus(err instanceof Error ? err.message : "Lỗi tìm địa chỉ", true)
        )
      }
      onSelectPointCandidate={(uid, candidate) => {
        const target = customPointsRef.current.find((p) => p.uid === uid)
        if (!target) return
        const updated = applyCandidateToDraft(
          target,
          candidate,
          target.address.trim(),
          target.candidates
        )
        if (updated) {
          setCustomPoints((prev) =>
            prev.map((item) => (item.uid === uid ? updated : item))
          )
        }
      }}
      onPickPointOnMap={(uid, index) => {
        setPendingPick({
          kind: "customer",
          uid,
          label: `Điểm giao ${index + 1}`,
        })
        setStatus("Bấm chuột vào đúng vị trí giao hàng trên bản đồ")
        setMobileMenuOpen(false)
      }}
      onPreviewPoints={() => void handlePreviewCustomPoints()}
      onToggleAlgorithm={(name) =>
        setSelectedAlgorithms((prev) =>
          prev.includes(name)
            ? prev.filter((item) => item !== name)
            : [...prev, name]
        )
      }
      onPopulationChange={setPopulationSize}
      onBudgetChange={setBudget}
      onSeedChange={setSeed}
      onNumSeedsChange={setNumSeeds}
      onEarliestDepartureChange={setEarliestDeparture}
      onLatestDepartureChange={setLatestDeparture}
      onStartOptimization={() => void handleStartOptimization()}
      onCreateDriver={handleCreateDriver}
      onOpenDriversPage={() => {
        setActiveView("drivers")
        setMobileMenuOpen(false)
      }}
      onOpenTripsPage={() => {
        setActiveView("trips")
        setMobileMenuOpen(false)
      }}
      onOpenTrip={(trip) => void handleOpenTrip(trip)}
      onStartTrip={(trip) => void handleStartTrip(trip)}
      onViewAllJobRoutes={(jobId, focusIdx) =>
        void handleViewAllJobRoutes(jobId, focusIdx)
      }
    />
  )

  return (
    <div className="flex min-h-svh flex-col bg-background text-foreground">
      <HeaderBar
        user={user}
        dark={dark}
        clockText={clockText}
        busy={busy}
        activeView={activeView}
        routesCount={totalRoutesCount}
        driverCount={drivers.length}
        liveTripCount={inProgressTripsCount}
        onChangeView={setActiveView}
        onToggleTheme={toggleTheme}
        onLogout={() => void handleLogout()}
        onOpenMobileMenu={() => setMobileMenuOpen(true)}
      />

      {/* View: Trips / Routes Management (Admin only) */}
      {user.role === "admin" && activeView === "trips" ? (
        <main className="min-h-0 flex-1 bg-muted/10">
          <TripsManagementPage
            adminJobs={adminJobs}
            activeDrivers={activeDrivers}
            loading={tripsLoading}
            onBackToMap={() => setActiveView("map")}
            onRefresh={async () => {
              setTripsLoading(true)
              try {
                await Promise.all([reloadTrips(), loadDrivers()])
              } finally {
                setTripsLoading(false)
              }
            }}
            onOpenTripOnMap={async (trip) => {
              setActiveView("map")
              await handleOpenTrip(trip)
            }}
            onViewRouteOnMap={async (jobId, routeIdx) => {
              setActiveView("map")
              await handleViewAllJobRoutes(jobId, routeIdx)
            }}
            onAssignRoute={async (jobId, algorithm, routeIndex, driverId) => {
              try {
                await api("/api/trips", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    job_id: jobId,
                    algorithm,
                    route_index: routeIndex,
                    driver_id: driverId,
                  }),
                })
                await Promise.all([reloadTrips(), loadDrivers()])
                return { ok: true }
              } catch (err) {
                return {
                  ok: false,
                  error:
                    err instanceof Error ? err.message : "Lỗi giao tuyến",
                }
              }
            }}
            onReassignDriver={async (tripId, driverId, row) => {
              try {
                try {
                  await api(`/api/trips/${tripId}/driver`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ driver_id: driverId }),
                  })
                } catch (patchErr) {
                  if (row) {
                    await api("/api/trips", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        job_id: row.jobId,
                        algorithm: row.algorithm,
                        route_index: row.routeIndex,
                        driver_id: driverId,
                      }),
                    })
                  } else {
                    throw patchErr
                  }
                }
                await Promise.all([reloadTrips(), loadDrivers()])
                return { ok: true }
              } catch (err) {
                return {
                  ok: false,
                  error:
                    err instanceof Error ? err.message : "Lỗi đổi tài xế",
                }
              }
            }}
            onCancelTrip={async (tripId) => {
              try {
                await api(`/api/trips/${tripId}/status`, {
                  method: "PATCH",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ status: "cancelled" }),
                })
                await Promise.all([reloadTrips(), loadDrivers()])
                return { ok: true }
              } catch (err) {
                return {
                  ok: false,
                  error:
                    err instanceof Error ? err.message : "Lỗi huỷ chuyến",
                }
              }
            }}
          />
        </main>
      ) : user.role === "admin" && activeView === "drivers" ? (
        <main className="min-h-0 flex-1 bg-muted/10">
          <DriversManagementPage
            drivers={drivers}
            loading={driversLoading}
            onBackToMap={() => setActiveView("map")}
            onRefresh={loadDrivers}
            onCreateDriver={handleCreateDriverWithFeedback}
            onUpdateDriver={handleUpdateDriver}
            onDeleteDriver={handleDeleteDriver}
          />
        </main>
      ) : (
        <>
          {/* Mobile Sidebar Sheet */}
          <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
            <SheetContent side="left" className="w-[90vw] max-w-sm p-0">
              <SheetHeader className="border-b border-border px-4 py-3">
                <SheetTitle>Điều phối & Chuyến giao hàng</SheetTitle>
                <SheetDescription>
                  {user.role === "admin"
                    ? "Cấu hình tối ưu, quản lý điểm giao và phân công Driver."
                    : "Danh sách chuyến giao hàng được giao cho bạn."}
                </SheetDescription>
              </SheetHeader>
              <ScrollArea className="h-[calc(100svh-5rem)]">
                {sidebarContent}
              </ScrollArea>
            </SheetContent>
          </Sheet>

          {/* Main Split Workspace */}
          <div className="flex min-h-0 flex-1">
            {/* Desktop Sidebar */}
            <aside className="hidden w-[380px] shrink-0 border-r border-border bg-muted/15 lg:block xl:w-[410px]">
              <ScrollArea className="h-[calc(100svh-3.5rem)]">
                {sidebarContent}
              </ScrollArea>
            </aside>

            {/* Right Workspace: Map + Results */}
            <main className="min-w-0 flex-1 bg-muted/10">
              <div className="mx-auto flex max-w-[1600px] flex-col gap-4 p-3 sm:p-4">
                <MapPanel
                  ref={mapRef}
                  points={points}
                  routes={activeRoutes}
                  algorithmNames={resultAlgorithmNames}
                  selectedAlgorithm={selectedResultAlg}
                  pendingPick={pendingPick}
                  onAlgorithmChange={setSelectedResultAlg}
                  onCancelMapPick={() => {
                    setPendingPick(null)
                    setStatus("Đã hủy chọn vị trí trên bản đồ")
                  }}
                  onMapPick={handleMapPick}
                />

                <ResultsWorkspace
                  user={user}
                  points={points}
                  payload={payload}
                  capacity={manualMode ? Number(manualCapacity) || 100 : datasetCapacity}
                  selectedAlgorithm={selectedResultAlg}
                  drivers={activeDrivers}
                  allTrips={allTrips}
                  currentJobId={currentJobId}
                  currentTrip={currentTrip}
                  deliveryStatuses={deliveryStatuses}
                  showStopsPanel={showStopsPanel}
                  onFocusRoute={(index) => mapRef.current?.focusRoute(index)}
                  onFitMap={() => mapRef.current?.fit()}
                  onExportJSON={() => {
                    if (payload) exportPayloadJSON(payload)
                  }}
                  onExportCSV={() => {
                    if (payload && selectedResultAlg) {
                      exportAlgorithmCSV(payload, selectedResultAlg)
                    }
                  }}
                  onAssignRoute={handleAssignRoute}
                  onCompleteStop={handleCompleteStop}
                />
              </div>
            </main>
          </div>
        </>
      )}
      <Toaster />
    </div>
  )
}

export default App
