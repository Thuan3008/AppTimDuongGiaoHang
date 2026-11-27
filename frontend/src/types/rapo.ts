export type UserRole = "admin" | "driver"
export type AppView = "map" | "trips" | "drivers"

export interface User {
  id: number
  username: string
  full_name?: string
  role: UserRole
}

export interface Driver {
  id: number
  username: string
  full_name: string
  role?: UserRole
  active: boolean
  created_at?: string
  trips_total?: number
  trips_active?: number
  trips_completed?: number
}

export interface DatasetEntry {
  id: string
  name: string
  missing: string[]
  algorithms: string[]
}

export interface MapPoint {
  id: number
  name?: string
  address?: string
  lat: number
  lon: number
  demand: number
  depot?: boolean
  snap_distance_m?: number
}

export type SnapStatus = "auto" | "confirm" | "rejected"
export type HouseMatch = "exact" | "missing" | "wrong" | "not_requested"

export interface GeocodeCandidate {
  lat: number
  lon: number
  display_name: string
  provider: string
  house_match: HouseMatch | string
  road_node_id?: number
  road_lat?: number
  road_lon?: number
  snap_distance_m: number
  snap_status: SnapStatus | string
}

export interface GeocodeResponse {
  result: GeocodeCandidate
  candidates: GeocodeCandidate[]
  requires_confirmation: boolean
  warning?: string | null
}

export interface SnapPointResponse {
  lat: number
  lon: number
  road_node_id: number
  road_lat: number
  road_lon: number
  snap_distance_m: number
  snap_status: SnapStatus | string
}

export interface CustomDepotPayload {
  name: string
  address: string
  lat: number
  lon: number
  snap_confirmed?: boolean
  snap_status?: string
}

export interface CustomCustomerPayload {
  name: string
  address: string
  lat: number
  lon: number
  demand: number
  service_time: number
  time_window: [number, number] | null
  snap_confirmed?: boolean
  snap_status?: string
}

export interface CustomPointsPayload {
  depot: CustomDepotPayload
  num_vehicles: number
  vehicle_capacity: number
  customers: CustomCustomerPayload[]
}

export interface CustomPointDraft {
  uid: string
  name: string
  address: string
  lat: string
  lon: string
  demand: string
  serviceMinutes: string
  windowStart: string
  windowEnd: string
  resolvedAddress?: string
  snapConfirmed: boolean
  snapStatus?: SnapStatus | string
  snapDistanceM?: number | null
  snapCheckedLat?: string
  snapCheckedLon?: string
  note: string
  warning: string | null
  candidates: GeocodeCandidate[]
  selectedCandidate: GeocodeCandidate | null
  resolving?: boolean
}

export interface DepotDraft {
  name: string
  address: string
  lat: string
  lon: string
  resolvedAddress?: string
  snapConfirmed: boolean
  snapStatus?: SnapStatus | string
  snapDistanceM?: number | null
  snapCheckedLat?: string
  snapCheckedLon?: string
  note: string
  warning: string | null
  candidates: GeocodeCandidate[]
  selectedCandidate: GeocodeCandidate | null
  resolving?: boolean
}

export interface CsvParsedPoint {
  name: string
  address: string
  lat: number | null
  lon: number | null
  snap_distance_m: number | null
  snap_status: SnapStatus | string | null
  warning: string | null
  demand: number
  service_time: number
  time_window: [number, number] | null
  needs_geocode: boolean
}

export interface CsvParseResponse {
  dataset: string
  depot: CsvParsedPoint
  customers: CsvParsedPoint[]
  vehicles: number | null
  capacity: number | null
  geocode_count: number
  snapped_count: number
}

export interface OptimizationConfig {
  trip_name: string
  dataset: string
  algorithms: string[]
  population_size: number
  budget: number
  seed: number
  num_seeds: number
  earliest_departure: number
  latest_departure?: number
  custom_points?: CustomPointsPayload
}

export interface RouteStop {
  customer_id: number
  demand?: number
  arrival_time: number
  service_start_time?: number
  service_time: number
  departure_time: number
  waiting_time?: number
  road_node_id?: number
  cumulative_load?: number
}

export interface RouteLeg {
  from_customer_id?: number
  to_customer_id: number
  travel_time: number
}

export interface RouteResult {
  vehicle: number
  visits: number[]
  load: number
  departure_time: number
  end_time: number
  travel_time: number
  service_time: number
  duration: number
  objective: number
  edge_path?: (string | number)[]
  geometry: GeoJSON.GeoJsonObject
  geometry_source: "osrm" | "dataset-road-graph" | "straight-line" | string
  warning?: string
  stops?: RouteStop[]
  legs?: RouteLeg[]
}

export interface AlgorithmResult {
  seed?: number | null
  fitness: number
  nfe: number
  runtime_seconds: number
  departure_time_mode: string
  routes: RouteResult[]
}

export interface AlgorithmSummary {
  completed: number
  failed: number
  best?: number | null
  mean?: number | null
  median?: number | null
  sd?: number | null
  runtime_mean_seconds?: number | null
  best_seed?: number | null
}

export interface OptimizationPayload {
  config: OptimizationConfig
  instance?: string
  capacity?: number
  points: MapPoint[]
  results: Record<string, AlgorithmResult>
  runs?: Record<string, AlgorithmResult[]>
  summary?: Record<string, AlgorithmSummary>
  errors?: Record<string, string>
  elapsed_seconds?: number
  metric_source?: string
}

export interface TripStop {
  stop_index: number
  customer_id: number
  status: "pending" | "completed" | string
  completed_at?: string | null
  route_index?: number
}

export interface PlanRouteSummary {
  route_index: number
  vehicle: number
  stops_count: number
  load: number
  duration: number
  deleted_at?: string | null
}

export interface PlanMetadata {
  trip_name?: string
  instance?: string
  dataset?: string
  algorithm?: string
  departure_time?: number
  latest_departure?: number
  total_routes?: number
  routes_summary?: PlanRouteSummary[]
}

export type TripStatus = "assigned" | "in_progress" | "completed" | "cancelled"

export interface Trip {
  id: string
  job_id: string
  algorithm: string
  route_index: number
  driver_id: number
  driver_username: string
  driver_full_name?: string
  status: TripStatus | string
  stops_completed: number
  stops_total: number
  job_status?: string
  deleted_at?: string | null
  plan?: PlanMetadata
}

export interface AdminJobGroup {
  job_id: string
  deleted_at?: string | null
  job_deleted?: boolean
  plan: PlanMetadata
  trips: Trip[]
}

export interface TripsResponse {
  trips: Trip[]
  jobs: AdminJobGroup[] | null
  deleted_jobs?: AdminJobGroup[] | null
  deleted_routes_count?: number
}

export interface RunJobResponse {
  id: string
  config?: OptimizationConfig
  status: "queued" | "running" | "completed" | "failed" | string
  stage: string
  payload?: OptimizationPayload
  stops?: TripStop[]
  deleted?: boolean
}

export interface MapPickTarget {
  kind: "depot" | "customer"
  uid?: string
  label: string
}
