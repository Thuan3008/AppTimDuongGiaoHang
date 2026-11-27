import { useRef } from "react"
import {
  CheckCheckIcon,
  CrosshairIcon,
  DownloadIcon,
  EyeIcon,
  FileSpreadsheetIcon,
  Loader2Icon,
  MapPinIcon,
  PlusIcon,
  SearchIcon,
  Trash2Icon,
  UploadIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { fmt } from "@/lib/api"
import type {
  CustomPointDraft,
  DepotDraft,
  GeocodeCandidate,
  MapPickTarget,
} from "@/types/rapo"

const HOUSE_LABELS: Record<string, string> = {
  exact: "khớp số nhà",
  prefix: "khớp đầu hẻm / gần số nhà",
  missing: "thiếu số nhà",
  wrong: "sai số nhà",
  not_requested: "không có số nhà",
}

const SNAP_LABELS: Record<string, string> = {
  auto: "tự động chấp nhận",
  confirm: "cần xác nhận",
  rejected: "không thể dùng",
}

interface CandidateListProps {
  warning: string | null
  note: string
  candidates: GeocodeCandidate[]
  disabled: boolean
  isPickingThis: boolean
  onSelectCandidate: (candidate: GeocodeCandidate) => void
  onPickOnMap: () => void
}

function GeocodeFeedback({
  warning,
  note,
  candidates,
  disabled,
  isPickingThis,
  onSelectCandidate,
  onPickOnMap,
}: CandidateListProps) {
  return (
    <div className="flex flex-col gap-1.5 text-xs">
      {warning ? (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5 text-amber-700 dark:text-amber-300">
          {warning} Hãy chọn một kết quả bên dưới hoặc nhập tọa độ nâng cao.
        </div>
      ) : (
        <p className="text-muted-foreground">{note}</p>
      )}

      {(warning || isPickingThis) && (
        <Button
          type="button"
          variant={isPickingThis ? "default" : "outline"}
          size="xs"
          disabled={disabled}
          onClick={onPickOnMap}
          className="w-fit"
        >
          <CrosshairIcon data-icon="inline-start" />
          {isPickingThis
            ? "Đang chọn trên bản đồ… (bấm vào bản đồ)"
            : "Chọn vị trí trực tiếp trên bản đồ"}
        </Button>
      )}

      {candidates.length > 0 && (
        <details
          className="rounded-md border border-border bg-muted/40 p-2"
          open={Boolean(warning)}
        >
          <summary className="cursor-pointer font-medium text-foreground select-none">
            {warning ? "Chọn" : "Xem"} kết quả gợi ý ({candidates.length})
          </summary>
          <div className="mt-2 flex flex-col gap-1.5">
            {candidates.map((candidate, idx) => {
              const rejected = candidate.snap_status === "rejected"
              return (
                <button
                  key={`${candidate.lat}-${candidate.lon}-${idx}`}
                  type="button"
                  disabled={disabled || rejected}
                  onClick={() => onSelectCandidate(candidate)}
                  className="rounded-md border border-border bg-background px-2 py-1.5 text-left text-[11px] leading-relaxed transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <div className="font-medium text-foreground">
                    {candidate.display_name}
                  </div>
                  <div className="text-muted-foreground">
                    {candidate.provider} ·{" "}
                    {HOUSE_LABELS[candidate.house_match] || candidate.house_match}{" "}
                    · cách đường {fmt(candidate.snap_distance_m)} m (
                    {SNAP_LABELS[candidate.snap_status] || candidate.snap_status}) ·{" "}
                    {Number(candidate.lat).toFixed(6)},{" "}
                    {Number(candidate.lon).toFixed(6)}
                  </div>
                </button>
              )
            })}
          </div>
        </details>
      )}
    </div>
  )
}

interface ManualPointsEditorProps {
  disabled: boolean
  csvMode?: boolean
  csvImporting: boolean
  csvFileName: string | null
  csvSummary: string | null
  manualVehicles: string
  manualCapacity: string
  manualServiceMinutes: string
  depot: DepotDraft
  points: CustomPointDraft[]
  pendingPick: MapPickTarget | null
  onVehiclesChange: (value: string) => void
  onCapacityChange: (value: string) => void
  onServiceMinutesChange: (value: string) => void
  onDepotChange: (patch: Partial<DepotDraft>) => void
  onDepotAddressInput: (address: string) => void
  onDepotCoordInput: (field: "lat" | "lon", value: string) => void
  onResolveDepot: () => void
  onAutoResolveDepot?: () => void
  onSelectDepotCandidate: (candidate: GeocodeCandidate) => void
  onPickDepotOnMap: () => void
  onAddPoint: () => void
  onRemovePoint: (uid: string) => void
  onPointChange: (uid: string, patch: Partial<CustomPointDraft>) => void
  onPointAddressInput: (uid: string, address: string) => void
  onPointCoordInput: (uid: string, field: "lat" | "lon", value: string) => void
  onResolvePoint: (uid: string) => void
  onAutoResolvePoint?: (uid: string) => void
  onSelectPointCandidate: (uid: string, candidate: GeocodeCandidate) => void
  onPickPointOnMap: (uid: string, index: number) => void
  onPreviewPoints: () => void
  onImportCsv: (file: File) => void
  onDownloadCsvTemplate: () => void
  onConfirmAllSuggested: () => void
}

export function ManualPointsEditor({
  disabled,
  csvMode = false,
  csvImporting,
  csvFileName,
  csvSummary,
  manualVehicles,
  manualCapacity,
  manualServiceMinutes,
  depot,
  points,
  pendingPick,
  onVehiclesChange,
  onCapacityChange,
  onServiceMinutesChange,
  onDepotChange,
  onDepotAddressInput,
  onDepotCoordInput,
  onResolveDepot,
  onAutoResolveDepot,
  onSelectDepotCandidate,
  onPickDepotOnMap,
  onAddPoint,
  onRemovePoint,
  onPointChange,
  onPointAddressInput,
  onPointCoordInput,
  onResolvePoint,
  onAutoResolvePoint,
  onSelectPointCandidate,
  onPickPointOnMap,
  onPreviewPoints,
  onImportCsv,
  onDownloadCsvTemplate,
  onConfirmAllSuggested,
}: ManualPointsEditorProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const unconfirmedSuggestedCount = [depot, ...points].filter(
    (item) =>
      Boolean(item.warning) &&
      !item.snapConfirmed &&
      item.lat.trim() !== "" &&
      item.lon.trim() !== "" &&
      item.snapStatus !== "rejected"
  ).length

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-muted/30 p-3">
      <div
        className={`flex flex-col gap-2 rounded-lg border p-2.5 ${
          csvMode
            ? "border-primary/50 bg-primary/5"
            : "border-border bg-card"
        }`}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-1.5 text-xs font-semibold">
            <FileSpreadsheetIcon className="size-3.5 text-primary" />
            Nhập từ 1 file CSV
          </span>
          {csvFileName && (
            <Badge variant="secondary" className="max-w-[140px] truncate text-[10px]">
              {csvFileName}
            </Badge>
          )}
        </div>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Cột <code className="font-mono">address</code> (địa chỉ hoặc tên địa điểm) bắt buộc nhập;
          cột <code className="font-mono">lat, lon</code> có thể bỏ trống để hệ thống tự tìm tọa độ và chấm lên bản đồ.
        </p>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            e.target.value = ""
            if (file) onImportCsv(file)
          }}
        />

        <div className="grid grid-cols-2 gap-1.5">
          <Button
            type="button"
            variant="default"
            size="xs"
            disabled={disabled || csvImporting}
            onClick={() => fileInputRef.current?.click()}
          >
            {csvImporting ? (
              <Loader2Icon data-icon="inline-start" className="animate-spin" />
            ) : (
              <UploadIcon data-icon="inline-start" />
            )}
            {csvImporting ? "Đang dò địa chỉ…" : "Chọn file CSV"}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="xs"
            disabled={disabled || csvImporting}
            onClick={onDownloadCsvTemplate}
          >
            <DownloadIcon data-icon="inline-start" />
            Tải CSV mẫu
          </Button>
        </div>

        {csvSummary && (
          <p className="text-[11px] font-medium text-foreground">{csvSummary}</p>
        )}

        {unconfirmedSuggestedCount > 0 && (
          <Button
            type="button"
            variant="secondary"
            size="xs"
            disabled={disabled || csvImporting}
            onClick={onConfirmAllSuggested}
            className="w-full"
          >
            <CheckCheckIcon data-icon="inline-start" />
            Xác nhận tất cả điểm gợi ý ({unconfirmedSuggestedCount})
          </Button>
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        Tọa độ gần road graph TP.HCM sẽ được tự động ghép vào nút giao thông gần nhất. Nhập tên địa điểm hoặc địa chỉ rồi nhấn Enter / rời ô để tự dò và chấm lên bản đồ.
      </p>

      <div className="grid grid-cols-3 gap-2">
        <div className="flex flex-col gap-1">
          <label htmlFor="manual-vehicles" className="text-xs font-medium">
            Số xe
          </label>
          <Input
            id="manual-vehicles"
            type="number"
            min={1}
            max={50}
            value={manualVehicles}
            onChange={(e) => onVehiclesChange(e.target.value)}
            disabled={disabled}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="manual-capacity" className="text-xs font-medium">
            Tải trọng (kg)
          </label>
          <Input
            id="manual-capacity"
            type="number"
            min={0.01}
            step="any"
            value={manualCapacity}
            onChange={(e) => onCapacityChange(e.target.value)}
            disabled={disabled}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="manual-service-time" className="text-xs font-medium" title="Thời gian phục vụ mặc định cho các điểm giao">
            Phục vụ (phút)
          </label>
          <Input
            id="manual-service-time"
            type="number"
            min={0}
            step={0.5}
            placeholder="5"
            value={manualServiceMinutes}
            onChange={(e) => onServiceMinutesChange(e.target.value)}
            disabled={disabled}
            title="Thời gian phục vụ mặc định cho các điểm giao (phút)"
          />
        </div>
      </div>

      {/* Depot Section */}
      <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-1.5 text-xs font-semibold">
            <MapPinIcon className="size-3.5 text-destructive" />
            Kho xuất phát (Depot)
          </span>
          <Badge variant="outline" className="text-[10px]">
            K
          </Badge>
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor="depot-name" className="text-xs text-muted-foreground">
            Tên kho
          </label>
          <Input
            id="depot-name"
            type="text"
            maxLength={100}
            value={depot.name}
            onChange={(e) => onDepotChange({ name: e.target.value })}
            disabled={disabled}
          />
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor="depot-address" className="text-xs text-muted-foreground">
            Địa chỉ kho
          </label>
          <div className="flex gap-1.5">
            <Input
              id="depot-address"
              type="text"
              maxLength={300}
              placeholder="Địa chỉ hoặc tên địa điểm (VD: 140 Lê Trọng Tấn, Tân Phú)"
              value={depot.address}
              onChange={(e) => onDepotAddressInput(e.target.value)}
              onBlur={() => onAutoResolveDepot?.()}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault()
                  onResolveDepot()
                }
              }}
              disabled={disabled || csvImporting}
            />
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={disabled || depot.resolving}
              onClick={onResolveDepot}
            >
              {depot.resolving ? (
                <Loader2Icon data-icon="inline-start" className="animate-spin" />
              ) : (
                <SearchIcon data-icon="inline-start" />
              )}
              Tìm
            </Button>
          </div>
        </div>

        <GeocodeFeedback
          warning={depot.warning}
          note={depot.note}
          candidates={depot.candidates}
          disabled={disabled}
          isPickingThis={pendingPick?.kind === "depot"}
          onSelectCandidate={onSelectDepotCandidate}
          onPickOnMap={onPickDepotOnMap}
        />

        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground select-none">
            Tọa độ nâng cao (tùy chọn)
          </summary>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <div className="flex flex-col gap-1">
              <label htmlFor="depot-lat" className="text-[11px] text-muted-foreground">
                Vĩ độ
              </label>
              <Input
                id="depot-lat"
                type="number"
                step="any"
                placeholder="Tự điền"
                value={depot.lat}
                onChange={(e) => onDepotCoordInput("lat", e.target.value)}
                disabled={disabled}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label htmlFor="depot-lon" className="text-[11px] text-muted-foreground">
                Kinh độ
              </label>
              <Input
                id="depot-lon"
                type="number"
                step="any"
                placeholder="Tự điền"
                value={depot.lon}
                onChange={(e) => onDepotCoordInput("lon", e.target.value)}
                disabled={disabled}
              />
            </div>
          </div>
        </details>
      </div>

      {/* Delivery Points */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold">
            Các điểm giao ({points.length})
          </span>
        </div>

        {points.map((point, index) => (
          <div
            key={point.uid}
            className="flex flex-col gap-2 rounded-lg border border-border bg-card p-2.5"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold">Điểm giao {index + 1}</span>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                disabled={disabled}
                onClick={() => onRemovePoint(point.uid)}
                title="Xóa điểm"
              >
                <Trash2Icon data-icon="inline-start" />
                Xóa
              </Button>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">
                Tên điểm / người nhận
              </label>
              <Input
                type="text"
                maxLength={100}
                placeholder="Ví dụ: Anh Nam"
                value={point.name}
                onChange={(e) => onPointChange(point.uid, { name: e.target.value })}
                disabled={disabled}
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">
                Địa chỉ giao hàng
              </label>
              <div className="flex gap-1.5">
                <Input
                  type="text"
                  maxLength={300}
                  placeholder="Địa chỉ hoặc tên địa điểm (VD: Chợ Bến Thành, Q.1)"
                  value={point.address}
                  onChange={(e) => onPointAddressInput(point.uid, e.target.value)}
                  onBlur={() => onAutoResolvePoint?.(point.uid)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault()
                      onResolvePoint(point.uid)
                    }
                  }}
                  disabled={disabled || csvImporting}
                />
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  disabled={disabled || point.resolving}
                  onClick={() => onResolvePoint(point.uid)}
                >
                  {point.resolving ? (
                    <Loader2Icon data-icon="inline-start" className="animate-spin" />
                  ) : (
                    <SearchIcon data-icon="inline-start" />
                  )}
                  Tìm
                </Button>
              </div>
            </div>

            <GeocodeFeedback
              warning={point.warning}
              note={point.note}
              candidates={point.candidates}
              disabled={disabled}
              isPickingThis={
                pendingPick?.kind === "customer" && pendingPick.uid === point.uid
              }
              onSelectCandidate={(candidate) =>
                onSelectPointCandidate(point.uid, candidate)
              }
              onPickOnMap={() => onPickPointOnMap(point.uid, index)}
            />

            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground select-none">
                Tọa độ nâng cao (tùy chọn)
              </summary>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <div className="flex flex-col gap-1">
                  <label className="text-[11px] text-muted-foreground">Vĩ độ</label>
                  <Input
                    type="number"
                    step="any"
                    placeholder="Tự điền"
                    value={point.lat}
                    onChange={(e) =>
                      onPointCoordInput(point.uid, "lat", e.target.value)
                    }
                    disabled={disabled}
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-[11px] text-muted-foreground">
                    Kinh độ
                  </label>
                  <Input
                    type="number"
                    step="any"
                    placeholder="Tự điền"
                    value={point.lon}
                    onChange={(e) =>
                      onPointCoordInput(point.uid, "lon", e.target.value)
                    }
                    disabled={disabled}
                  />
                </div>
              </div>
            </details>

            <div className="grid grid-cols-2 gap-2">
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">
                  Khối lượng (kg)
                </label>
                <Input
                  type="number"
                  min={0.01}
                  step="any"
                  value={point.demand}
                  onChange={(e) =>
                    onPointChange(point.uid, { demand: e.target.value })
                  }
                  disabled={disabled}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">
                  Phục vụ (phút)
                </label>
                <Input
                  type="number"
                  min={0}
                  step={0.5}
                  placeholder={manualServiceMinutes || "5"}
                  value={point.serviceMinutes}
                  onChange={(e) =>
                    onPointChange(point.uid, { serviceMinutes: e.target.value })
                  }
                  disabled={disabled}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">
                  Từ giờ (tùy chọn)
                </label>
                <Input
                  type="time"
                  value={point.windowStart}
                  onChange={(e) =>
                    onPointChange(point.uid, { windowStart: e.target.value })
                  }
                  disabled={disabled}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">
                  Đến giờ (tùy chọn)
                </label>
                <Input
                  type="time"
                  value={point.windowEnd}
                  onChange={(e) =>
                    onPointChange(point.uid, { windowEnd: e.target.value })
                  }
                  disabled={disabled}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={onAddPoint}
        >
          <PlusIcon data-icon="inline-start" />
          Thêm điểm
        </Button>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={disabled}
          onClick={onPreviewPoints}
        >
          <EyeIcon data-icon="inline-start" />
          Xem điểm
        </Button>
      </div>
    </div>
  )
}
