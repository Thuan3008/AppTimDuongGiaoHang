import {
  ClockIcon,
  LogOutIcon,
  MapIcon,
  MenuIcon,
  MoonIcon,
  RouteIcon,
  SunIcon,
  TruckIcon,
  UsersIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { AppView, User } from "@/types/rapo"

interface HeaderBarProps {
  user: User
  dark: boolean
  clockText: string
  busy: boolean
  activeView: AppView
  routesCount: number
  driverCount: number
  liveTripCount?: number
  onChangeView: (view: AppView) => void
  onToggleTheme: () => void
  onLogout: () => void
  onOpenMobileMenu: () => void
}

export function HeaderBar({
  user,
  dark,
  clockText,
  busy,
  activeView,
  routesCount,
  driverCount,
  liveTripCount = 0,
  onChangeView,
  onToggleTheme,
  onLogout,
  onOpenMobileMenu,
}: HeaderBarProps) {
  const isAdmin = user.role === "admin"
  const displayUser = user.full_name?.trim() || user.username

  return (
    <header className="sticky top-0 z-30 flex min-h-14 shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border bg-card/95 px-3 py-1.5 backdrop-blur-sm sm:px-4">
      <div className="flex min-w-0 items-center gap-2.5">
        {activeView === "map" && (
          <Button
            variant="outline"
            size="icon-sm"
            className="lg:hidden"
            onClick={onOpenMobileMenu}
            aria-label="Mở bảng điều khiển"
          >
            <MenuIcon />
          </Button>
        )}

        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <TruckIcon className="size-4" />
        </div>

        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-sm font-semibold tracking-tight sm:text-base">
              Bản đồ tuyến giao hàng
            </h1>
            <Badge
              variant="outline"
              className="hidden font-mono text-[10px] uppercase 2xl:inline-flex"
            >
              RAPO / TDVRP
            </Badge>
          </div>
          <p className="hidden truncate text-xs text-muted-foreground 2xl:block">
            Hệ thống điều phối tối ưu đa phương thức
          </p>
        </div>
      </div>

      {/* Admin View Navigation */}
      {isAdmin && (
        <nav
          aria-label="Điều hướng chính"
          className="flex flex-wrap items-center gap-1 rounded-lg border border-border bg-muted/50 p-0.5"
        >
          <Button
            type="button"
            variant={activeView === "map" ? "default" : "ghost"}
            size="sm"
            onClick={() => onChangeView("map")}
          >
            <MapIcon data-icon="inline-start" />
            <span>Bản đồ điều phối</span>
          </Button>
          <Button
            type="button"
            variant={activeView === "trips" ? "default" : "ghost"}
            size="sm"
            onClick={() => onChangeView("trips")}
          >
            <RouteIcon data-icon="inline-start" />
            <span>Quản lý các tuyến</span>
            <Badge
              variant={activeView === "trips" ? "secondary" : "outline"}
              className="ml-0.5 px-1.5 text-[10px]"
            >
              {routesCount}
            </Badge>
          </Button>
          <Button
            type="button"
            variant={activeView === "drivers" ? "default" : "ghost"}
            size="sm"
            onClick={() => onChangeView("drivers")}
          >
            <UsersIcon data-icon="inline-start" />
            <span>Quản lý tài xế</span>
            <Badge
              variant={activeView === "drivers" ? "secondary" : "outline"}
              className="ml-0.5 px-1.5 text-[10px]"
            >
              {driverCount}
            </Badge>
          </Button>
        </nav>
      )}

      <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
        {/* Live sync badge with pulsing radar indicator */}
        {liveTripCount > 0 && (
          <div
            className="flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1 text-xs font-semibold text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/15 dark:text-emerald-300 shadow-2xs"
            title={`${liveTripCount} chuyến xe đang ở trạng thái Đang giao (in_progress), đồng bộ trực tiếp tiến độ tài xế`}
          >
            <span className="relative flex size-2 items-center justify-center">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex size-1.5 rounded-full bg-emerald-500" />
            </span>
            <span className="font-medium">Đang đồng bộ Live</span>
            <Badge
              variant="default"
              className="ml-0.5 bg-emerald-600 px-1 py-0 text-[10px] text-white hover:bg-emerald-600 font-mono"
            >
              {liveTripCount}
            </Badge>
          </div>
        )}

        <Badge
          variant={busy ? "default" : "secondary"}
          className="hidden gap-1 font-mono text-xs xl:inline-flex"
        >
          <ClockIcon className="size-3" />
          {clockText}
        </Badge>

        <Badge
          variant="outline"
          className="hidden font-mono text-xs md:inline-flex"
        >
          {displayUser} · {user.role.toUpperCase()}
        </Badge>

        <Button
          variant="outline"
          size="sm"
          onClick={onToggleTheme}
          title={
            dark ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"
          }
        >
          {dark ? (
            <SunIcon data-icon="inline-start" />
          ) : (
            <MoonIcon data-icon="inline-start" />
          )}
          <span className="hidden sm:inline">{dark ? "Light" : "Dark"}</span>
        </Button>

        <Button variant="ghost" size="sm" onClick={onLogout}>
          <LogOutIcon data-icon="inline-start" />
          <span className="hidden sm:inline">Đăng xuất</span>
        </Button>
      </div>
    </header>
  )
}
