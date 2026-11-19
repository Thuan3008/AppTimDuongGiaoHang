import { useState, type FormEvent } from "react"
import { Loader2Icon, LogInIcon, MoonIcon, SunIcon, TruckIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"

interface LoginScreenProps {
  initialError?: string | null
  dark: boolean
  onToggleTheme: () => void
  onSubmit: (username: string, password: string) => Promise<void>
}

export function LoginScreen({
  initialError,
  dark,
  onToggleTheme,
  onSubmit,
}: LoginScreenProps) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<string | null>(initialError ?? null)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = username.trim()
    if (!trimmed || !password) {
      setError("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu")
      return
    }
    setError(null)
    setSubmitting(true)
    try {
      await onSubmit(trimmed, password)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đăng nhập thất bại")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="relative flex min-h-svh items-center justify-center bg-muted/40 p-4">
      <div className="absolute top-4 right-4">
        <Button variant="outline" size="sm" onClick={onToggleTheme}>
          {dark ? (
            <SunIcon data-icon="inline-start" />
          ) : (
            <MoonIcon data-icon="inline-start" />
          )}
          {dark ? "Light" : "Dark"}
        </Button>
      </div>

      <Card className="w-full max-w-md shadow-lg">
        <CardHeader className="gap-2">
          <div className="flex items-center justify-between gap-2">
            <Badge variant="secondary" className="gap-1.5 font-mono text-[11px] uppercase tracking-wider">
              <TruckIcon className="size-3" />
              RAPO DELIVERY
            </Badge>
            <span className="text-xs text-muted-foreground">TDVRP Logistics</span>
          </div>
          <CardTitle className="text-xl font-semibold">
            Đăng nhập hệ thống
          </CardTitle>
          <CardDescription>
            Điều phối lộ trình giao hàng phụ thuộc thời gian và theo dõi tiến độ tài xế.
          </CardDescription>
        </CardHeader>

        <form onSubmit={handleSubmit}>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="login-username"
                className="text-sm font-medium text-foreground"
              >
                Tài khoản
              </label>
              <Input
                id="login-username"
                type="text"
                autoComplete="username"
                placeholder="Nhập tên đăng nhập"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={submitting}
                required
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="login-password"
                className="text-sm font-medium text-foreground"
              >
                Mật khẩu
              </label>
              <Input
                id="login-password"
                type="password"
                autoComplete="current-password"
                placeholder="Nhập mật khẩu"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={submitting}
                required
              />
            </div>

            {(error || initialError) && (
              <div
                role="alert"
                className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
              >
                {error || initialError}
              </div>
            )}
          </CardContent>

          <CardFooter className="flex flex-col items-stretch gap-3 pt-4">
            <Button type="submit" disabled={submitting} className="w-full">
              {submitting ? (
                <Loader2Icon data-icon="inline-start" className="animate-spin" />
              ) : (
                <LogInIcon data-icon="inline-start" />
              )}
              {submitting ? "Đang đăng nhập…" : "Đăng nhập"}
            </Button>
            <p className="text-center text-xs text-muted-foreground">
              Đăng nhập bằng tài khoản Điều phối (Admin) hoặc Tài xế (Driver).
            </p>
          </CardFooter>
        </form>
      </Card>
    </div>
  )
}
