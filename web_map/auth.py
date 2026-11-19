"""SQLite authentication and two-role authorization for the delivery app."""

from __future__ import annotations

import os
import secrets
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import current_app, g, jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash


ROLES = ("admin", "driver")
TRIP_STATUSES = ("assigned", "in_progress", "completed", "cancelled")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db() -> sqlite3.Connection:
    if "auth_db" not in g:
        database = Path(current_app.config["DATABASE"])
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        g.auth_db = connection
    return g.auth_db


def close_db(_error=None) -> None:
    connection = g.pop("auth_db", None)
    if connection is not None:
        connection.close()


def init_db(*, seed_defaults: bool = True) -> None:
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            full_name TEXT NOT NULL DEFAULT '',
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'driver')),
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            created_at TEXT NOT NULL
        );

        """
    )
    _ensure_user_schema(db)
    _ensure_trip_schema(db)
    if seed_defaults:
        _insert_default_user(
            os.environ.get("RAPO_ADMIN_USERNAME", "admin"),
            os.environ.get("RAPO_ADMIN_PASSWORD", "admin123"),
            "admin",
            os.environ.get("RAPO_ADMIN_NAME", "Quản trị điều phối"),
        )
        _insert_default_user(
            os.environ.get("RAPO_DRIVER_USERNAME", "driver"),
            os.environ.get("RAPO_DRIVER_PASSWORD", "driver123"),
            "driver",
            os.environ.get("RAPO_DRIVER_NAME", "Tài xế mặc định"),
        )
    db.commit()


def _ensure_user_schema(db: sqlite3.Connection) -> None:
    """Add full_name column to existing SQLite databases created before driver profile support."""
    columns = {row[1] for row in db.execute("PRAGMA table_info(users)").fetchall()}
    if columns and "full_name" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN full_name TEXT NOT NULL DEFAULT ''")


def _ensure_trip_schema(db: sqlite3.Connection) -> None:
    """Migrate the first job-level trip schema to one trip per vehicle route."""
    columns = {row[1] for row in db.execute("PRAGMA table_info(trips)").fetchall()}
    if columns and "route_index" not in columns:
        db.execute("DROP INDEX IF EXISTS idx_trips_driver")
        db.execute("ALTER TABLE trips RENAME TO trips_legacy")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS trips (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            route_index INTEGER NOT NULL CHECK (route_index >= 0),
            driver_id INTEGER NOT NULL REFERENCES users(id),
            created_by INTEGER NOT NULL REFERENCES users(id),
            status TEXT NOT NULL CHECK (status IN ('assigned', 'in_progress', 'completed', 'cancelled')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(job_id, algorithm, route_index)
        );

        CREATE TABLE IF NOT EXISTS trip_stops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id TEXT NOT NULL REFERENCES trips(id) ON DELETE CASCADE,
            stop_index INTEGER NOT NULL CHECK (stop_index >= 0),
            customer_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed')),
            completed_at TEXT,
            UNIQUE(trip_id, stop_index)
        );

        CREATE INDEX IF NOT EXISTS idx_trips_driver ON trips(driver_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_trip_stops_trip ON trip_stops(trip_id, stop_index);
        """
    )
    legacy_exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'trips_legacy'"
    ).fetchone()
    if legacy_exists:
        # Old rows represented an entire optimization job and cannot identify a
        # route safely. Keep them as route 0 under a legacy label.
        db.execute(
            "INSERT OR IGNORE INTO trips(id, job_id, algorithm, route_index, driver_id, created_by, status, created_at, updated_at) "
            "SELECT id, job_id, 'legacy-job', 0, driver_id, created_by, status, created_at, updated_at FROM trips_legacy"
        )
        db.execute("DROP TABLE trips_legacy")


def init_auth(app, database: str | Path) -> None:
    app.config.setdefault("DATABASE", str(database))
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
    app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
    app.config.setdefault("PERMANENT_SESSION_LIFETIME", 8 * 60 * 60)
    if not app.secret_key:
        app.secret_key = os.environ.get("RAPO_SECRET_KEY") or secrets.token_hex(32)
    app.teardown_appcontext(close_db)

    @app.before_request
    def load_logged_in_user():
        user_id = session.get("user_id")
        g.user = (
            get_db().execute(
                "SELECT id, username, full_name, role, active FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if user_id is not None
            else None
        )
        if g.user is not None and not g.user["active"]:
            session.clear()
            g.user = None

    with app.app_context():
        init_db()


def _insert_default_user(username: str, password: str, role: str, full_name: str = "") -> None:
    if not get_db().execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
        create_user(username, password, role, full_name=full_name)


def create_user(username: str, password: str, role: str, full_name: str = "") -> dict:
    username = username.strip()
    full_name = full_name.strip()
    if role not in ROLES:
        raise ValueError("Role không hợp lệ")
    if not 3 <= len(username) <= 50 or not username.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Username cần 3-50 ký tự chữ, số, dấu gạch ngang hoặc gạch dưới")
    if len(full_name) > 100:
        raise ValueError("Tên hiển thị không được vượt quá 100 ký tự")
    if len(password) < 8:
        raise ValueError("Mật khẩu phải có ít nhất 8 ký tự")
    now = utc_now()
    db = get_db()
    try:
        cursor = db.execute(
            "INSERT INTO users(username, full_name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, full_name, generate_password_hash(password), role, now),
        )
        db.commit()
    except sqlite3.IntegrityError as error:
        raise ValueError("Username đã tồn tại") from error
    return {
        "id": cursor.lastrowid,
        "username": username,
        "full_name": full_name,
        "role": role,
        "active": True,
        "created_at": now,
    }


def authenticate(username: str, password: str):
    user = get_db().execute(
        "SELECT id, username, full_name, password_hash, role, active FROM users WHERE username = ?",
        (username.strip(),),
    ).fetchone()
    if user is None or not user["active"] or not check_password_hash(user["password_hash"], password):
        return None
    return user


def user_payload(user=None) -> dict:
    user = user if user is not None else g.user
    return {
        "id": int(user["id"]),
        "username": user["username"],
        "full_name": user["full_name"] or "",
        "role": user["role"],
    }


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return jsonify(error="Chưa đăng nhập"), 401
        return view(*args, **kwargs)

    return wrapped


def role_required(role: str):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if g.user["role"] != role:
                return jsonify(error="Không có quyền thực hiện chức năng này"), 403
            return view(*args, **kwargs)

        return wrapped

    return decorator


def driver_exists(driver_id: int) -> bool:
    return get_db().execute(
        "SELECT 1 FROM users WHERE id = ? AND role = 'driver' AND active = 1",
        (driver_id,),
    ).fetchone() is not None


def _driver_row_payload(row) -> dict:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "full_name": row["full_name"] or "",
        "role": row["role"],
        "active": bool(row["active"]),
        "created_at": row["created_at"],
        "trips_total": int(row["trips_total"]),
        "trips_active": int(row["trips_active"]),
        "trips_completed": int(row["trips_completed"]),
    }


def list_drivers(*, include_inactive: bool = False) -> list[dict]:
    active_filter = "" if include_inactive else "AND u.active = 1"
    rows = get_db().execute(
        f"""
        SELECT
            u.id,
            u.username,
            u.full_name,
            u.role,
            u.active,
            u.created_at,
            COUNT(t.id) AS trips_total,
            SUM(CASE WHEN t.status IN ('assigned', 'in_progress') THEN 1 ELSE 0 END) AS trips_active,
            SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS trips_completed
        FROM users u
        LEFT JOIN trips t ON t.driver_id = u.id
        WHERE u.role = 'driver' {active_filter}
        GROUP BY u.id
        ORDER BY u.active DESC, u.created_at DESC, u.username ASC
        """
    ).fetchall()
    return [
        _driver_row_payload(
            {
                "id": row["id"],
                "username": row["username"],
                "full_name": row["full_name"],
                "role": row["role"],
                "active": row["active"],
                "created_at": row["created_at"],
                "trips_total": row["trips_total"] or 0,
                "trips_active": row["trips_active"] or 0,
                "trips_completed": row["trips_completed"] or 0,
            }
        )
        for row in rows
    ]


def driver_by_id(driver_id: int) -> dict | None:
    row = get_db().execute(
        """
        SELECT
            u.id,
            u.username,
            u.full_name,
            u.role,
            u.active,
            u.created_at,
            COUNT(t.id) AS trips_total,
            SUM(CASE WHEN t.status IN ('assigned', 'in_progress') THEN 1 ELSE 0 END) AS trips_active,
            SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS trips_completed
        FROM users u
        LEFT JOIN trips t ON t.driver_id = u.id
        WHERE u.id = ? AND u.role = 'driver'
        GROUP BY u.id
        """,
        (driver_id,),
    ).fetchone()
    if row is None:
        return None
    return _driver_row_payload(
        {
            "id": row["id"],
            "username": row["username"],
            "full_name": row["full_name"],
            "role": row["role"],
            "active": row["active"],
            "created_at": row["created_at"],
            "trips_total": row["trips_total"] or 0,
            "trips_active": row["trips_active"] or 0,
            "trips_completed": row["trips_completed"] or 0,
        }
    )


def update_driver(
    driver_id: int,
    *,
    full_name: str | None = None,
    active: bool | None = None,
    password: str | None = None,
) -> dict:
    existing = driver_by_id(driver_id)
    if existing is None:
        raise LookupError("Không tìm thấy tài khoản tài xế")
    updates: list[str] = []
    params: list[object] = []
    if full_name is not None:
        cleaned_name = full_name.strip()
        if len(cleaned_name) > 100:
            raise ValueError("Tên tài xế không được vượt quá 100 ký tự")
        updates.append("full_name = ?")
        params.append(cleaned_name)
    if active is not None:
        updates.append("active = ?")
        params.append(1 if active else 0)
    if password is not None:
        if len(password) < 8:
            raise ValueError("Mật khẩu mới phải có ít nhất 8 ký tự")
        updates.append("password_hash = ?")
        params.append(generate_password_hash(password))
    if not updates:
        raise ValueError("Không có thông tin nào cần cập nhật")
    params.append(driver_id)
    db = get_db()
    db.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ? AND role = 'driver'", params)
    db.commit()
    return driver_by_id(driver_id)


def delete_driver(driver_id: int) -> None:
    existing = driver_by_id(driver_id)
    if existing is None:
        raise LookupError("Không tìm thấy tài khoản tài xế")
    if existing["trips_total"] > 0:
        raise ValueError("Không thể xóa tài xế đã có lịch sử chuyến giao; hãy tạm khóa tài khoản thay vì xóa")
    db = get_db()
    db.execute("DELETE FROM users WHERE id = ? AND role = 'driver'", (driver_id,))
    db.commit()


def create_trip(
    trip_id: str,
    job_id: str,
    algorithm: str,
    route_index: int,
    driver_id: int,
    created_by: int,
    customer_ids: list[int],
) -> None:
    now = utc_now()
    db = get_db()
    try:
        db.execute(
            "INSERT INTO trips(id, job_id, algorithm, route_index, driver_id, created_by, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'assigned', ?, ?)",
            (trip_id, job_id, algorithm, route_index, driver_id, created_by, now, now),
        )
        db.executemany(
            "INSERT INTO trip_stops(trip_id, stop_index, customer_id) VALUES (?, ?, ?)",
            [(trip_id, index, customer_id) for index, customer_id in enumerate(customer_ids)],
        )
        db.commit()
    except sqlite3.IntegrityError as error:
        db.rollback()
        raise ValueError("Route này đã được giao cho Driver") from error


def trip_by_id(trip_id: str):
    return get_db().execute(
        "SELECT t.*, d.username AS driver_username, d.full_name AS driver_full_name, a.username AS admin_username "
        "FROM trips t JOIN users d ON d.id = t.driver_id JOIN users a ON a.id = t.created_by "
        "WHERE t.id = ?",
        (trip_id,),
    ).fetchone()


def trip_payload(row) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "algorithm": row["algorithm"],
        "route_index": int(row["route_index"]),
        "driver_id": int(row["driver_id"]),
        "driver_username": row["driver_username"],
        "driver_full_name": row["driver_full_name"] or "",
        "created_by": int(row["created_by"]),
        "admin_username": row["admin_username"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_trips_for_user(user) -> list[dict]:
    where = "" if user["role"] == "admin" else "WHERE t.driver_id = ?"
    parameters = () if user["role"] == "admin" else (int(user["id"]),)
    rows = get_db().execute(
        "SELECT t.*, d.username AS driver_username, d.full_name AS driver_full_name, a.username AS admin_username "
        "FROM trips t JOIN users d ON d.id = t.driver_id JOIN users a ON a.id = t.created_by "
        f"{where} ORDER BY t.created_at DESC",
        parameters,
    ).fetchall()
    return [trip_payload(row) for row in rows]


def list_trip_stops(trip_id: str) -> list[dict]:
    rows = get_db().execute(
        "SELECT stop_index, customer_id, status, completed_at FROM trip_stops "
        "WHERE trip_id = ? ORDER BY stop_index",
        (trip_id,),
    ).fetchall()
    return [
        {
            "stop_index": int(row["stop_index"]),
            "customer_id": int(row["customer_id"]),
            "status": row["status"],
            "completed_at": row["completed_at"],
        }
        for row in rows
    ]


def list_job_stops(job_id: str) -> list[dict]:
    rows = get_db().execute(
        "SELECT s.stop_index, s.customer_id, s.status, s.completed_at, t.route_index, t.id AS trip_id "
        "FROM trip_stops s JOIN trips t ON t.id = s.trip_id "
        "WHERE t.job_id = ? ORDER BY t.route_index, s.stop_index",
        (job_id,),
    ).fetchall()
    return [
        {
            "stop_index": int(row["stop_index"]),
            "customer_id": int(row["customer_id"]),
            "status": row["status"],
            "completed_at": row["completed_at"],
            "route_index": int(row["route_index"]),
            "trip_id": row["trip_id"],
        }
        for row in rows
    ]


def complete_trip_stop(row, stop_index: int, user) -> None:
    if user["role"] != "driver" or int(row["driver_id"]) != int(user["id"]):
        raise PermissionError("Điểm giao này không thuộc chuyến của tài xế hiện tại")
    db = get_db()
    target = db.execute(
        "SELECT status FROM trip_stops WHERE trip_id = ? AND stop_index = ?",
        (row["id"], stop_index),
    ).fetchone()
    if target is None:
        raise ValueError("Điểm giao không tồn tại trong chuyến")
    # PATCH lặp lại cùng một điểm là an toàn. Trường hợp thường gặp là SQLite đã
    # ghi thành công nhưng trình duyệt mất phản hồi và gửi lại yêu cầu.
    if target["status"] == "completed":
        return
    if row["status"] == "assigned":
        raise ValueError("Phải bắt đầu chuyến trước khi hoàn thành điểm giao")
    if row["status"] in {"completed", "cancelled"}:
        raise ValueError("Chuyến đã kết thúc")
    pending = db.execute(
        "SELECT stop_index FROM trip_stops WHERE trip_id = ? AND status = 'pending' ORDER BY stop_index LIMIT 1",
        (row["id"],),
    ).fetchone()
    if pending is None:
        raise ValueError("Chuyến không còn điểm giao đang chờ")
    if int(pending["stop_index"]) != stop_index:
        raise ValueError("Phải hoàn thành điểm giao theo đúng thứ tự tuyến")
    now = utc_now()
    db.execute(
        "UPDATE trip_stops SET status = 'completed', completed_at = ? WHERE trip_id = ? AND stop_index = ?",
        (now, row["id"], stop_index),
    )
    remaining = db.execute(
        "SELECT COUNT(*) FROM trip_stops WHERE trip_id = ? AND status = 'pending'",
        (row["id"],),
    ).fetchone()[0]
    status = "completed" if remaining == 0 else "in_progress"
    db.execute(
        "UPDATE trips SET status = ?, updated_at = ? WHERE id = ?",
        (status, now, row["id"]),
    )
    db.commit()


def update_trip_status(row, new_status: str, user) -> None:
    current = row["status"]
    if new_status not in TRIP_STATUSES:
        raise ValueError("Trạng thái chuyến không hợp lệ")
    if user["role"] == "driver":
        if int(row["driver_id"]) != int(user["id"]):
            raise PermissionError("Chuyến này không được gán cho tài xế hiện tại")
        if current != "assigned" or new_status != "in_progress":
            raise ValueError("Driver chỉ được bắt đầu chuyến; hoàn thành chuyến được tự động tính từ các điểm giao")
    elif new_status != "cancelled":
        raise ValueError("Admin chỉ được huỷ chuyến")
    get_db().execute(
        "UPDATE trips SET status = ?, updated_at = ? WHERE id = ?",
        (new_status, utc_now(), row["id"]),
    )
    get_db().commit()


def reassign_trip(trip_id: str, new_driver_id: int, user) -> dict:
    if user["role"] != "admin":
        raise PermissionError("Chỉ quản trị viên mới có quyền điều chuyển tài xế")
    if not driver_exists(new_driver_id):
        raise ValueError("Tài xế mới không tồn tại hoặc đã bị khóa")
    row = trip_by_id(trip_id)
    if row is None:
        raise LookupError("Không tìm thấy chuyến giao hàng")
    if row["status"] in ("completed", "cancelled"):
        raise ValueError("Không thể đổi tài xế cho chuyến đã hoàn tất hoặc đã huỷ")
    db = get_db()
    db.execute(
        "UPDATE trips SET driver_id = ?, updated_at = ? WHERE id = ?",
        (new_driver_id, utc_now(), trip_id),
    )
    db.commit()
    return trip_payload(trip_by_id(trip_id))
