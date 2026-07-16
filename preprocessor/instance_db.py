"""算例 SQLite 持久化：每个算例一个文件 instances/{name}.db。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from common.data_models import InstanceData
from configuration import INSTANCES_DIR

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS instances (
    name TEXT PRIMARY KEY,
    set_type TEXT NOT NULL,
    num_periods INTEGER NOT NULL,
    period_hours REAL NOT NULL DEFAULT 1.0,
    seed INTEGER
);

CREATE TABLE IF NOT EXISTS configs (
    instance_name TEXT NOT NULL,
    config_id TEXT NOT NULL,
    lead_time INTEGER NOT NULL,
    PRIMARY KEY (instance_name, config_id),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS config_wait_steps (
    instance_name TEXT NOT NULL,
    config_id TEXT NOT NULL,
    threshold REAL NOT NULL,
    delta REAL NOT NULL,
    PRIMARY KEY (instance_name, config_id, threshold),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS item_wait_steps (
    instance_name TEXT NOT NULL,
    item_id TEXT NOT NULL,
    threshold REAL NOT NULL,
    delta REAL NOT NULL,
    PRIMARY KEY (instance_name, item_id, threshold),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS machines (
    instance_name TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    capacity REAL NOT NULL,
    PRIMARY KEY (instance_name, machine_id),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS items (
    instance_name TEXT NOT NULL,
    item_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    tray_length REAL NOT NULL,
    item_lead_time INTEGER NOT NULL,
    holding_cost REAL NOT NULL,
    backorder_cost REAL NOT NULL,
    scrap_cost REAL NOT NULL,
    earliest_start INTEGER NOT NULL,
    deadline INTEGER NOT NULL,
    wip_qty INTEGER NOT NULL,
    PRIMARY KEY (instance_name, item_id),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS demand (
    instance_name TEXT NOT NULL,
    item_id TEXT NOT NULL,
    period INTEGER NOT NULL,
    qty INTEGER NOT NULL,
    PRIMARY KEY (instance_name, item_id, period),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS production_costs (
    instance_name TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    cost REAL NOT NULL,
    PRIMARY KEY (instance_name, machine_id, config_id),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS item_machines (
    instance_name TEXT NOT NULL,
    item_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    PRIMARY KEY (instance_name, item_id, machine_id),
    FOREIGN KEY (instance_name) REFERENCES instances(name) ON DELETE CASCADE
);
"""


def instance_db_path(name: str, directory: Path | None = None) -> Path:
    """单个算例数据库路径：instances/{name}.db"""
    return (directory or INSTANCES_DIR) / f"{name}.db"


def _ensure_items_machine_column(conn: sqlite3.Connection) -> None:
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(items)").fetchall()}
    if "machine_id" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN machine_id TEXT")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_items_machine_column(conn)
    return conn


def init_db(db_path: Path | None = None) -> Path:
    """初始化（创建空库文件）。未指定路径时仅确保 instances 目录存在。"""
    if db_path is None:
        INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        return INSTANCES_DIR
    with _connect(db_path) as conn:
        conn.commit()
    return db_path


def list_instances(directory: Path | None = None) -> list[str]:
    root = directory or INSTANCES_DIR
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.db"))


def delete_instance(name: str, directory: Path | None = None) -> None:
    path = instance_db_path(name, directory)
    if path.exists():
        path.unlink()


def save_instance(
    instance: InstanceData,
    db_path: Path | None = None,
) -> Path:
    """写入单个算例文件（覆盖同名 .db）。"""
    path = db_path or instance_db_path(instance.name)
    if path.exists():
        path.unlink()
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO instances (name, set_type, num_periods, period_hours, seed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                instance.name,
                instance.set_type,
                instance.num_periods,
                instance.period_hours,
                instance.seed,
            ),
        )
        for u in instance.config_ids:
            conn.execute(
                "INSERT INTO configs (instance_name, config_id, lead_time) VALUES (?, ?, ?)",
                (instance.name, u, instance.config_lead_times[u]),
            )
            for step in instance.config_wait_steps.get(u, []):
                conn.execute(
                    """
                    INSERT INTO config_wait_steps (instance_name, config_id, threshold, delta)
                    VALUES (?, ?, ?, ?)
                    """,
                    (instance.name, u, float(step[0]), float(step[1])),
                )
        for m in instance.machine_ids:
            conn.execute(
                "INSERT INTO machines (instance_name, machine_id, capacity) VALUES (?, ?, ?)",
                (instance.name, m, instance.machine_capacities[m]),
            )
        for i in instance.cured_item_ids:
            conn.execute(
                """
                INSERT INTO items (
                    instance_name, item_id, config_id, tray_length, item_lead_time,
                    holding_cost, backorder_cost, scrap_cost,
                    earliest_start, deadline, wip_qty, machine_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    instance.name,
                    i,
                    instance.item_config[i],
                    instance.tray_lengths[i],
                    instance.item_lead_times[i],
                    instance.holding_costs[i],
                    instance.backorder_costs[i],
                    instance.scrap_costs.get(i, instance.backorder_costs[i] * 2),
                    instance.earliest_starts.get(i, 1),
                    instance.deadlines.get(i, instance.num_periods),
                    instance.wip_quantities.get(i, 0),
                    None,
                ),
            )
            allowed = instance.item_machines.get(i)
            if allowed:
                for m_name in dict.fromkeys(allowed):
                    conn.execute(
                        """
                        INSERT INTO item_machines (instance_name, item_id, machine_id)
                        VALUES (?, ?, ?)
                        """,
                        (instance.name, i, m_name),
                    )
            for t, qty in instance.demand.get(i, {}).items():
                conn.execute(
                    "INSERT INTO demand (instance_name, item_id, period, qty) VALUES (?, ?, ?, ?)",
                    (instance.name, i, int(t), int(qty)),
                )
            for step in instance.item_wait_steps.get(i, []):
                conn.execute(
                    """
                    INSERT INTO item_wait_steps (instance_name, item_id, threshold, delta)
                    VALUES (?, ?, ?, ?)
                    """,
                    (instance.name, i, float(step[0]), float(step[1])),
                )
        for m in instance.machine_ids:
            for u in instance.config_ids:
                conn.execute(
                    """
                    INSERT INTO production_costs (instance_name, machine_id, config_id, cost)
                    VALUES (?, ?, ?, ?)
                    """,
                    (instance.name, m, u, instance.production_costs[m][u]),
                )
        conn.commit()
    return path


def load_instance_from_db(name: str, db_path: Path | None = None) -> InstanceData:
    path = db_path or instance_db_path(name)
    if not path.exists():
        raise FileNotFoundError(f"算例 '{name}' 不存在: {path}")

    with _connect(path) as conn:
        meta = conn.execute("SELECT * FROM instances LIMIT 1").fetchone()
        if meta is None:
            raise FileNotFoundError(f"算例文件为空: {path}")
        inst_name = str(meta["name"])

        config_rows = conn.execute(
            "SELECT config_id, lead_time FROM configs WHERE instance_name = ? ORDER BY config_id",
            (inst_name,),
        ).fetchall()
        config_ids = [str(r["config_id"]) for r in config_rows]
        config_lead_times = {str(r["config_id"]): int(r["lead_time"]) for r in config_rows}

        config_wait_steps: dict[str, list[list[int]]] = {u: [] for u in config_ids}
        for r in conn.execute(
            """
            SELECT config_id, threshold, delta FROM config_wait_steps
            WHERE instance_name = ? ORDER BY config_id, threshold
            """,
            (inst_name,),
        ):
            config_wait_steps[str(r["config_id"])].append(
                [int(r["threshold"]), int(r["delta"])]
            )

        machine_rows = conn.execute(
            "SELECT machine_id, capacity FROM machines WHERE instance_name = ? ORDER BY machine_id",
            (inst_name,),
        ).fetchall()
        machine_ids = [str(r["machine_id"]) for r in machine_rows]
        machine_capacities = {str(r["machine_id"]): float(r["capacity"]) for r in machine_rows}

        item_rows = conn.execute(
            "SELECT * FROM items WHERE instance_name = ? ORDER BY item_id",
            (inst_name,),
        ).fetchall()
        cured_item_ids = [str(r["item_id"]) for r in item_rows]
        item_config = {str(r["item_id"]): str(r["config_id"]) for r in item_rows}
        item_machines: dict[str, list[str]] = {i: [] for i in cured_item_ids}
        for r in conn.execute(
            """
            SELECT item_id, machine_id FROM item_machines
            WHERE instance_name = ? ORDER BY item_id, machine_id
            """,
            (inst_name,),
        ):
            item_machines[str(r["item_id"])].append(str(r["machine_id"]))
        if not any(item_machines.values()):
            item_machines = {}
            for r in item_rows:
                mid = r["machine_id"] if "machine_id" in r.keys() else None
                if mid is not None and str(mid).strip():
                    item_machines[str(r["item_id"])] = [str(mid)]
        else:
            item_machines = {k: v for k, v in item_machines.items() if v}
        item_lead_times = {str(r["item_id"]): int(r["item_lead_time"]) for r in item_rows}
        tray_lengths = {str(r["item_id"]): float(r["tray_length"]) for r in item_rows}
        holding_costs = {str(r["item_id"]): float(r["holding_cost"]) for r in item_rows}
        backorder_costs = {str(r["item_id"]): float(r["backorder_cost"]) for r in item_rows}
        scrap_costs = {str(r["item_id"]): float(r["scrap_cost"]) for r in item_rows}
        earliest_starts = {str(r["item_id"]): int(r["earliest_start"]) for r in item_rows}
        deadlines = {str(r["item_id"]): int(r["deadline"]) for r in item_rows}
        wip_quantities = {str(r["item_id"]): int(r["wip_qty"]) for r in item_rows}

        demand: dict[str, dict[int, int]] = {i: {} for i in cured_item_ids}
        for r in conn.execute(
            "SELECT item_id, period, qty FROM demand WHERE instance_name = ?",
            (inst_name,),
        ):
            demand[str(r["item_id"])][int(r["period"])] = int(r["qty"])

        item_wait_steps: dict[str, list[list[int]]] = {i: [] for i in cured_item_ids}
        for r in conn.execute(
            """
            SELECT item_id, threshold, delta FROM item_wait_steps
            WHERE instance_name = ? ORDER BY item_id, threshold
            """,
            (inst_name,),
        ):
            item_wait_steps[str(r["item_id"])].append(
                [int(r["threshold"]), int(r["delta"])]
            )

        production_costs: dict[str, dict[str, float]] = {m: {} for m in machine_ids}
        for r in conn.execute(
            """
            SELECT machine_id, config_id, cost FROM production_costs
            WHERE instance_name = ?
            """,
            (inst_name,),
        ):
            production_costs[str(r["machine_id"])][str(r["config_id"])] = float(r["cost"])

    return InstanceData(
        name=inst_name,
        set_type=str(meta["set_type"]),
        num_periods=int(meta["num_periods"]),
        config_ids=config_ids,
        machine_ids=machine_ids,
        cured_item_ids=cured_item_ids,
        config_lead_times=config_lead_times,
        item_config=item_config,
        item_machines=item_machines,
        item_lead_times=item_lead_times,
        tray_lengths=tray_lengths,
        machine_capacities=machine_capacities,
        production_costs=production_costs,
        holding_costs=holding_costs,
        backorder_costs=backorder_costs,
        demand=demand,
        deadlines=deadlines,
        earliest_starts=earliest_starts,
        wip_quantities=wip_quantities,
        scrap_costs=scrap_costs,
        config_wait_steps=config_wait_steps,
        item_wait_steps=item_wait_steps,
        period_hours=float(meta["period_hours"]),
        seed=meta["seed"],
    )


def instance_name_from_arg(path_or_name: str | Path) -> str:
    p = Path(path_or_name)
    if p.suffix in (".json", ".db"):
        return p.stem
    return str(path_or_name)
