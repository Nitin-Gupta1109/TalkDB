"""Seed a realistic SQLite ecommerce DB for TalkDB smoke testing."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, text

random.seed(42)

DB_PATH = Path("data/example.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
if DB_PATH.exists():
    DB_PATH.unlink()

engine = create_engine(f"sqlite:///{DB_PATH}")

DDL = [
    """
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        tier TEXT NOT NULL CHECK (tier IN ('bronze', 'silver', 'gold', 'platinum')),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
    """,
    """
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(id),
        status TEXT NOT NULL CHECK (status IN ('pending','processing','completed','refunded','cancelled')),
        total_amount REAL NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL REFERENCES orders(id),
        product_id INTEGER NOT NULL REFERENCES products(id),
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL
    )
    """,
]

CATEGORIES = ["electronics", "apparel", "home", "books", "sports"]
TIERS = ["bronze", "silver", "gold", "platinum"]
STATUSES = ["pending", "processing", "completed", "refunded", "cancelled"]
STATUS_WEIGHTS = [0.05, 0.05, 0.80, 0.05, 0.05]

FIRST = ["Ava", "Ben", "Cara", "Dan", "Eli", "Fae", "Gus", "Hana", "Ivy", "Jun", "Kai", "Lia", "Max", "Nia", "Owen"]
LAST = ["Smith", "Jones", "Lee", "Patel", "Garcia", "Kim", "Brown", "Singh", "Khan", "Silva"]


def seed() -> None:
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))

        now = datetime(2026, 4, 24)
        customers = []
        for i in range(1, 201):
            first = random.choice(FIRST)
            last = random.choice(LAST)
            customers.append(
                {
                    "id": i,
                    "email": f"{first.lower()}.{last.lower()}{i}@example.com",
                    "name": f"{first} {last}",
                    "tier": random.choices(TIERS, weights=[0.5, 0.3, 0.15, 0.05])[0],
                    "created_at": (now - timedelta(days=random.randint(0, 720))).isoformat(),
                }
            )
        conn.execute(
            text(
                "INSERT INTO customers (id,email,name,tier,created_at) VALUES (:id,:email,:name,:tier,:created_at)"
            ),
            customers,
        )

        products = []
        for i in range(1, 51):
            cat = random.choice(CATEGORIES)
            products.append(
                {
                    "id": i,
                    "name": f"{cat.capitalize()} item {i}",
                    "category": cat,
                    "price": round(random.uniform(5, 500), 2),
                }
            )
        conn.execute(
            text("INSERT INTO products (id,name,category,price) VALUES (:id,:name,:category,:price)"),
            products,
        )

        orders = []
        items = []
        order_id = 1
        item_id = 1
        for customer in customers:
            for _ in range(random.randint(0, 8)):
                status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
                created = now - timedelta(days=random.randint(0, 365), minutes=random.randint(0, 1440))
                chosen = random.sample(products, random.randint(1, 4))
                total = 0.0
                order_items: list[dict] = []
                for p in chosen:
                    qty = random.randint(1, 3)
                    total += p["price"] * qty
                    order_items.append(
                        {
                            "id": item_id,
                            "order_id": order_id,
                            "product_id": p["id"],
                            "quantity": qty,
                            "unit_price": p["price"],
                        }
                    )
                    item_id += 1
                orders.append(
                    {
                        "id": order_id,
                        "customer_id": customer["id"],
                        "status": status,
                        "total_amount": round(total, 2),
                        "created_at": created.isoformat(),
                    }
                )
                items.extend(order_items)
                order_id += 1

        conn.execute(
            text(
                "INSERT INTO orders (id,customer_id,status,total_amount,created_at) "
                "VALUES (:id,:customer_id,:status,:total_amount,:created_at)"
            ),
            orders,
        )
        conn.execute(
            text(
                "INSERT INTO order_items (id,order_id,product_id,quantity,unit_price) "
                "VALUES (:id,:order_id,:product_id,:quantity,:unit_price)"
            ),
            items,
        )

    print(f"Seeded {DB_PATH}")
    print(f"  customers: {len(customers)}")
    print(f"  products:  {len(products)}")
    print(f"  orders:    {len(orders)}")
    print(f"  order_items: {len(items)}")


if __name__ == "__main__":
    seed()
