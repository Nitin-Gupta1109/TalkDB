"""
Hard benchmark tier — 20 cases designed to stress:
- window functions, running totals, partitioned rankings
- correlated subqueries, NOT EXISTS, HAVING filters
- temporal reasoning (sqlite date arithmetic, month-over-month, date windows)
- conversational chains with stacked filters and transformations
- semantic ambiguity (the class of error dual-path should catch)

Gold SQL is written for SQLite. Values verified against the seeded ecommerce DB.
"""

from __future__ import annotations

from tests.benchmarks.ecommerce_suite import Case, ConvCase


HARD_AGGREGATION: list[Case] = [
    Case(
        "h_agg1", "hard_aggregation",
        "Running total of completed revenue by month",
        """
        SELECT month, SUM(monthly_rev) OVER (ORDER BY month) AS running_total
        FROM (
            SELECT strftime('%Y-%m', created_at) AS month, SUM(total_amount) AS monthly_rev
            FROM orders WHERE status='completed' GROUP BY month
        )
        ORDER BY month
        """,
        ["sum", "over", "order by", "month"],
    ),
    Case(
        "h_agg2", "hard_aggregation",
        "What share of total completed revenue comes from gold-tier customers?",
        """
        SELECT
            (SELECT SUM(o.total_amount) FROM customers c JOIN orders o ON c.id=o.customer_id
             WHERE c.tier='gold' AND o.status='completed') * 1.0 /
            (SELECT SUM(total_amount) FROM orders WHERE status='completed') AS share
        """,
        ["gold", "completed", "/"],
    ),
    Case(
        "h_agg3", "hard_aggregation",
        "What is the average revenue per completed order?",
        "SELECT AVG(total_amount) AS avg_rev FROM orders WHERE status='completed'",
        ["avg", "total_amount", "completed"],
    ),
    Case(
        "h_agg4", "hard_aggregation",
        "How much revenue comes from customers whose total spend is above the overall average customer spend?",
        """
        WITH customer_spend AS (
            SELECT customer_id, SUM(total_amount) AS spend
            FROM orders WHERE status='completed' GROUP BY customer_id
        )
        SELECT SUM(spend) AS rev_from_above_avg
        FROM customer_spend
        WHERE spend > (SELECT AVG(spend) FROM customer_spend)
        """,
        ["sum", "avg", "group by"],
    ),
]


HARD_TEMPORAL: list[Case] = [
    Case(
        "h_temp1", "hard_temporal",
        "Monthly completed revenue with the change vs the previous month",
        """
        SELECT month, rev, rev - LAG(rev) OVER (ORDER BY month) AS change_vs_prev
        FROM (
            SELECT strftime('%Y-%m', created_at) AS month, SUM(total_amount) AS rev
            FROM orders WHERE status='completed' GROUP BY month
        )
        ORDER BY month
        """,
        ["lag", "over", "order by", "month"],
    ),
    Case(
        "h_temp2", "hard_temporal",
        "How many orders per day of week?",
        """
        SELECT strftime('%w', created_at) AS dow, COUNT(*) AS n
        FROM orders GROUP BY dow ORDER BY dow
        """,
        ["strftime", "%w", "group by"],
    ),
    Case(
        "h_temp3", "hard_temporal",
        "What was the total completed revenue in Q4 2025?",
        """
        SELECT SUM(total_amount) AS rev FROM orders
        WHERE status='completed' AND created_at >= '2025-10-01' AND created_at < '2026-01-01'
        """,
        ["sum", "2025", "completed"],
    ),
    Case(
        "h_temp4", "hard_temporal",
        "What was the first order date for each customer?",
        """
        SELECT customer_id, MIN(created_at) AS first_order_date
        FROM orders GROUP BY customer_id
        """,
        ["min", "created_at", "group by"],
    ),
]


HARD_RANKING: list[Case] = [
    Case(
        "h_rank1", "hard_ranking",
        "Top 2 customers in each tier by total completed revenue",
        """
        SELECT tier, customer_id, total_spend FROM (
            SELECT c.tier, c.id AS customer_id, SUM(o.total_amount) AS total_spend,
                   ROW_NUMBER() OVER (PARTITION BY c.tier ORDER BY SUM(o.total_amount) DESC) AS rn
            FROM customers c JOIN orders o ON c.id=o.customer_id
            WHERE o.status='completed'
            GROUP BY c.tier, c.id
        )
        WHERE rn <= 2
        ORDER BY tier, total_spend DESC
        """,
        ["row_number", "partition by", "tier"],
    ),
    Case(
        "h_rank2", "hard_ranking",
        "Which customers have total spend above the average total spend across all customers?",
        """
        SELECT customer_id, SUM(total_amount) AS total_spend
        FROM orders WHERE status='completed' GROUP BY customer_id
        HAVING SUM(total_amount) > (
            SELECT AVG(s) FROM (
                SELECT SUM(total_amount) AS s FROM orders WHERE status='completed' GROUP BY customer_id
            )
        )
        """,
        ["having", "avg", "group by"],
    ),
    Case(
        "h_rank3", "hard_ranking",
        "What is the least-sold product in the electronics category, by total units sold?",
        """
        SELECT p.id, p.name, SUM(oi.quantity) AS units
        FROM order_items oi JOIN products p ON oi.product_id=p.id
        WHERE p.category='electronics'
        GROUP BY p.id, p.name ORDER BY units ASC LIMIT 1
        """,
        ["electronics", "order by", "asc", "limit 1"],
    ),
    Case(
        "h_rank4", "hard_ranking",
        "How many customers placed at least 5 completed orders?",
        """
        SELECT COUNT(*) AS n FROM (
            SELECT customer_id FROM orders WHERE status='completed'
            GROUP BY customer_id HAVING COUNT(*) >= 5
        )
        """,
        ["having", "count", "5"],
    ),
]


HARD_SUBQUERY: list[Case] = [
    Case(
        "h_sub1", "hard_subquery",
        "List all products that have never been ordered",
        """
        SELECT p.id, p.name FROM products p
        WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id=p.id)
        """,
        ["not exists", "products"],
    ),
    Case(
        "h_sub2", "hard_subquery",
        "Customers whose average completed order value exceeds 1500",
        """
        SELECT customer_id, AVG(total_amount) AS aov
        FROM orders WHERE status='completed' GROUP BY customer_id HAVING AVG(total_amount) > 1500
        """,
        ["having", "avg", "1500"],
    ),
    Case(
        "h_sub3", "hard_subquery",
        "How many product categories have at least one product priced above 400?",
        """
        SELECT COUNT(DISTINCT category) AS n FROM products WHERE price > 400
        """,
        ["count", "distinct", "category", "400"],
    ),
    Case(
        "h_sub4", "hard_subquery",
        "Which customer placed the single most expensive completed order?",
        """
        SELECT c.id, c.name, o.total_amount FROM customers c
        JOIN orders o ON c.id=o.customer_id
        WHERE o.status='completed'
        ORDER BY o.total_amount DESC LIMIT 1
        """,
        ["order by", "total_amount", "desc", "limit 1"],
    ),
]


HARD_CONVERSATIONAL: list[ConvCase] = [
    ConvCase(
        "h_conv1", "hard_conversational",
        [
            "Revenue by month for completed orders",
            "just March and April 2026",
        ],
        """
        SELECT strftime('%Y-%m', created_at) AS month, SUM(total_amount) AS rev
        FROM orders WHERE status='completed'
          AND strftime('%Y-%m', created_at) IN ('2026-03', '2026-04')
        GROUP BY month ORDER BY month
        """,
        ["2026", "month", "sum"],
    ),
    ConvCase(
        "h_conv2", "hard_conversational",
        [
            "Top 10 customers by total completed revenue",
            "only gold tier",
        ],
        """
        SELECT c.id, c.name, SUM(o.total_amount) AS rev
        FROM customers c JOIN orders o ON c.id=o.customer_id
        WHERE o.status='completed' AND c.tier='gold'
        GROUP BY c.id, c.name ORDER BY rev DESC LIMIT 10
        """,
        ["gold", "order by", "desc", "limit 10"],
    ),
    ConvCase(
        "h_conv3", "hard_conversational",
        [
            "List products in the electronics category",
            "sort by price descending",
            "just the top 5",
        ],
        """
        SELECT id, name, price FROM products
        WHERE category='electronics' ORDER BY price DESC LIMIT 5
        """,
        ["electronics", "order by", "desc", "limit 5"],
    ),
    ConvCase(
        "h_conv4", "hard_conversational",
        [
            "Total completed revenue by customer tier",
            "as a percentage of the grand total",
        ],
        """
        SELECT c.tier,
               SUM(o.total_amount) * 100.0 /
                   (SELECT SUM(total_amount) FROM orders WHERE status='completed') AS pct
        FROM customers c JOIN orders o ON c.id=o.customer_id
        WHERE o.status='completed' GROUP BY c.tier
        """,
        ["tier", "/", "sum"],
    ),
]


ALL_HARD: list = [
    *HARD_AGGREGATION,
    *HARD_TEMPORAL,
    *HARD_RANKING,
    *HARD_SUBQUERY,
    *HARD_CONVERSATIONAL,
]
