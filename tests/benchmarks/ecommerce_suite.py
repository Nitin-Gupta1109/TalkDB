"""
Mini benchmark suite for the seeded ecommerce SQLite DB.

30 questions across 5 difficulty tiers. Each case has:
- question: user NL
- gold_sql: a reference SQL that returns the correct answer against the seed DB
- keywords: tokens the generated SQL should contain (lenient grading signal)

Conversational cases are 3 scenarios x 2 turns = 6 turns.
Multi-turn grading compares the FINAL turn's result to a single standalone gold SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Case:
    id: str
    category: str
    question: str
    gold_sql: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class ConvCase:
    id: str
    category: str  # always "conversational"
    turns: list[str]          # user questions in order
    gold_sql: str             # expected final-turn SQL (graded on result)
    keywords: list[str] = field(default_factory=list)


SIMPLE_AGGREGATION: list[Case] = [
    Case("agg1", "simple_aggregation", "How many customers are there?",
         "SELECT COUNT(*) AS n FROM customers",
         ["count", "customers"]),
    Case("agg2", "simple_aggregation", "What is our total revenue?",
         "SELECT SUM(total_amount) AS rev FROM orders WHERE status='completed'",
         ["sum", "total_amount", "completed"]),
    Case("agg3", "simple_aggregation", "What is the average order value?",
         "SELECT AVG(total_amount) AS aov FROM orders WHERE status='completed'",
         ["avg", "total_amount", "completed"]),
    Case("agg4", "simple_aggregation", "How many products are in the catalog?",
         "SELECT COUNT(*) AS n FROM products",
         ["count", "products"]),
    Case("agg5", "simple_aggregation", "What is the highest order amount?",
         "SELECT MAX(total_amount) AS max_amt FROM orders",
         ["max", "total_amount"]),
    Case("agg6", "simple_aggregation", "How many orders have been placed in total?",
         "SELECT COUNT(*) AS n FROM orders",
         ["count", "orders"]),
]

FILTER_LOOKUP: list[Case] = [
    Case("filt1", "filter_lookup", "How many platinum-tier customers do we have?",
         "SELECT COUNT(*) AS n FROM customers WHERE tier='platinum'",
         ["count", "tier", "platinum"]),
    Case("filt2", "filter_lookup", "How many orders are in completed status?",
         "SELECT COUNT(*) AS n FROM orders WHERE status='completed'",
         ["count", "status", "completed"]),
    Case("filt3", "filter_lookup", "How many products cost more than 400 dollars?",
         "SELECT COUNT(*) AS n FROM products WHERE price > 400",
         ["count", "price", "400"]),
    Case("filt4", "filter_lookup", "How many refunded orders are there?",
         "SELECT COUNT(*) AS n FROM orders WHERE status='refunded'",
         ["count", "refunded"]),
    Case("filt5", "filter_lookup", "How many products are in the electronics category?",
         "SELECT COUNT(*) AS n FROM products WHERE category='electronics'",
         ["count", "category", "electronics"]),
    Case("filt6", "filter_lookup", "How many gold or silver customers are there?",
         "SELECT COUNT(*) AS n FROM customers WHERE tier IN ('gold','silver')",
         ["count", "tier", "gold", "silver"]),
]

SINGLE_JOIN: list[Case] = [
    Case("j1", "single_join", "What is the total revenue from gold tier customers?",
         "SELECT SUM(o.total_amount) AS rev FROM customers c JOIN orders o ON c.id=o.customer_id "
         "WHERE c.tier='gold' AND o.status='completed'",
         ["join", "gold", "sum", "completed"]),
    Case("j2", "single_join", "How many orders has each customer tier placed?",
         "SELECT c.tier, COUNT(*) AS n FROM customers c JOIN orders o ON c.id=o.customer_id GROUP BY c.tier",
         ["join", "tier", "group by", "count"]),
    Case("j3", "single_join", "Top 5 customers by total completed revenue",
         "SELECT c.id, c.name, SUM(o.total_amount) AS rev FROM customers c JOIN orders o ON c.id=o.customer_id "
         "WHERE o.status='completed' GROUP BY c.id, c.name ORDER BY rev DESC LIMIT 5",
         ["join", "sum", "group by", "order by", "limit"]),
    Case("j4", "single_join", "Total revenue by customer tier",
         "SELECT c.tier, SUM(o.total_amount) AS rev FROM customers c JOIN orders o ON c.id=o.customer_id "
         "WHERE o.status='completed' GROUP BY c.tier",
         ["join", "tier", "sum", "group by"]),
    Case("j5", "single_join", "Which customer has placed the most orders?",
         "SELECT c.id, c.name, COUNT(*) AS n FROM customers c JOIN orders o ON c.id=o.customer_id "
         "GROUP BY c.id, c.name ORDER BY n DESC LIMIT 1",
         ["join", "count", "group by", "order by"]),
    Case("j6", "single_join", "How many customers have never placed an order?",
         "SELECT COUNT(*) AS n FROM customers c LEFT JOIN orders o ON c.id=o.customer_id WHERE o.id IS NULL",
         ["left join", "is null", "count"]),
]

MULTI_JOIN: list[Case] = [
    Case("m1", "multi_join", "What is the revenue by product category, for completed orders only?",
         "SELECT p.category, SUM(oi.quantity*oi.unit_price) AS rev FROM order_items oi "
         "JOIN orders o ON oi.order_id=o.id JOIN products p ON oi.product_id=p.id "
         "WHERE o.status='completed' GROUP BY p.category",
         ["join", "category", "sum", "completed", "group by"]),
    Case("m2", "multi_join", "What are the top 3 best-selling products by total quantity?",
         "SELECT p.id, p.name, SUM(oi.quantity) AS units FROM order_items oi JOIN products p ON oi.product_id=p.id "
         "GROUP BY p.id, p.name ORDER BY units DESC LIMIT 3",
         ["join", "sum", "quantity", "order by", "limit"]),
    Case("m3", "multi_join", "Which product category has generated the most revenue?",
         "SELECT p.category, SUM(oi.quantity*oi.unit_price) AS rev FROM order_items oi "
         "JOIN orders o ON oi.order_id=o.id JOIN products p ON oi.product_id=p.id "
         "WHERE o.status='completed' GROUP BY p.category ORDER BY rev DESC LIMIT 1",
         ["join", "category", "sum", "limit"]),
    Case("m4", "multi_join", "How many units have been sold in each product category, for completed orders?",
         "SELECT p.category, SUM(oi.quantity) AS units FROM order_items oi JOIN orders o ON oi.order_id=o.id "
         "JOIN products p ON oi.product_id=p.id WHERE o.status='completed' GROUP BY p.category",
         ["join", "category", "quantity", "group by"]),
    Case("m5", "multi_join", "What is the total revenue by customer tier and product category, for completed orders?",
         "SELECT c.tier, p.category, SUM(oi.quantity*oi.unit_price) AS rev FROM customers c "
         "JOIN orders o ON c.id=o.customer_id JOIN order_items oi ON o.id=oi.order_id "
         "JOIN products p ON oi.product_id=p.id WHERE o.status='completed' GROUP BY c.tier, p.category",
         ["join", "tier", "category", "group by"]),
    Case("m6", "multi_join", "How many distinct products has each customer purchased?",
         "SELECT c.id, c.name, COUNT(DISTINCT oi.product_id) AS distinct_products "
         "FROM customers c JOIN orders o ON c.id=o.customer_id JOIN order_items oi ON o.id=oi.order_id "
         "GROUP BY c.id, c.name",
         ["join", "distinct", "product", "group by"]),
]

CONVERSATIONAL: list[ConvCase] = [
    ConvCase("c1", "conversational",
             ["Total revenue by customer tier", "just gold and platinum"],
             "SELECT c.tier, SUM(o.total_amount) AS rev FROM customers c JOIN orders o ON c.id=o.customer_id "
             "WHERE o.status='completed' AND c.tier IN ('gold','platinum') GROUP BY c.tier",
             ["tier", "gold", "platinum", "sum"]),
    ConvCase("c2", "conversational",
             ["How many products in each category?", "only those priced over 100"],
             "SELECT category, COUNT(*) AS n FROM products WHERE price > 100 GROUP BY category",
             ["category", "count", "price", "100", "group by"]),
    ConvCase("c3", "conversational",
             ["Top 10 customers by revenue", "sort by descending revenue and show only gold tier"],
             "SELECT c.id, c.name, SUM(o.total_amount) AS rev FROM customers c JOIN orders o ON c.id=o.customer_id "
             "WHERE o.status='completed' AND c.tier='gold' GROUP BY c.id, c.name ORDER BY rev DESC LIMIT 10",
             ["gold", "sum", "order by", "desc", "limit"]),
]


ALL_CASES: list = [*SIMPLE_AGGREGATION, *FILTER_LOOKUP, *SINGLE_JOIN, *MULTI_JOIN, *CONVERSATIONAL]
