
-- Instamart Customer Intelligence & Support Copilot
-- Part A: SQL analysis

-- Q1. Customers whose completed orders above INR 500 total more than INR 8,000.
SELECT
    customer_id,
    COUNT(*) AS qualifying_orders,
    ROUND(SUM(order_amount), 2) AS qualifying_spend_inr
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend_inr DESC;


-- Q2. Completed revenue by city and product category.
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr,
    COUNT(*) AS completed_orders
FROM orders AS o
JOIN customers AS c ON o.customer_id = c.customer_id
JOIN products AS p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, revenue_inr DESC;


-- Q3. Products priced above the average catalogue price.
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- Q4. Customers who have never placed a completed order.
SELECT
    c.customer_id,
    c.city,
    c.age,
    c.gender
FROM customers AS c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders AS o
    WHERE o.customer_id = c.customer_id
      AND o.status = 'completed'
)
ORDER BY c.customer_id;


-- Q5. Completed spend per customer compared with the average
-- completed spend among customers who have completed orders.
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend_inr) AS avg_spend_inr
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend_inr, 2) AS total_spend_inr,
    ROUND(a.avg_spend_inr, 2) AS average_spend_inr,
    ROUND(cs.total_spend_inr - a.avg_spend_inr, 2) AS difference_inr
FROM customer_spend AS cs
CROSS JOIN average_spend AS a
WHERE cs.total_spend_inr > a.avg_spend_inr
ORDER BY cs.total_spend_inr DESC;


-- Q6. Rank customers by completed spend within each city.
-- RANK and DENSE_RANK handle ties differently; ROW_NUMBER is unique.
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend_inr
    FROM customers AS c
    JOIN orders AS o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    ROUND(total_spend_inr, 2) AS total_spend_inr,
    RANK() OVER (
        PARTITION BY city ORDER BY total_spend_inr DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city ORDER BY total_spend_inr DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city ORDER BY total_spend_inr DESC, customer_id
    ) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;


-- Q7. Daily completed revenue and cumulative revenue.
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS order_date,
        SUM(order_amount) AS daily_revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    order_date,
    ROUND(daily_revenue_inr, 2) AS daily_revenue_inr,
    ROUND(
        SUM(daily_revenue_inr) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY order_date;

