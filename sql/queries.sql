-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2 three-table join
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3 scalar subquery
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4 LEFT JOIN
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5 common table expression
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_total AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_totals
)
SELECT ct.customer_id,
       ROUND(ct.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2) AS avg_spend
FROM customer_totals ct
CROSS JOIN avg_total a
WHERE ct.total_spend > a.avg_spend;

-- Q6 ranking window functions
WITH totals AS (
    SELECT c.city, o.customer_id, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, ROUND(total_spend, 2) AS total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM totals;

-- Q7 running total
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day,
       ROUND(daily_revenue, 2) AS daily_revenue,
       ROUND(SUM(daily_revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;