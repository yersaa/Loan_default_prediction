-- name: default_by_grade
-- 1. JOIN + GROUP BY: дефолтность и средняя ставка по грейдам
SELECT g.grade,
       COUNT(*)                        AS loans,
       ROUND(AVG(o.is_default) * 100, 2) AS default_rate_pct,
       ROUND(AVG(l.int_rate), 2)        AS avg_int_rate
FROM loans l
JOIN outcomes o  ON o.loan_id = l.loan_id
JOIN dim_grade g ON g.sub_grade = l.sub_grade
WHERE o.is_default IS NOT NULL
GROUP BY g.grade
ORDER BY g.grade;

-- name: purpose_having
-- 2. GROUP BY + HAVING: цели кредита с объёмом от 5000 займов
SELECT p.purpose,
       COUNT(*)                          AS loans,
       ROUND(AVG(o.is_default) * 100, 2) AS default_rate_pct
FROM loans l
JOIN outcomes o    ON o.loan_id = l.loan_id
JOIN dim_purpose p ON p.purpose_id = l.purpose_id
WHERE o.is_default IS NOT NULL
GROUP BY p.purpose
HAVING COUNT(*) >= 5000
ORDER BY default_rate_pct DESC;

-- name: vintage_by_year_term
-- 3. Винтажный анализ: дефолтность по году выдачи и сроку
SELECT CAST(substr(l.issue_d, 1, 4) AS INTEGER) AS issue_year,
       l.term_months,
       COUNT(*)                                 AS completed_loans,
       ROUND(AVG(o.is_default) * 100, 2)        AS default_rate_pct
FROM loans l
JOIN outcomes o ON o.loan_id = l.loan_id
WHERE o.is_default IS NOT NULL
GROUP BY issue_year, l.term_months
ORDER BY issue_year, l.term_months;

-- name: state_rank
-- 4. CTE + оконные RANK(): рейтинг штатов по дефолтности в целом и внутри региона
WITH state_stats AS (
    SELECT s.state_code, s.region,
           COUNT(*)                          AS loans,
           ROUND(AVG(o.is_default) * 100, 2) AS default_rate_pct
    FROM loans l
    JOIN outcomes o  ON o.loan_id = l.loan_id
    JOIN dim_state s ON s.state_code = l.state_code
    WHERE o.is_default IS NOT NULL
    GROUP BY s.state_code, s.region
    HAVING COUNT(*) >= 1000
)
SELECT *,
       RANK() OVER (ORDER BY default_rate_pct DESC)                     AS rank_overall,
       RANK() OVER (PARTITION BY region ORDER BY default_rate_pct DESC) AS rank_in_region
FROM state_stats
ORDER BY rank_overall;

-- name: monthly_rolling
-- 5. Оконная функция: скользящее среднее дефолтности за 3 месяца
WITH monthly AS (
    SELECT substr(l.issue_d, 1, 7)  AS issue_month,
           COUNT(*)                 AS loans,
           AVG(o.is_default) * 100  AS default_rate_pct
    FROM loans l
    JOIN outcomes o ON o.loan_id = l.loan_id
    WHERE o.is_default IS NOT NULL
    GROUP BY issue_month
)
SELECT issue_month, loans,
       ROUND(default_rate_pct, 2) AS default_rate_pct,
       ROUND(AVG(default_rate_pct) OVER (ORDER BY issue_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
           AS rolling_3m_pct
FROM monthly
ORDER BY issue_month;

-- name: yoy_volume_lag
-- 6. LAG: прирост объёма выдач по целям год к году (все займы, включая открытые)
WITH yearly AS (
    SELECT p.purpose,
           CAST(substr(l.issue_d, 1, 4) AS INTEGER) AS issue_year,
           ROUND(SUM(l.loan_amnt) / 1e6, 1)         AS volume_musd
    FROM loans l
    JOIN dim_purpose p ON p.purpose_id = l.purpose_id
    GROUP BY p.purpose, issue_year
)
SELECT purpose, issue_year, volume_musd,
       LAG(volume_musd) OVER (PARTITION BY purpose ORDER BY issue_year) AS prev_year_musd,
       ROUND(100.0 * (volume_musd - LAG(volume_musd) OVER (PARTITION BY purpose ORDER BY issue_year))
             / LAG(volume_musd) OVER (PARTITION BY purpose ORDER BY issue_year), 1) AS yoy_growth_pct
FROM yearly
WHERE purpose IN ('debt_consolidation', 'credit_card', 'home_improvement', 'small_business')
ORDER BY purpose, issue_year;

-- name: dti_deciles
-- 7. CTE + NTILE: дефолтность по децилям долговой нагрузки (DTI)
WITH ranked AS (
    SELECT b.dti, o.is_default,
           NTILE(10) OVER (ORDER BY b.dti) AS dti_decile
    FROM borrowers b
    JOIN outcomes o ON o.loan_id = b.loan_id
    WHERE o.is_default IS NOT NULL AND b.dti IS NOT NULL
)
SELECT dti_decile,
       ROUND(MIN(dti), 1)                AS dti_from,
       ROUND(MAX(dti), 1)                AS dti_to,
       COUNT(*)                          AS loans,
       ROUND(AVG(is_default) * 100, 2)   AS default_rate_pct
FROM ranked
GROUP BY dti_decile
ORDER BY dti_decile;

-- name: income_vs_grade_avg
-- 8. Подзапрос: доход выше/ниже среднего по своему подгрейду — влияет ли на дефолт
WITH grade_income AS (
    SELECT l.sub_grade, AVG(b.annual_inc) AS avg_inc
    FROM loans l JOIN borrowers b ON b.loan_id = l.loan_id
    GROUP BY l.sub_grade
)
SELECT g.grade,
       CASE WHEN b.annual_inc > (SELECT gi.avg_inc FROM grade_income gi WHERE gi.sub_grade = l.sub_grade)
            THEN 'above_avg' ELSE 'below_avg' END AS income_group,
       COUNT(*)                          AS loans,
       ROUND(AVG(o.is_default) * 100, 2) AS default_rate_pct
FROM loans l
JOIN borrowers b ON b.loan_id = l.loan_id
JOIN outcomes o  ON o.loan_id = l.loan_id
JOIN dim_grade g ON g.sub_grade = l.sub_grade
WHERE o.is_default IS NOT NULL
GROUP BY g.grade, income_group
ORDER BY g.grade, income_group;

-- name: loss_by_grade
-- 9. Потери и возвраты по списанным займам в разрезе грейда
SELECT g.grade,
       COUNT(*)                                                       AS charged_off,
       ROUND(AVG(l.loan_amnt), 0)                                     AS avg_loan,
       ROUND(100.0 * SUM(o.total_pymnt) / SUM(l.loan_amnt), 1)        AS paid_back_pct,
       ROUND(100.0 * SUM(o.recoveries) / SUM(l.loan_amnt), 2)         AS recoveries_pct
FROM loans l
JOIN outcomes o  ON o.loan_id = l.loan_id
JOIN dim_grade g ON g.sub_grade = l.sub_grade
WHERE o.is_default = 1
GROUP BY g.grade
ORDER BY g.grade;

-- name: fico_home_matrix
-- 10. CASE + JOIN 4 таблиц: дефолтность по диапазонам FICO и типу жилья
SELECT CASE WHEN b.fico < 670 THEN '1: <670'
            WHEN b.fico < 710 THEN '2: 670-709'
            WHEN b.fico < 750 THEN '3: 710-749'
            ELSE '4: 750+' END            AS fico_band,
       h.home_ownership,
       COUNT(*)                           AS loans,
       ROUND(AVG(o.is_default) * 100, 2)  AS default_rate_pct
FROM loans l
JOIN borrowers b ON b.loan_id = l.loan_id
JOIN dim_home h  ON h.home_id = b.home_id
JOIN outcomes o  ON o.loan_id = l.loan_id
WHERE o.is_default IS NOT NULL AND h.home_ownership IN ('MORTGAGE', 'OWN', 'RENT')
GROUP BY fico_band, h.home_ownership
ORDER BY fico_band, h.home_ownership;
