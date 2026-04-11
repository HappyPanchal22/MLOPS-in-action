-- Query 5: Blockbuster Hit Rate by Publisher
-- Classifies every game as Mega Hit / Hit / Moderate / Filler based on sales tiers
-- Then calculates each publisher's "hit rate"
-- Demonstrates: CTE, CASE for tiering, GROUP BY with conditional aggregation, HAVING

WITH GameTiers AS (
    SELECT
        Publisher,
        Name,
        Global_Sales,
        CASE
            WHEN Global_Sales >= 10 THEN 'Mega Hit (10M+)'
            WHEN Global_Sales >= 5 THEN 'Hit (5-10M)'
            WHEN Global_Sales >= 1 THEN 'Moderate (1-5M)'
            ELSE 'Filler (<1M)'
        END AS SalesTier
    FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE Publisher IS NOT NULL AND Publisher != 'N/A'
)
SELECT
    Publisher,
    COUNT(*) AS TotalGames,
    SUM(CASE WHEN SalesTier = 'Mega Hit (10M+)' THEN 1 ELSE 0 END) AS MegaHits,
    SUM(CASE WHEN SalesTier = 'Hit (5-10M)' THEN 1 ELSE 0 END) AS Hits,
    SUM(CASE WHEN SalesTier = 'Moderate (1-5M)' THEN 1 ELSE 0 END) AS Moderate,
    SUM(CASE WHEN SalesTier = 'Filler (<1M)' THEN 1 ELSE 0 END) AS Filler,
    ROUND(
        SUM(CASE WHEN Global_Sales >= 5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    ) AS HitRatePct,
    ROUND(SUM(Global_Sales), 2) AS TotalRevenue_M,
    ROUND(
        SUM(CASE WHEN Global_Sales >= 5 THEN Global_Sales ELSE 0 END) * 100.0
        / SUM(Global_Sales), 1
    ) AS RevenueFromHitsPct
FROM
    GameTiers
GROUP BY
    Publisher
HAVING
    COUNT(*) >= 20
ORDER BY
    HitRatePct DESC, TotalRevenue_M DESC;
