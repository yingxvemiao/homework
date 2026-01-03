USE ESI;
GO

-- 第6题：分析中国大陆高校在各学科的表现
SELECT
    discipline,                        -- 学科
    COUNT(*) AS institution_count,     -- 上榜机构数量
    AVG([rank]) AS avg_rank,           -- 平均排名
    SUM(top_papers) AS total_top_papers, -- 各学科高被引论文总数
    SUM(docs) AS total_docs,           -- 各学科论文总数
    SUM(cites) AS total_cites,         -- 各学科引用总数
    ROUND(SUM(cites) * 1.0 / NULLIF(SUM(docs), 0), 2) AS avg_cites_per_paper -- 平均每篇引用
FROM dbo.esi_rankings
WHERE country_region LIKE N'%CHINA MAINLAND%'  -- 限定中国大陆高校
GROUP BY discipline
ORDER BY avg_rank;                             -- 按平均排名升序排列