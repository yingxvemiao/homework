USE ESI;
GO

-- 第7题：分析全球不同区域在各学科的表现
SELECT
    region_group,                     -- 区域（亚洲、欧洲、北美等）
    discipline,                       -- 学科
    COUNT(*) AS institution_count,    -- 上榜机构数
    AVG([rank]) AS avg_rank,          -- 平均排名
    SUM(top_papers) AS total_top_papers,  -- 高被引论文总数
    SUM(docs) AS total_docs,          -- 论文总数
    SUM(cites) AS total_cites,        -- 引用总数
    ROUND(SUM(cites)*1.0/NULLIF(SUM(docs),0),2) AS avg_cites_per_paper  -- 平均每篇引用
FROM dbo.esi_rankings
WHERE region_group IS NOT NULL
GROUP BY region_group, discipline
ORDER BY region_group, avg_rank;