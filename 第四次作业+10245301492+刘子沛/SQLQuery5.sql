USE ESI;
GO

-- 查询华东师范大学在各学科的ESI排名及相关指标
SELECT
    discipline,              -- 学科名
    [rank],                  -- 排名
    docs,                    -- 论文数
    cites,                   -- 引用数
    cites_per_paper,         -- 每篇论文引用数
    top_papers               -- 高被引论文数
FROM dbo.esi_rankings
WHERE institution LIKE N'%EAST CHINA NORMAL UNIVERSITY%'    -- 模糊匹配机构名
ORDER BY [rank];                            -- 按排名升序排列（数值小代表排名靠前）
