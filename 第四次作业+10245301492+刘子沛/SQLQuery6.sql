USE ESI;
GO

-- ��6�⣺�����й���½��У�ڸ�ѧ�Ƶı���
SELECT
    discipline,                        -- ѧ��
    COUNT(*) AS institution_count,     -- �ϰ��������
    AVG([rank]) AS avg_rank,           -- ƽ������
    SUM(top_papers) AS total_top_papers, -- ��ѧ�Ƹ߱�����������
    SUM(docs) AS total_docs,           -- ��ѧ����������
    SUM(cites) AS total_cites,         -- ��ѧ����������
    ROUND(SUM(cites) * 1.0 / NULLIF(SUM(docs), 0), 2) AS avg_cites_per_paper -- ƽ��ÿƪ����
FROM dbo.esi_rankings
WHERE country_region LIKE N'%CHINA MAINLAND%'  -- �޶��й���½��У
GROUP BY discipline
ORDER BY avg_rank;                             -- ��ƽ��������������