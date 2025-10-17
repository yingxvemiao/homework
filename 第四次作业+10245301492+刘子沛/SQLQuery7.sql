USE ESI;
GO

-- ��7�⣺����ȫ��ͬ�����ڸ�ѧ�Ƶı���
SELECT
    region_group,                     -- �������ޡ�ŷ�ޡ������ȣ�
    discipline,                       -- ѧ��
    COUNT(*) AS institution_count,    -- �ϰ������
    AVG([rank]) AS avg_rank,          -- ƽ������
    SUM(top_papers) AS total_top_papers,  -- �߱�����������
    SUM(docs) AS total_docs,          -- ��������
    SUM(cites) AS total_cites,        -- ��������
    ROUND(SUM(cites)*1.0/NULLIF(SUM(docs),0),2) AS avg_cites_per_paper  -- ƽ��ÿƪ����
FROM dbo.esi_rankings
WHERE region_group IS NOT NULL
GROUP BY region_group, discipline
ORDER BY region_group, avg_rank;