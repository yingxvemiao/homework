USE ESI;
GO

-- ��ѯ����ʦ����ѧ�ڸ�ѧ�Ƶ�ESI���������ָ��
SELECT
    discipline,              -- ѧ����
    [rank],                  -- ����
    docs,                    -- ������
    cites,                   -- ������
    cites_per_paper,         -- ÿƪ����������
    top_papers               -- �߱���������
FROM dbo.esi_rankings
WHERE institution LIKE N'%EAST CHINA NORMAL UNIVERSITY%'    -- ģ��ƥ�������
ORDER BY [rank];                            -- �������������У���ֵС����������ǰ��
