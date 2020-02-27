SELECT erijobid AS job_id
  , totalcash AS comp
  , revenuem AS cut_point
FROM xa.CompensationByRevenue
WHERE releaseid = (SELECT MAX(releaseid) FROM dbo.ProductJob)
  AND erinationid = 193
  AND revenuem IN (1, 100000)
