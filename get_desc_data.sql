(SELECT f.erijobid AS job_id
  , d.title + '. ' + d.ShortDesc AS title_desc
  , f.ajblvidp AS flsa
FROM sa.Sadescriptions as d
INNER JOIN sa.fname AS f
  ON d.edot=f.jobdot
INNER JOIN dbo.JobNation as jn
  ON f.erijobid=jn.erijobid
INNER JOIN dbo.ProductJob as pj
  ON f.erijobid=pj.erijobid
WHERE jn.erinationid = 193
  AND pj.ReleaseId = (SELECT MAX(ReleaseId) FROM dbo.ProductJob))
UNION
(SELECT f.erijobid AS job_id
  , d.title + '. ' + d.LongDesc AS title_desc
  , f.ajblvidp AS flsa
FROM sa.Sadescriptions as d
INNER JOIN sa.fname AS f
  ON d.edot=f.jobdot
INNER JOIN dbo.JobNation as jn
  ON f.erijobid=jn.erijobid
INNER JOIN dbo.ProductJob as pj
  ON f.erijobid=pj.erijobid
WHERE jn.erinationid = 193
  AND pj.ReleaseId = (SELECT MAX(ReleaseId) FROM dbo.ProductJob))
UNION
(SELECT d.erijobid AS job_id
  , d.title + '. ' + d.desc_matched AS title_desc
  , f.ajblvidp AS flsa
FROM dbo.Descriptions_Matched AS d
INNER JOIN sa.fname AS f
  ON d.erijobid=f.erijobid
INNER JOIN dbo.JobNation as jn
  ON f.erijobid=jn.erijobid
INNER JOIN dbo.ProductJob as pj
  ON f.erijobid=pj.erijobid
--WHERE quality_match = 1
WHERE jn.erinationid = 193
  AND pj.ReleaseId = (SELECT MAX(ReleaseId) FROM dbo.ProductJob))
UNION
(SELECT d.erijobid AS job_id
  , d.cbtitle + '. ' + d.cbdescription AS title_desc
  , f.ajblvidp AS flsa
FROM dbo.Descriptions_Matched_CB AS d
INNER JOIN sa.fname AS f
  ON d.erijobid=f.erijobid
INNER JOIN dbo.JobNation as jn
  ON f.erijobid=jn.erijobid
INNER JOIN dbo.ProductJob as pj
  ON f.erijobid=pj.erijobid
WHERE jn.erinationid = 193
  AND pj.ReleaseId = (SELECT MAX(ReleaseId) FROM dbo.ProductJob))
