SELECT erijobid AS job_id
  , lowsalary + (lowsalary * firstyearbonuspct) AS low_comp
  , highsalary + (highsalary * highyearbonuspct) AS high_comp
FROM sa.CompensationByExperience
WHERE releaseid = (SELECT MAX(releaseid) FROM dbo.ProductJob)
  AND erinationid = 193
