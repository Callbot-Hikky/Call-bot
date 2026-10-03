SELECT to_char(r.starts_at AT TIME ZONE 'Europe/Paris', 'HH24:MI') AS h,
       c.last_name  AS nom,
       r.party_size AS cv,
       t.name       AS tbl,
       r.source,
       r.status,
       r.notes
FROM reservations r
LEFT JOIN customers c ON c.id = r.customer_id
LEFT JOIN tables    t ON t.id = r.table_id
WHERE r.restaurant_id = 'b995eb7a-b9fe-437d-b959-c2f618456b97'
  AND r.starts_at >= '2026-09-25 00:00:00+02'
  AND r.starts_at <  '2026-09-26 00:00:00+02'
ORDER BY r.starts_at;
