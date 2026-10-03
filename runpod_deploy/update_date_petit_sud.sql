-- Décale le jeu de démo "Le Petit Sud" du 2026-09-25 au 2026-09-23 (-2 jours), mêmes heures.
\set ON_ERROR_STOP on
BEGIN;

UPDATE reservations r
SET starts_at = r.starts_at - INTERVAL '2 days',
    ends_at   = r.ends_at   - INTERVAL '2 days'
WHERE r.restaurant_id = 'b995eb7a-b9fe-437d-b959-c2f618456b97'
  AND r.customer_id IN (
    SELECT id FROM customers
    WHERE restaurant_id = 'b995eb7a-b9fe-437d-b959-c2f618456b97'
      AND phone LIKE '+336000000%'
  );

UPDATE calls
SET started_at = started_at - INTERVAL '2 days',
    ended_at   = ended_at   - INTERVAL '2 days'
WHERE twilio_call_sid LIKE 'seed-petitsud-%';

SELECT to_char(r.starts_at AT TIME ZONE 'Europe/Paris', 'YYYY-MM-DD HH24:MI') AS quand,
       c.last_name AS nom, r.party_size AS cv, t.name AS tbl, r.source, r.status
FROM reservations r
LEFT JOIN customers c ON c.id = r.customer_id
LEFT JOIN tables    t ON t.id = r.table_id
WHERE r.restaurant_id = 'b995eb7a-b9fe-437d-b959-c2f618456b97'
  AND c.phone LIKE '+336000000%'
ORDER BY r.starts_at;

COMMIT;
