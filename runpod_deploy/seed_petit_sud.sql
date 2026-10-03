-- Seed démo "Le Petit Sud" : 8 tables + 10 réservations (vendredi 2026-09-25 soir)
-- + clients (pour l'affichage des noms) + appels (durées). Transactionnel.
-- Idempotent sur tables/clients/appels ; réservations = à lancer UNE fois.
\set ON_ERROR_STOP on
BEGIN;

-- 0) Garde : le restaurant doit exister
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM restaurants
                 WHERE id = 'b995eb7a-b9fe-437d-b959-c2f618456b97') THEN
    RAISE EXCEPTION 'Restaurant Le Petit Sud (b995eb7a...) introuvable — abandon';
  END IF;
END $$;

-- 1) Tables (8) — idempotent par nom
INSERT INTO tables (restaurant_id, name, capacity)
SELECT 'b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid, v.name, v.capacity
FROM (VALUES
  ('T1',4),('T2',4),('T3',4),('T4',2),
  ('Table 5',8),('T6',6),('T7',6),('Table 8',6)
) AS v(name, capacity)
WHERE NOT EXISTS (
  SELECT 1 FROM tables t
  WHERE t.restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid
    AND t.name = v.name
);

-- 2) Clients (10) — idempotent par (restaurant, phone)
INSERT INTO customers (restaurant_id, phone, last_name)
SELECT 'b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid, v.phone, v.last_name
FROM (VALUES
  ('+33600000001','Moreau'),
  ('+33600000002','Benali'),
  ('+33600000003','Lefèvre'),
  ('+33600000004','Da Silva'),
  ('+33600000005','N''Diaye'),
  ('+33600000006','Chevalier'),
  ('+33600000007','Perrin'),
  ('+33600000008','Ouedraogo'),
  ('+33600000009','Marchand'),
  ('+33600000010','Roussel')
) AS v(phone, last_name)
ON CONFLICT (restaurant_id, phone) DO NOTHING;

-- 3) Appels (7 lignes Bot, avec durées) — idempotent par twilio_call_sid
INSERT INTO calls (restaurant_id, customer_id, twilio_call_sid, direction, status,
                   outcome, captured, duration_seconds, started_at, ended_at)
SELECT 'b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid, c.id, v.sid, 'inbound', 'completed',
       v.outcome, TRUE, v.dur, v.starts, v.starts + (v.dur || ' seconds')::interval
FROM (VALUES
  ('seed-petitsud-benali',   '+33600000002', 41, 'reservation_created', TIMESTAMPTZ '2026-09-25 19:15:00+02'),
  ('seed-petitsud-lefevre',  '+33600000003', 52, 'reservation_created', TIMESTAMPTZ '2026-09-25 19:30:00+02'),
  ('seed-petitsud-dasilva',  '+33600000004', 33, 'reservation_created', TIMESTAMPTZ '2026-09-25 19:30:00+02'),
  ('seed-petitsud-ndiaye',   '+33600000005', 68, 'reservation_created', TIMESTAMPTZ '2026-09-25 20:00:00+02'),
  ('seed-petitsud-perrin',   '+33600000007', 47, 'reservation_created', TIMESTAMPTZ '2026-09-25 20:30:00+02'),
  ('seed-petitsud-ouedraogo','+33600000008', 39, 'reservation_created', TIMESTAMPTZ '2026-09-25 21:00:00+02'),
  ('seed-petitsud-roussel',  '+33600000010', 72, 'callback_requested',  TIMESTAMPTZ '2026-09-25 21:45:00+02')
) AS v(sid, phone, dur, outcome, starts)
JOIN customers c ON c.restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid AND c.phone=v.phone
WHERE NOT EXISTS (SELECT 1 FROM calls k WHERE k.twilio_call_sid = v.sid);

-- 4) Réservations (10) — durée d'occupation 1h45 (pas de chevauchement sur T4 / Table 5)
INSERT INTO reservations (restaurant_id, customer_id, table_id, call_id,
                          starts_at, ends_at, party_size, status, source, notes, cancelled_at)
SELECT 'b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid, cu.id, tb.id, ca.id,
       v.starts, v.starts + INTERVAL '1 hour 45 minutes', v.party, v.status, v.source,
       NULLIF(v.notes, ''),
       CASE WHEN v.status='cancelled' THEN now() ELSE NULL END
FROM (VALUES
  ('+33600000001','T4',      NULL,                     TIMESTAMPTZ '2026-09-25 19:00:00+02', 2, 'confirmed','web',    ''),
  ('+33600000002','T1',      'seed-petitsud-benali',   TIMESTAMPTZ '2026-09-25 19:15:00+02', 4, 'confirmed','callbot','Allergie arachide'),
  ('+33600000003','T6',      'seed-petitsud-lefevre',  TIMESTAMPTZ '2026-09-25 19:30:00+02', 6, 'confirmed','callbot','Table calme si possible'),
  ('+33600000004','Table 5', 'seed-petitsud-dasilva',  TIMESTAMPTZ '2026-09-25 19:30:00+02', 2, 'cancelled','callbot','Annulation par téléphone'),
  ('+33600000005','Table 5', 'seed-petitsud-ndiaye',   TIMESTAMPTZ '2026-09-25 20:00:00+02', 8, 'confirmed','callbot','Anniversaire, gâteau apporté'),
  ('+33600000006','T2',      NULL,                     TIMESTAMPTZ '2026-09-25 20:00:00+02', 4, 'confirmed','manual', 'Habitués, même table'),
  ('+33600000007','T3',      'seed-petitsud-perrin',   TIMESTAMPTZ '2026-09-25 20:30:00+02', 4, 'confirmed','callbot','Poussette, place bébé'),
  ('+33600000008','T7',      'seed-petitsud-ouedraogo',TIMESTAMPTZ '2026-09-25 21:00:00+02', 6, 'confirmed','callbot',''),
  ('+33600000009','T4',      NULL,                     TIMESTAMPTZ '2026-09-25 21:15:00+02', 2, 'confirmed','web',    '2e service sur T4'),
  ('+33600000010','Table 8', 'seed-petitsud-roussel',  TIMESTAMPTZ '2026-09-25 21:45:00+02', 6, 'pending',  'callbot','Rappel demandé, créneau à confirmer')
) AS v(phone, table_name, call_sid, starts, party, status, source, notes)
JOIN customers cu ON cu.restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid AND cu.phone=v.phone
JOIN tables tb    ON tb.restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid AND tb.name=v.table_name
LEFT JOIN calls ca ON ca.twilio_call_sid = v.call_sid;

-- 5) Table de liaison reservation_tables (indispensable pour la détection d'occupation)
INSERT INTO reservation_tables (reservation_id, table_id)
SELECT r.id, r.table_id
FROM reservations r
WHERE r.restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid
  AND r.table_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM reservation_tables rt
    WHERE rt.reservation_id=r.id AND rt.table_id=r.table_id
  );

-- Récap
SELECT
  (SELECT count(*) FROM tables       WHERE restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid) AS tables,
  (SELECT count(*) FROM customers    WHERE restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid) AS clients,
  (SELECT count(*) FROM calls        WHERE restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid) AS appels,
  (SELECT count(*) FROM reservations WHERE restaurant_id='b995eb7a-b9fe-437d-b959-c2f618456b97'::uuid) AS reservations;

COMMIT;
