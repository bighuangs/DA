-- TODO-96 follow-up: align summary authority with the pre-existing
-- conversation_memory_outbox / conversation tables.
--
-- MySQL treats comparisons between two IMPLICIT, incompatible utf8mb4
-- collations as an error.  The original summary migration used the database
-- default collation, while an older Meta schema may use utf8mb4_general_ci.
-- Backfill readiness compares outbox.memory_id with segment.summary_id, so all
-- TODO-96 authority tables must share the established Meta collation.
--
-- Run only after 20260804_todo96_conversation_summary.sql.  CONVERT is safe to
-- repeat in outcome, although it rebuilds a table and should remain a managed
-- maintenance-window migration once rows exist.

ALTER TABLE `conversation_summary_segment`
  CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

ALTER TABLE `conversation_summary_signal`
  CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

ALTER TABLE `conversation_summary_state`
  CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
