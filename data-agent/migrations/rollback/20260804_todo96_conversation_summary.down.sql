-- Disable memory.summary_enabled before applying this rollback.
-- Summary data is derived; the authoritative conversation transcript is untouched.
DROP TABLE IF EXISTS `conversation_summary_signal`;
DROP TABLE IF EXISTS `conversation_summary_state`;
DROP TABLE IF EXISTS `conversation_summary_segment`;
ALTER TABLE `conversation_memory_outbox`
  DROP COLUMN `worker_id`,
  DROP COLUMN `lease_until`,
  DROP COLUMN `claimed_at`;
