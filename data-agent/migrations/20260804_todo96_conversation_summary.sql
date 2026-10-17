-- TODO-96: append-only conversation summary authority.
-- Apply to the Meta MySQL database before enabling memory.summary_enabled.

-- Make pre-existing memory projection claims recoverable after worker crash.
ALTER TABLE `conversation_memory_outbox`
  ADD COLUMN `claimed_at` DATETIME NULL,
  ADD COLUMN `lease_until` DATETIME NULL,
  ADD COLUMN `worker_id` VARCHAR(80) NULL;

CREATE TABLE IF NOT EXISTS `conversation_summary_segment` (
  `summary_id` VARCHAR(36) NOT NULL,
  `owner_id` VARCHAR(36) NOT NULL,
  `workspace_id` VARCHAR(36) NOT NULL,
  `chat_id` VARCHAR(64) NOT NULL,
  `level` INT NOT NULL DEFAULT 0,
  `source_start_message_id` INT NOT NULL,
  `source_end_message_id` INT NOT NULL,
  `source_message_ids` JSON NOT NULL,
  `child_summary_ids` JSON NOT NULL,
  `source_content_hash` VARCHAR(80) NOT NULL,
  `content_masked` TEXT NOT NULL,
  `structured_facts` JSON NOT NULL,
  `open_items` JSON NOT NULL,
  `source_token_count` INT NOT NULL DEFAULT 0,
  `summary_token_count` INT NOT NULL DEFAULT 0,
  `summary_schema_version` VARCHAR(80) NOT NULL,
  `extractor_version` VARCHAR(80) NOT NULL,
  `model_version` VARCHAR(128) NULL,
  `prompt_version` VARCHAR(80) NULL,
  `projection_version` VARCHAR(160) NOT NULL,
  `status` VARCHAR(24) NOT NULL DEFAULT 'active',
  `superseded_by_id` VARCHAR(36) NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `valid_to` DATETIME NULL,
  `deleted_at` DATETIME NULL,
  PRIMARY KEY (`summary_id`),
  UNIQUE KEY `uq_conversation_summary_identity`
    (`owner_id`,`workspace_id`,`chat_id`,`level`,`source_content_hash`,`extractor_version`),
  KEY `ix_conversation_summary_scope_chat_status`
    (`owner_id`,`workspace_id`,`chat_id`,`status`,`level`),
  KEY `ix_conversation_summary_source_range`
    (`owner_id`,`workspace_id`,`chat_id`,`source_start_message_id`,`source_end_message_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `conversation_summary_signal` (
  `signal_id` VARCHAR(36) NOT NULL,
  `summary_id` VARCHAR(36) NOT NULL,
  `owner_id` VARCHAR(36) NOT NULL,
  `workspace_id` VARCHAR(36) NOT NULL,
  `chat_id` VARCHAR(64) NOT NULL,
  `signal_type` VARCHAR(24) NOT NULL,
  `normalized_value` VARCHAR(255) NOT NULL,
  `weight` DOUBLE NOT NULL DEFAULT 1,
  `source_order` INT NOT NULL DEFAULT 0,
  `source_message_ids` JSON NOT NULL,
  `extractor_version` VARCHAR(80) NOT NULL,
  `sensitivity` VARCHAR(24) NOT NULL DEFAULT 'masked',
  `status` VARCHAR(24) NOT NULL DEFAULT 'active',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `deleted_at` DATETIME NULL,
  PRIMARY KEY (`signal_id`),
  UNIQUE KEY `uq_conversation_summary_signal_identity`
    (`summary_id`,`signal_type`,`normalized_value`,`extractor_version`),
  KEY `ix_conversation_summary_signal_lookup`
    (`owner_id`,`workspace_id`,`chat_id`,`status`,`normalized_value`),
  KEY `ix_conversation_summary_signal_summary` (`summary_id`,`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `conversation_summary_state` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `owner_id` VARCHAR(36) NOT NULL,
  `workspace_id` VARCHAR(36) NOT NULL,
  `chat_id` VARCHAR(64) NOT NULL,
  `version` INT NOT NULL DEFAULT 0,
  `covered_through_message_id` INT NULL,
  `active_segment_count` INT NOT NULL DEFAULT 0,
  `last_source_hash` VARCHAR(80) NULL,
  `schema_version` VARCHAR(80) NOT NULL,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_conversation_summary_state_scope_chat`
    (`owner_id`,`workspace_id`,`chat_id`),
  KEY `ix_conversation_summary_state_scope`
    (`owner_id`,`workspace_id`,`chat_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
