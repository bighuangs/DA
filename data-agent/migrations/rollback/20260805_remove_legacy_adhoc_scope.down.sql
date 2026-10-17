-- Schema-only rollback. Purged anonymous rows cannot be reconstructed.
ALTER TABLE `adhoc_dataset`
  DROP INDEX `uq_adhoc_dataset_version_id`,
  DROP INDEX `ix_adhoc_dataset_asset_id`,
  DROP INDEX `ix_adhoc_dataset_owner_id`,
  DROP INDEX `ix_adhoc_dataset_workspace_id`,
  DROP INDEX `ix_adhoc_dataset_scope_created`,
  ADD COLUMN `owner_key` varchar(128) NULL AFTER `id`,
  MODIFY COLUMN `owner_id` varchar(36) NULL,
  MODIFY COLUMN `workspace_id` varchar(36) NULL,
  MODIFY COLUMN `asset_id` varchar(64) NULL,
  MODIFY COLUMN `version_id` varchar(64) NULL;

UPDATE `adhoc_dataset`
SET `owner_key` = `owner_id`;

ALTER TABLE `adhoc_dataset`
  MODIFY COLUMN `owner_key` varchar(128) NOT NULL,
  ADD KEY `ix_adhoc_dataset_owner_key` (`owner_key`);
