-- Remove the retired anonymous upload scope after current owner/workspace data
-- has been verified. Rows without the current tenant contract cannot be served
-- safely and are intentionally not migrated into an arbitrary workspace.

DELETE FROM `adhoc_dataset`
WHERE `owner_id` IS NULL
   OR `owner_id` = ''
   OR `workspace_id` IS NULL
   OR `workspace_id` = ''
   OR `asset_id` IS NULL
   OR `asset_id` = ''
   OR `version_id` IS NULL
   OR `version_id` = '';

ALTER TABLE `adhoc_dataset`
  DROP INDEX `ix_adhoc_dataset_owner_key`,
  DROP COLUMN `owner_key`,
  MODIFY COLUMN `owner_id` varchar(36) NOT NULL,
  MODIFY COLUMN `workspace_id` varchar(36) NOT NULL,
  MODIFY COLUMN `asset_id` varchar(64) NOT NULL,
  MODIFY COLUMN `version_id` varchar(64) NOT NULL,
  ADD UNIQUE KEY `uq_adhoc_dataset_version_id` (`version_id`),
  ADD KEY `ix_adhoc_dataset_asset_id` (`asset_id`),
  ADD KEY `ix_adhoc_dataset_owner_id` (`owner_id`),
  ADD KEY `ix_adhoc_dataset_workspace_id` (`workspace_id`),
  ADD KEY `ix_adhoc_dataset_scope_created`
    (`owner_id`, `workspace_id`, `created_at`, `id`);
