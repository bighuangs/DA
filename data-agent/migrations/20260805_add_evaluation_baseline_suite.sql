-- Scope approved NL2SQL baselines by governed dataset suite. Before this
-- migration the only promotable suite was common_sql_v1, so every existing
-- registry row can be backfilled deterministically without reading artifacts.

ALTER TABLE `evaluation_baseline`
  ADD COLUMN `dataset_suite` VARCHAR(64) NULL AFTER `workspace_id`;

UPDATE `evaluation_baseline`
SET `dataset_suite` = 'common_sql_v1'
WHERE `dataset_suite` IS NULL OR `dataset_suite` = '';

ALTER TABLE `evaluation_baseline`
  MODIFY COLUMN `dataset_suite` VARCHAR(64) NOT NULL,
  DROP INDEX `ix_evaluation_baseline_status`,
  DROP INDEX `ix_evaluation_baseline_workspace_id`,
  ADD KEY `ix_evaluation_baseline_scope_status_reviewed`
    (`workspace_id`, `dataset_suite`, `status`, `reviewed_at`);
