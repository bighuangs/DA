-- Rolling back loses per-suite approval identity. Preserve one deterministic
-- approved row per workspace and supersede the others before dropping the scope.

UPDATE `evaluation_baseline` AS target
JOIN (
  SELECT `id`
  FROM (
    SELECT
      `id`,
      ROW_NUMBER() OVER (
        PARTITION BY `workspace_id`
        ORDER BY `reviewed_at` DESC, `id` DESC
      ) AS `approval_rank`
    FROM `evaluation_baseline`
    WHERE `status` = 'approved'
  ) AS ranked
  WHERE ranked.`approval_rank` > 1
) AS demoted ON demoted.`id` = target.`id`
SET
  target.`status` = 'superseded',
  target.`superseded_at` = COALESCE(target.`superseded_at`, CURRENT_TIMESTAMP);

ALTER TABLE `evaluation_baseline`
  DROP INDEX `ix_evaluation_baseline_scope_status_reviewed`,
  ADD KEY `ix_evaluation_baseline_status` (`status`),
  ADD KEY `ix_evaluation_baseline_workspace_id` (`workspace_id`),
  DROP COLUMN `dataset_suite`;
