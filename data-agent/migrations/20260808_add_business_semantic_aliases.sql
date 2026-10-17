-- Add governed, reusable business vocabulary to existing Meta column rows.
--
-- Every statement is append-only and guarded by JSON_CONTAINS.  Re-running the
-- migration is therefore safe and preserves the order/content of all aliases
-- that were curated before deployment.

START TRANSACTION;

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '成交金额')
WHERE `table_id` = 'fact_order'
  AND `name` = 'order_amount'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('成交金额'), '$');

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '单笔金额')
WHERE `table_id` = 'fact_order'
  AND `name` = 'order_amount'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('单笔金额'), '$');

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '成交件数')
WHERE `table_id` = 'fact_order'
  AND `name` = 'order_quantity'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('成交件数'), '$');

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '购买件数')
WHERE `table_id` = 'fact_order'
  AND `name` = 'order_quantity'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('购买件数'), '$');

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '订单数')
WHERE `table_id` = 'fact_order'
  AND `name` = 'order_id'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('订单数'), '$');

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '订单笔数')
WHERE `table_id` = 'fact_order'
  AND `name` = 'order_id'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('订单笔数'), '$');

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '商品')
WHERE `table_id` = 'dim_product'
  AND `name` = 'product_name'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('商品'), '$');

UPDATE `column_info`
SET `alias` = JSON_ARRAY_APPEND(COALESCE(`alias`, JSON_ARRAY()), '$', '产品')
WHERE `table_id` = 'dim_product'
  AND `name` = 'product_name'
  AND NOT JSON_CONTAINS(COALESCE(`alias`, JSON_ARRAY()), JSON_QUOTE('产品'), '$');

COMMIT;
