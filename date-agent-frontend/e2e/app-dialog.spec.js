import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

test("连续 confirm 到 prompt 保留关闭周期并聚焦第二个输入框", async ({ page }) => {
  await openAs(page);
  await page.evaluate(async () => {
    const dialogs = await import("/src/composables/appDialog.js");
    window.__dialogChainValue = "pending";
    dialogs.confirmDialog({
      title: "继续命名？",
      message: "确认后立即打开名称输入弹层。",
      confirmLabel: "继续",
    }).then(async (confirmed) => {
      window.__dialogChainValue = confirmed
        ? await dialogs.promptDialog({
          title: "输入连续弹层名称",
          label: "连续弹层名称",
          value: "默认名称",
          confirmLabel: "保存名称",
        })
        : "cancelled";
    });
  });

  const confirm = page.getByRole("dialog", { name: "继续命名？" });
  await confirm.getByRole("button", { name: "继续" }).click();
  await expect(confirm).toHaveCount(0);
  const prompt = page.getByRole("dialog", { name: "输入连续弹层名称" });
  const input = prompt.getByLabel("连续弹层名称");
  await expect(input).toBeVisible();
  await expect(input).toBeFocused();
  await input.fill("连续切换成功");
  await prompt.getByRole("button", { name: "保存名称" }).click();
  await expect.poll(() => page.evaluate(() => window.__dialogChainValue)).toBe("连续切换成功");
});

test("仪表盘创建、重命名和删除使用可访问受控对话框", async ({ page }) => {
  const state = await openAs(page);
  let nativeDialogOpened = false;
  page.on("dialog", async (dialog) => {
    nativeDialogOpened = true;
    await dialog.dismiss();
  });

  await page.getByRole("button", { name: "仪表盘", exact: true }).click();
  const createButton = page.getByRole("button", { name: "新建仪表盘", exact: true }).first();
  const postCount = state.calls.filter((call) => call.path === "/api/dashboard" && call.method === "POST").length;

  await createButton.click();
  const createDialog = page.getByRole("dialog", { name: "新建仪表盘" });
  const createInput = createDialog.getByLabel("仪表盘名称");
  await expect(createInput).toBeFocused();
  await expect(createInput).toHaveValue("新仪表盘");
  await page.keyboard.press("Shift+Tab");
  await expect(createDialog.getByRole("button", { name: "创建" })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(createInput).toBeFocused();
  await createDialog.getByRole("button", { name: "取消" }).click();
  await expect(createDialog).toHaveCount(0);
  await expect(createButton).toBeFocused();
  expect(state.calls.filter((call) => call.path === "/api/dashboard" && call.method === "POST")).toHaveLength(postCount);

  await createButton.click();
  await createDialog.getByLabel("仪表盘名称").fill("季度经营看板");
  await createDialog.getByRole("button", { name: "创建" }).click();
  await expect(page.getByText("季度经营看板", { exact: true })).toBeVisible();
  const createCall = state.calls.findLast((call) => call.path === "/api/dashboard" && call.method === "POST");
  expect(createCall.body.name).toBe("季度经营看板");

  const createdRow = page.locator(".tree-row").filter({ hasText: "季度经营看板" });
  await createdRow.hover();
  await createdRow.getByRole("button", { name: "重命名 季度经营看板" }).click();
  const renameDialog = page.getByRole("dialog", { name: "重命名仪表盘" });
  await expect(renameDialog.getByLabel("新名称")).toBeFocused();
  await expect(renameDialog.getByLabel("新名称")).toHaveValue("季度经营看板");
  await renameDialog.getByLabel("新名称").fill("年度经营看板");
  await renameDialog.getByRole("button", { name: "保存名称" }).click();
  await expect(page.getByText("年度经营看板", { exact: true })).toBeVisible();
  const renameCall = state.calls.findLast((call) => /^\/api\/dashboard\/[^/]+$/.test(call.path) && call.method === "PUT");
  expect(renameCall.body.name).toBe("年度经营看板");

  const renamedRow = page.locator(".tree-row").filter({ hasText: "年度经营看板" });
  await renamedRow.hover();
  const deleteButton = renamedRow.getByRole("button", { name: "删除 年度经营看板" });
  const deleteCount = state.calls.filter((call) => call.method === "DELETE" && call.path.startsWith("/api/dashboard/")).length;
  await deleteButton.click();
  const deleteDialog = page.getByRole("dialog", { name: "删除仪表盘？" });
  await expect(deleteDialog).toContainText("永久删除");
  await deleteDialog.getByRole("button", { name: "取消" }).click();
  await expect(deleteButton).toBeFocused();
  expect(state.calls.filter((call) => call.method === "DELETE" && call.path.startsWith("/api/dashboard/"))).toHaveLength(deleteCount);

  await deleteButton.click();
  await deleteDialog.getByRole("button", { name: "确认删除" }).click();
  await expect(page.getByText("年度经营看板", { exact: true })).toHaveCount(0);
  expect(state.calls.some((call) => call.method === "DELETE" && call.path.startsWith("/api/dashboard/"))).toBeTruthy();
  expect(nativeDialogOpened).toBe(false);
});

test("危险索引操作支持取消、确认和触发按钮焦点恢复", async ({ page }) => {
  const state = await openAs(page);
  await page.getByRole("button", { name: "元知识" }).click();
  const rebuildButton = page.getByRole("button", { name: "覆盖重建 BGE-M3 混合索引" });

  await rebuildButton.click();
  const dialog = page.getByRole("dialog", { name: "覆盖重建 BGE-M3 混合索引？" });
  await expect(dialog).toHaveClass(/tone-danger/);
  await expect(dialog).toContainText("3 个 Qdrant collection");
  await dialog.getByRole("button", { name: "取消" }).click();
  await expect(dialog).toHaveCount(0);
  await expect(rebuildButton).toBeFocused();
  expect(state.calls.some((call) => call.path === "/api/admin/meta/rebuild")).toBe(false);

  await rebuildButton.click();
  await dialog.getByRole("button", { name: "覆盖重建" }).click();
  await expect(page.getByText("BGE-M3 混合索引已覆盖")).toBeVisible();
  expect(state.calls.some((call) => call.path === "/api/admin/meta/rebuild")).toBe(true);
});

test("问数结果可在 DOM 对话框中命名新仪表盘并加入", async ({ page }) => {
  const state = await openAs(page);
  await page.getByLabel("输入问题").fill("各地区销售额");
  await page.getByRole("button", { name: "发送问题" }).click();
  await page.getByRole("button", { name: "加入仪表盘" }).click();

  const picker = page.getByRole("dialog", { name: "加入仪表盘" });
  const createAndAdd = picker.getByRole("button", { name: "新建仪表盘并加入" });
  await createAndAdd.click();
  const prompt = page.getByRole("dialog", { name: "新建仪表盘并加入" });
  await expect(prompt.getByLabel("仪表盘名称")).toBeFocused();
  await prompt.getByRole("button", { name: "取消" }).click();
  await expect(picker).toBeVisible();
  await expect(createAndAdd).toBeFocused();

  await createAndAdd.click();
  await prompt.getByLabel("仪表盘名称").fill("问数结果看板");
  await prompt.getByRole("button", { name: "创建并加入" }).click();
  await expect(page.getByText("问数结果看板", { exact: true }).first()).toBeVisible();
  const createCall = state.calls.findLast((call) => call.path === "/api/dashboard" && call.method === "POST");
  expect(createCall.body.name).toBe("问数结果看板");
  expect(state.calls.some((call) => /^\/api\/dashboard\/[^/]+\/block$/.test(call.path) && call.method === "POST")).toBe(true);
});
