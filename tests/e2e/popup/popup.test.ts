import { test, expect } from '../fixtures/extension';

test.describe('Popup UI', () => {
  test('打开并显示标题', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/popup.html`);

    // CardTitle renders as h2 with text "视频 GPU 超分辨率"
    await expect(
      page.locator('h2', { hasText: /视频 GPU 超分辨率/i })
    ).toBeVisible();
  });

  test('显示连接状态', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/popup.html`);

    // 应该显示连接标签（已连接 或 不可用 或 连接中）
    await expect(page.locator('.ui-badge').first()).toBeVisible();
  });

  test('显示引擎选择器', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/popup.html`);

    await expect(page.locator('select[aria-label="引擎"]')).toBeVisible();
  });

  test('显示重新扫描视频按钮', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/popup.html`);

    await expect(
      page.locator('button', { hasText: /重新扫描视频/i })
    ).toBeVisible();
  });

  test('显示完整设置按钮', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/popup.html`);

    await expect(
      page.locator('button', { hasText: /完整设置/i })
    ).toBeVisible();
  });
});
