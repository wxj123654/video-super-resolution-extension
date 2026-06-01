import { test, expect } from '../fixtures/extension';

test.describe('Options Page', () => {
  test('打开并显示标题', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/options.html`);

    await expect(
      page.locator('text=/video gpu super resolution/i')
    ).toBeVisible();
  });

  test('显示诊断部分', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/options.html`);

    await expect(
      page.locator('h2', { hasText: /诊断/i })
    ).toBeVisible();
  });

  test('显示常规设置部分', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/options.html`);

    await expect(
      page.locator('h2', { hasText: /常规设置/i })
    ).toBeVisible();
  });

  test('显示精细调整部分', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/options.html`);

    await expect(
      page.locator('h2', { hasText: /精细调整/i })
    ).toBeVisible();
  });

  test('显示关于部分', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/options.html`);

    await expect(
      page.locator('h2', { hasText: /关于/i })
    ).toBeVisible();
  });

  test('显示版本号', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/options.html`);

    await expect(page.locator('text=/\\d+\\.\\d+\\.\\d+/')).toBeVisible();
  });

  test('显示运行诊断按钮', async ({ extensionId, page }) => {
    await page.goto(`chrome-extension://${extensionId}/options.html`);

    await expect(
      page.locator('button', { hasText: /运行诊断/i })
    ).toBeVisible();
  });
});
