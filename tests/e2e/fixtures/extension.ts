import { test as base, chromium, type BrowserContext } from '@playwright/test';

export const test = base.extend<{
  extensionContext: BrowserContext;
  extensionId: string;
}>({
  extensionContext: async ({}, use) => {
    let context: BrowserContext | null = null;
    let retries = 3;

    while (retries > 0) {
      try {
        context = await chromium.launchPersistentContext('', {
          headless: false,
          args: [
            `--disable-extensions-except=./dist`,
            `--load-extension=./dist`,
          ],
          timeout: 30000,
        });
        break;
      } catch (error) {
        retries--;
        if (retries === 0) throw error;
        await new Promise(r => setTimeout(r, 1000));
      }
    }

    await use(context!);
    await context!.close();
  },
  extensionId: async ({ extensionContext }, use) => {
    let extensionId = '';

    // 等待扩展加载完成
    await extensionContext.waitForEvent('serviceworker', { timeout: 10000 });

    // 尝试从 service worker 获取
    const background = extensionContext.serviceWorkers()[0];
    if (background) {
      extensionId = new URL(background.url()).hostname;
    }

    // 如果还是没有，从页面获取
    if (!extensionId) {
      for (const page of extensionContext.pages()) {
        if (page.url().includes('chrome-extension://')) {
          extensionId = new URL(page.url()).hostname;
          break;
        }
      }
    }

    if (!extensionId) {
      throw new Error('无法获取扩展 ID，请确保扩展已正确构建');
    }

    await use(extensionId);
  },
});

export { expect } from '@playwright/test';
