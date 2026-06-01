import { test as base, chromium, type BrowserContext } from '@playwright/test';
import path from 'path';

const EXTENSION_PATH = path.resolve(__dirname, '../../../dist');
const LOAD_TIMEOUT = 30000;
const RETRY_COUNT = 3;
const RETRY_DELAY = 1000;

export const test = base.extend<{
  extensionContext: BrowserContext;
  extensionId: string;
}>({
  extensionContext: async ({}, use) => {
    let context: BrowserContext | null = null;
    let retries = RETRY_COUNT;

    while (retries > 0) {
      try {
        context = await chromium.launchPersistentContext('', {
          headless: false,
          args: [
            `--disable-extensions-except=${EXTENSION_PATH}`,
            `--load-extension=${EXTENSION_PATH}`,
          ],
          timeout: LOAD_TIMEOUT,
        });
        break;
      } catch (error) {
        retries--;
        if (retries === 0) throw error;
        await new Promise(r => setTimeout(r, RETRY_DELAY));
      }
    }

    if (!context) {
      throw new Error('Failed to launch extension context after retries');
    }

    await use(context);
    await context.close();
  },
  extensionId: async ({ extensionContext }, use) => {
    let extensionId = '';

    // Try to get from service worker first (if background script exists)
    const workers = extensionContext.serviceWorkers();
    if (workers.length > 0) {
      extensionId = new URL(workers[0].url()).hostname;
    }

    // If no service worker, navigate to chrome://extensions to get the ID
    if (!extensionId) {
      const page = await extensionContext.newPage();
      await page.goto('chrome://extensions');
      await page.waitForTimeout(1000);

      // Get extension ID from the extensions page
      const extensionCards = await page.locator('extensions-item').all();
      for (const card of extensionCards) {
        const name = await card.locator('#name').textContent();
        if (name && name.includes('Video')) {
          extensionId = await card.getAttribute('id') || '';
          break;
        }
      }

      await page.close();
    }

    if (!extensionId) {
      throw new Error('无法获取扩展 ID，请确保扩展已正确构建');
    }

    await use(extensionId);
  },
});

export { expect } from '@playwright/test';
