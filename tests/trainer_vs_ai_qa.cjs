const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { chromium } = require(
  process.env.PLAYWRIGHT_MODULE
    || "/Users/pauldudko/.npm/_npx/420ff84f11983ee5/node_modules/playwright",
);

const PYTHON = "/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python";
const REPO = path.resolve(__dirname, "..");
const QA_ROOT = "/private/tmp/lionjudo-trainer-ai-qa";

function parseArguments(argv) {
  if (argv.length !== 2 || !["--session-dir", "--base-url"].includes(argv[0])) {
    throw new Error("Usage: node tests/trainer_vs_ai_qa.cjs --session-dir PATH | --base-url URL");
  }
  return { mode: argv[0], value: argv[1] };
}

async function waitForSession(baseUrl, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/api/session`);
      if (response.ok) return response.json();
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw lastError || new Error("Lokalni server nije spreman");
}

function assertPrivateV3(review) {
  assert.equal(review.version, 3);
  assert.equal(review.session_ready, true);
  assert.ok(Array.isArray(review.derived_media_manifest));
  assert.ok(review.derived_media_manifest.length > 0);
  assert.ok(review.derived_media_manifest.every((row) => row.privacy_verified === true));
}

async function startIsolatedSession(sourceDir) {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "lionjudo-trainer-ai-"));
  const sessionDir = path.join(temporaryRoot, "session");
  fs.cpSync(path.resolve(sourceDir), sessionDir, { recursive: true, dereference: true });
  const port = 8767;
  const child = spawn(
    PYTHON,
    ["tools/video_review.py", "serve", "--session-dir", sessionDir, "--port", String(port)],
    { cwd: REPO, stdio: ["ignore", "pipe", "pipe"] },
  );
  let logs = "";
  child.stdout.on("data", (chunk) => { logs += chunk; });
  child.stderr.on("data", (chunk) => { logs += chunk; });
  child.on("exit", (code) => {
    if (code && !logs.includes("SIGTERM")) process.stderr.write(logs);
  });
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    async cleanup() {
      if (child.exitCode === null) {
        child.kill("SIGTERM");
        await new Promise((resolve) => child.once("exit", resolve));
      }
      fs.rmSync(temporaryRoot, { recursive: true, force: true });
    },
  };
}

async function scoreCanvas(page) {
  return page.locator("canvas[data-metric]").evaluateAll((canvases) => canvases.some((canvas) => {
    const context = canvas.getContext("2d");
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    if (data.length < 8) return false;
    const first = `${data[0]}:${data[1]}:${data[2]}:${data[3]}`;
    for (let index = 4; index < data.length; index += 4) {
      if (`${data[index]}:${data[index + 1]}:${data[index + 2]}:${data[index + 3]}` !== first) return true;
    }
    return false;
  }));
}

async function runViewport(browser, baseUrl, viewport, screenshotName, exerciseFlow, normalIndex) {
  const context = await browser.newContext({ viewport });
  const page = await context.newPage();
  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 15000 });
  await page.waitForFunction(
    () => document.querySelectorAll("#event-list button").length === 3,
    null,
    { timeout: 10000 },
  );
  const normalEvent = page.locator("#event-list button:not(.injury)")
    .nth(normalIndex);
  await normalEvent.click();
  await page.waitForFunction(
    () => [...document.querySelectorAll("video")].every((video) => video.readyState >= 2),
    null,
    { timeout: 20000 },
  );

  assert.equal(await page.locator("#ai-duel").isHidden(), true);
  assert.equal(await page.locator("#imu-panel").isHidden(), true);
  assert.equal(await page.locator("#system-facts").isHidden(), true);

  if (exerciseFlow) {
    await page.locator("input[name='visibility'][value='dovoljno_vidljivo']").locator("..").click();
    await page.locator("#confirmed-technique").fill(
      normalIndex === 0 ? "Tai-otoshi" : "Morote-seoi-nage",
    );
    await page.locator("input[name='trainer-score'][value='4']").locator("..").click();
    await page.locator("#trainer-reason").fill("Rotacija kasni u ulasku.");
    await page.locator("#add-current-second").click();
    assert.match(await page.locator("#trainer-reason").inputValue(), /\[\d+\.\d{3} s\]/);
    const lockResponse = page.waitForResponse(
      (response) => response.url().endsWith("/trainer-assessments")
        && response.request().method() === "POST",
    );
    await page.locator("#lock-assessment-button").click();
    assert.equal((await lockResponse).status(), 200);
    const revealResponse = page.waitForResponse(
      (response) => response.url().endsWith("/ai-reveal")
        && response.request().method() === "POST",
    );
    await page.locator("#reveal-ai-button").click();
    assert.equal((await revealResponse).status(), 200);
    await page.locator("#ai-duel").waitFor({ state: "visible" });
    assert.match(await page.locator("#duel-delta").textContent(), /AI odstupa za \d poena\. Odbrani procenu\./);
    assert.equal(await page.locator("#imu-panel").isVisible(), true);
    assert.equal(await page.locator("#system-facts").isVisible(), true);
    assert.equal(await page.locator("#imu-panel .imu-value").count(), 8);
    assert.equal(await page.locator("input[name='ai-relation']").count(), 3);
    await page.locator("input[name='ai-relation'][value='slazem_se']").locator("..").click();
    await page.locator("#feedback-reason").fill("Prihvatam činjenice; naziv ostaje trenerski.");
    const feedbackResponse = page.waitForResponse(
      (response) => response.url().endsWith("/ai-feedback")
        && response.request().method() === "PUT",
    );
    await page.locator("#save-feedback-button").click();
    assert.equal((await feedbackResponse).status(), 200);
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => document.querySelector("#ai-duel")?.hidden === false);
    assert.equal(
      await page.locator("input[name='ai-relation'][value='slazem_se']").isChecked(),
      true,
    );
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth === window.innerWidth),
      true,
    );
    fs.mkdirSync(QA_ROOT, { recursive: true });
    await page.screenshot({
      path: path.join(QA_ROOT, screenshotName.replace(".png", "-ai.png")),
      fullPage: true,
    });
  }

  await page.locator("#event-list button.injury").click();
  assert.equal(await page.locator("#lock-assessment-button").isDisabled(), true);
  assert.match(await page.locator("#visibility-state").textContent(), /samo za čitanje/);
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth === window.innerWidth),
    true,
  );
  assert.equal(await scoreCanvas(page), true);
  fs.mkdirSync(QA_ROOT, { recursive: true });
  await page.screenshot({ path: path.join(QA_ROOT, screenshotName), fullPage: true });
  await context.close();
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  let lifecycle = null;
  let baseUrl = args.value;
  let browser;
  try {
    if (args.mode === "--session-dir") {
      lifecycle = await startIsolatedSession(args.value);
      baseUrl = lifecycle.baseUrl;
    }
    const preflight = await waitForSession(baseUrl);
    assertPrivateV3(preflight);
    browser = await chromium.launch({ headless: true });
    await runViewport(browser, baseUrl, { width: 1440, height: 1000 }, "desktop.png", true, 0);
    await runViewport(browser, baseUrl, { width: 390, height: 844 }, "mobile.png", true, 1);
    process.stdout.write("trainer-vs-ai QA OK\n");
  } finally {
    if (browser) await browser.close();
    if (lifecycle) await lifecycle.cleanup();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
