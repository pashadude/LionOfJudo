const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { chromium } = require(
  process.env.PLAYWRIGHT_MODULE
    || "/Users/pauldudko/.npm/_npx/420ff84f11983ee5/node_modules/playwright",
);

const baseUrl = process.argv[2];
const sessionDir = path.resolve(
  process.argv[3] || "/private/tmp/lionjudo-video-review-session/session",
);
if (!baseUrl) throw new Error("Usage: node tests/coach_event_recovery_qa.cjs BASE_URL SESSION_DIR");

const reviewPath = path.join(sessionDir, "review.json");
const reportPaths = ["izvestaj.csv", "izvestaj.md"].map((name) => path.join(sessionDir, name));
const eventsPath = path.join(sessionDir, "events");
const backupRoot = fs.mkdtempSync(path.join(os.tmpdir(), "lionjudo-recovery-"));

function loadReview() {
  return JSON.parse(fs.readFileSync(reviewPath, "utf8"));
}

function sourcePaths(review) {
  return ["sony", "iphone"].map((camera) => {
    const source = review.sources?.[camera]?.path || review[`${camera}_video`];
    const candidate = path.resolve(sessionDir, source);
    return candidate;
  });
}

function sourceFingerprint(filePath) {
  const stat = fs.statSync(filePath);
  return `${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`;
}

function snapshot() {
  fs.copyFileSync(reviewPath, path.join(backupRoot, "review.json"));
  for (const reportPath of reportPaths) fs.copyFileSync(reportPath, path.join(backupRoot, path.basename(reportPath)));
  fs.cpSync(eventsPath, path.join(backupRoot, "events"), { recursive: true });
  return loadReview();
}

function restore() {
  fs.copyFileSync(path.join(backupRoot, "review.json"), reviewPath);
  for (const reportPath of reportPaths) {
    fs.copyFileSync(path.join(backupRoot, path.basename(reportPath)), reportPath);
  }
  fs.rmSync(eventsPath, { recursive: true, force: true });
  fs.cpSync(path.join(backupRoot, "events"), eventsPath, { recursive: true });
}

function sameBytes(first, second) {
  return fs.readFileSync(first).equals(fs.readFileSync(second));
}

async function main() {
  const baseline = snapshot();
  const sourceFiles = sourcePaths(baseline);
  const sourceFingerprintsBefore = sourceFiles.map(sourceFingerprint);
  const browser = await chromium.launch({ headless: true });
  let recovery;
  try {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const page = await context.newPage();
    page.on("dialog", (dialog) => dialog.accept());
    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 10000 });
    await page.waitForFunction(
      () => document.querySelectorAll("#event-list button").length === 2,
      null,
      { timeout: 5000 },
    );

    const normalButton = page.locator("#event-list button:not(.injury)");
    assert.equal(await normalButton.count(), 1);
    assert.equal(await normalButton.getAttribute("data-event-id"), "e-001");
    await normalButton.click();

    const deleteResponse = page.waitForResponse(
      (response) => response.url().endsWith("/api/events/e-001")
        && response.request().method() === "DELETE",
    );
    await page.locator("#delete-button").click();
    assert.equal((await deleteResponse).status(), 200);
    await page.waitForFunction(
      () => document.querySelector("#event-list button.injury")?.getAttribute("aria-current") === "true",
      null,
      { timeout: 5000 },
    );

    assert.equal(await page.locator("#create-event-button").isEnabled(), true);
    assert.equal(await page.locator("#event-start").isEnabled(), true);
    assert.equal(await page.locator("#event-end").isEnabled(), true);
    assert.equal(await page.locator("#update-bounds-button").isDisabled(), true);
    assert.equal(await page.locator("#delete-button").isDisabled(), true);
    const draft = await page.locator("#event-start, #event-end").evaluateAll(
      (items) => items.map((item) => Number(item.value)),
    );
    const afterDelete = await page.evaluate(async () => (await fetch("/api/session")).json());
    const firstAnchor = Math.min(
      ...afterDelete.anchors.filter((anchor) => anchor.user_confirmed).map((anchor) => anchor.sony_s),
    );
    assert.ok(draft[0] >= firstAnchor);
    assert.ok(draft[1] <= afterDelete.injury_cutoff_s);
    assert.ok(draft[1] > draft[0]);

    const createResponse = page.waitForResponse(
      (response) => response.url().endsWith("/api/events") && response.request().method() === "POST",
    );
    await page.locator("#create-event-button").click();
    assert.equal((await createResponse).status(), 201);
    await page.waitForFunction(
      () => [...document.querySelectorAll("#event-list button:not(.injury)")]
        .some((button) => button.getAttribute("aria-current") === "true"),
      null,
      { timeout: 5000 },
    );
    const selected = await page.locator("#event-list button[aria-current='true']").getAttribute("data-event-id");
    assert.match(selected, /^e-coach-/);
    recovery = {
      injurySelectedAfterDelete: true,
      draftStart: draft[0],
      draftEnd: draft[1],
      createdEventSelected: selected,
    };
    await context.close();
  } finally {
    await browser.close();
    restore();
    const sourceFingerprintsAfter = sourceFiles.map(sourceFingerprint);
    const restoredReview = sameBytes(reviewPath, path.join(backupRoot, "review.json"));
    const restoredReports = reportPaths.every((reportPath) => (
      sameBytes(reportPath, path.join(backupRoot, path.basename(reportPath)))
    ));
    const restoredEvents = sameBytes(
      path.join(eventsPath, "e-001", "sony.mp4"),
      path.join(backupRoot, "events", "e-001", "sony.mp4"),
    ) && sameBytes(
      path.join(eventsPath, "e-001", "iphone.mp4"),
      path.join(backupRoot, "events", "e-001", "iphone.mp4"),
    );
    const sourcesUnchanged = sourceFingerprintsBefore.every(
      (value, index) => value === sourceFingerprintsAfter[index],
    );
    assert.equal(restoredReview, true);
    assert.equal(restoredReports, true);
    assert.equal(restoredEvents, true);
    assert.equal(sourcesUnchanged, true);
    const restored = loadReview();
    const event = restored.events.find((item) => item.event_id === "e-001");
    assert.deepEqual(
      {
        bounds: [event.sony_start_s, event.sony_end_s],
        technique: event.potvrdena_tehnika,
        score: event.ocena,
        note: event.napomena,
      },
      {
        bounds: [73.94153333333333, 77.2762],
        technique: "O-soto-gari",
        score: 4,
        note: "Real-session QA anotacija.",
      },
    );
    recovery = {
      ...recovery,
      restoredReview: true,
      restoredE001: true,
      restoredReports: true,
      sourcesUnchanged: true,
    };
  }
  fs.rmSync(backupRoot, { recursive: true, force: true });
  process.stdout.write(`${JSON.stringify(recovery, null, 2)}\n`);
}

main().catch((error) => {
  fs.rmSync(backupRoot, { recursive: true, force: true });
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
