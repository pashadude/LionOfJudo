const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(
  process.env.PLAYWRIGHT_MODULE
    || "/Users/pauldudko/.npm/_npx/420ff84f11983ee5/node_modules/playwright",
);

const [baseUrl, sessionDirArgument] = process.argv.slice(2);
if (!baseUrl || !sessionDirArgument || process.argv.length !== 4) {
  throw new Error(
    "Usage: node tests/coach_trainer_dataset_qa.cjs BASE_URL SESSION_DIR",
  );
}

const TRAINER_NAME = "Demo trener";
const WRESTLER_NAME = "Dusan";
const AI_ONLY_KEYS = new Set([
  "ai_procene",
  "imu_eksperimentalno",
  "procene_ai_predloga",
  "aktivni_duel",
  "ai_score",
  "evaluator_id",
  "predlozena_ocena",
  "pouzdanost_0_1",
  "dokazi",
  "ai_otkriven_u",
  "trener_revizija",
]);

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForSession(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/api/session`);
      if (response.ok) return response.json();
      lastError = new Error(`GET /api/session returned ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(150);
  }
  throw lastError || new Error("Local review server did not become ready");
}

function normalEventId(review) {
  const event = review.events?.find((item) => !(
    item.prijavljen_povredni_dogadjaj
    || item.iskljuceno_iz_statistike
    || item.status === "povreda"
  ));
  assert.ok(event?.event_id, "session must contain a normal review event");
  return event.event_id;
}

async function selectEventAndWaitForMedia(page, eventId) {
  await page.locator(`#event-list button[data-event-id='${eventId}']`).click();
  await page.waitForFunction(
    () => [...document.querySelectorAll("video")].length === 2
      && [...document.querySelectorAll("video")].every((video) => (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
        && Number.isFinite(video.duration)
        && video.duration > 0
        && video.error === null
      )),
    null,
    { timeout: 20000 },
  );
}

async function mediaSummary(page) {
  const media = await page.locator("video").evaluateAll((videos) => videos.map((video) => ({
    id: video.id,
    readyState: video.readyState,
    duration: video.duration,
    error: video.error ? video.error.code : null,
  })));
  assert.equal(media.length, 2, "exactly two review videos must be present");
  for (const video of media) {
    assert.ok(video.readyState >= 2, `${video.id} must have current media data`);
    assert.ok(video.duration > 0, `${video.id} must have a duration`);
    assert.equal(video.error, null, `${video.id} must not have a media error`);
  }
  return media;
}

async function fillValidAssessmentDraft(page) {
  await page.locator("input[name='visibility'][value='dovoljno_vidljivo']").locator("..").click();
  await page.locator("#confirmed-technique").fill("Tai-otoshi");
  await page.locator("input[name='trainer-score'][value='4']").locator("..").click();
  await page.locator("#trainer-reason").fill("QA proverava samo dostupnost zaključavanja.");
  await page.locator("#add-current-second").click();
  assert.match(await page.locator("#citation-list").textContent(), /\d+\.\d{3} s/);
}

async function verifyUnsavedIdentityKeepsLockDisabled(browser, eventId) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.route("**/api/session", async (route) => {
    const response = await route.fetch();
    const review = await response.json();
    delete review.participants;
    await route.fulfill({ response, json: review });
  });
  const page = await context.newPage();
  try {
    await page.goto(`${baseUrl}/?qa=unsaved-identity`, {
      waitUntil: "domcontentloaded",
      timeout: 15000,
    });
    await page.waitForFunction(
      () => document.querySelectorAll("#event-list button").length > 0,
      null,
      { timeout: 10000 },
    );
    await selectEventAndWaitForMedia(page, eventId);
    await fillValidAssessmentDraft(page);
    assert.equal(
      await page.locator("#lock-assessment-button").isDisabled(),
      true,
      "a valid draft without a saved identity must remain locked",
    );
  } finally {
    await context.close();
  }
}

async function saveAndVerifyParticipants(page, eventId) {
  await page.locator("#trainer-name").fill(TRAINER_NAME);
  await page.locator("#wrestler-name").fill(WRESTLER_NAME);
  const saved = page.waitForResponse(
    (response) => response.url().endsWith("/api/session/participants")
      && response.request().method() === "PUT",
  );
  await page.locator("#save-participants-button").click();
  assert.equal((await saved).status(), 200, "participant save must succeed");

  // Saving an identity alone must not make a lockable assessment.
  assert.equal(await page.locator("#lock-assessment-button").isDisabled(), true);

  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => document.querySelectorAll("#event-list button").length > 0,
    null,
    { timeout: 10000 },
  );
  assert.equal(await page.locator("#trainer-name").inputValue(), TRAINER_NAME);
  assert.equal(await page.locator("#wrestler-name").inputValue(), WRESTLER_NAME);
  await selectEventAndWaitForMedia(page, eventId);
  assert.equal(await page.locator("#lock-assessment-button").isDisabled(), true);

  await fillValidAssessmentDraft(page);
  assert.equal(await page.locator("#lock-assessment-button").isDisabled(), false);
}

function assertNoAiKeys(value, location = "dataset") {
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoAiKeys(item, `${location}[${index}]`));
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    assert.equal(AI_ONLY_KEYS.has(key), false, `${location}.${key} serializes AI-only data`);
    assertNoAiKeys(child, `${location}.${key}`);
  }
}

function safeGenerationPath(root, relativePath) {
  assert.equal(typeof relativePath, "string", "media path must be a string");
  assert.match(relativePath, /^events\/[A-Za-z0-9_-]+\/(sony|iphone)\.mp4$/);
  const candidate = path.resolve(root, relativePath);
  assert.ok(candidate.startsWith(`${root}${path.sep}`), "media path must remain in generation root");
  return candidate;
}

async function sha256(filePath) {
  return new Promise((resolve, reject) => {
    const digest = crypto.createHash("sha256");
    const input = fs.createReadStream(filePath);
    input.on("error", reject);
    input.on("data", (chunk) => digest.update(chunk));
    input.on("end", () => resolve(digest.digest("hex")));
  });
}

function activeGenerationRoot(sessionDir, generationId) {
  const sessionRoot = fs.realpathSync(sessionDir);
  const pointer = JSON.parse(fs.readFileSync(path.join(sessionRoot, "current-generation.json"), "utf8"));
  assert.equal(pointer.generation_id, generationId, "exports must use the active generation");
  const root = path.join(sessionRoot, ".review-generations", generationId);
  assert.equal(fs.statSync(root).isDirectory(), true, "active generation directory must exist");
  return root;
}

async function downloadAndVerifyExports(page, sessionDir) {
  const [datasetResponse, auditResponse] = await Promise.all([
    page.request.get(`${baseUrl}/trener_dataset.json`),
    page.request.get(`${baseUrl}/trener_assessment_audit.json`),
  ]);
  assert.equal(datasetResponse.ok(), true, "dataset JSON must download");
  assert.equal(auditResponse.ok(), true, "audit JSON must download");
  const dataset = await datasetResponse.json();
  const audit = await auditResponse.json();
  assert.equal(dataset.generation_id, audit.generation_id, "exports must share a generation");
  assert.match(dataset.generation_id, /^[0-9a-f]{32}$/);
  assert.deepEqual(dataset.participants, {
    trainer_name: TRAINER_NAME,
    wrestler_name: WRESTLER_NAME,
  });
  assertNoAiKeys(dataset);

  const generationRoot = activeGenerationRoot(sessionDir, dataset.generation_id);
  const review = JSON.parse(fs.readFileSync(path.join(generationRoot, "review.json"), "utf8"));
  const manifest = new Map((review.derived_media_manifest || []).map((row) => [row.relative_path, row]));
  const cleanExamples = dataset.training_examples || [];
  for (const example of cleanExamples) {
    assert.equal(example.generation_id, dataset.generation_id);
    assert.equal(example.assessment_phase, "pre_ai");
    assert.equal(example.assessment_revision, 1);
    assert.equal(example.training_eligible, true);
    for (const camera of ["sony", "iphone"]) {
      const reference = example.evidence?.[`${camera}_clip`];
      assert.ok(reference, `${camera} evidence must be present`);
      assert.equal(reference.review_url, `/media/${reference.bundle_relative_path}`);
      const manifestRow = manifest.get(reference.bundle_relative_path);
      assert.deepEqual(
        {
          media_type: manifestRow?.media_type,
          privacy_verified: manifestRow?.privacy_verified,
          failure_reason: manifestRow?.failure_reason,
        },
        { media_type: "event_clip", privacy_verified: true, failure_reason: null },
        `${camera} evidence must be privacy-bound to the active generation`,
      );
      assert.equal(await sha256(safeGenerationPath(generationRoot, reference.bundle_relative_path)), reference.sha256);
    }
  }

  const corrected = (audit.assessments || []).filter(
    (assessment) => Number(assessment.assessment_revision) > 1,
  );
  for (const assessment of corrected) {
    assert.equal(assessment.assessment_phase, "post_ai_correction");
    assert.equal(assessment.training_eligible, false);
    assert.ok(assessment.ineligibility_reasons?.includes("post_ai_correction"));
  }
  return {
    generationId: dataset.generation_id,
    cleanExamples: cleanExamples.length,
    auditAssessments: (audit.assessments || []).length,
    postAiCorrections: corrected.length,
  };
}

async function assertViewportLayout(browser, viewport, eventId) {
  const context = await browser.newContext({ viewport });
  const page = await context.newPage();
  try {
    await page.goto(`${baseUrl}/?qa=${viewport.width}x${viewport.height}`, {
      waitUntil: "domcontentloaded",
      timeout: 15000,
    });
    await page.waitForFunction(
      () => document.querySelectorAll("#event-list button").length > 0,
      null,
      { timeout: 10000 },
    );
    await selectEventAndWaitForMedia(page, eventId);
    const layout = await page.evaluate(() => {
      const visibleControls = [...document.querySelectorAll("button, a, input, select, textarea")]
        .map((element) => {
          const style = getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          return {
            tag: element.tagName,
            id: element.id,
            name: element.getAttribute("name"),
            rect: { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom },
            visible: style.display !== "none" && style.visibility !== "hidden"
              && rect.width > 4 && rect.height > 4 && Number(style.opacity) > 0,
          };
        })
        .filter((control) => control.visible);
      const overlaps = [];
      const outsideViewport = [];
      for (let first = 0; first < visibleControls.length; first += 1) {
        const control = visibleControls[first];
        if (control.rect.left < -0.5 || control.rect.right > window.innerWidth + 0.5) {
          outsideViewport.push({
            control: control.id || control.name || control.tag,
            left: control.rect.left,
            right: control.rect.right,
          });
        }
        for (let second = first + 1; second < visibleControls.length; second += 1) {
          const a = visibleControls[first];
          const b = visibleControls[second];
          const width = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left);
          const height = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top);
          if (width > 2 && height > 2) overlaps.push([a.id || a.name || a.tag, b.id || b.name || b.tag]);
        }
      }
      return {
        scrollWidth: document.documentElement.scrollWidth,
        viewportWidth: window.innerWidth,
        overlaps,
        outsideViewport,
      };
    });
    assert.ok(
      layout.scrollWidth <= layout.viewportWidth,
      `${viewport.width}px viewport has horizontal overflow: ${layout.scrollWidth}px`,
    );
    assert.deepEqual(layout.overlaps, [], `${viewport.width}px viewport has overlapping controls`);
    assert.deepEqual(
      layout.outsideViewport,
      [],
      `${viewport.width}px viewport has controls outside the viewport`,
    );
    return { viewport, ...layout };
  } finally {
    await context.close();
  }
}

async function main() {
  const initialReview = await waitForSession();
  const eventId = normalEventId(initialReview);
  const browser = await chromium.launch({ headless: true });
  try {
    const desktopContext = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const desktop = await desktopContext.newPage();
    await desktop.goto(`${baseUrl}/?qa=trainer-dataset`, {
      waitUntil: "domcontentloaded",
      timeout: 15000,
    });
    await desktop.waitForFunction(
      () => document.querySelectorAll("#event-list button").length > 0,
      null,
      { timeout: 10000 },
    );
    await selectEventAndWaitForMedia(desktop, eventId);
    const media = await mediaSummary(desktop);
    await verifyUnsavedIdentityKeepsLockDisabled(browser, eventId);
    await saveAndVerifyParticipants(desktop, eventId);
    const exports = await downloadAndVerifyExports(desktop, sessionDirArgument);
    const desktopLayout = await assertViewportLayout(browser, { width: 1440, height: 900 }, eventId);
    const mobileLayout = await assertViewportLayout(browser, { width: 390, height: 844 }, eventId);
    await desktopContext.close();
    process.stdout.write(`${JSON.stringify({
      status: "PASS",
      participants: { trainer_name: TRAINER_NAME, wrestler_name: WRESTLER_NAME },
      media,
      exports,
      viewports: [desktopLayout, mobileLayout].map(({
        viewport, scrollWidth, viewportWidth, outsideViewport,
      }) => ({
        viewport,
        scrollWidth,
        viewportWidth,
        horizontalOverflow: false,
        overlappingControls: 0,
        controlsOutsideViewport: outsideViewport.length,
      })),
    }, null, 2)}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
