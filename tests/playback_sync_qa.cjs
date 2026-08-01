const assert = require("node:assert/strict");
const { chromium } = require(
  process.env.PLAYWRIGHT_MODULE || "playwright",
);

const baseUrl = process.argv[2];
if (!baseUrl) throw new Error("Usage: node tests/playback_sync_qa.cjs BASE_URL");

async function mediaState(page) {
  return page.locator("video").evaluateAll((videos) => videos.map((video) => ({
    currentTime: video.currentTime,
    paused: video.paused,
    playbackRate: video.playbackRate,
    readyState: video.readyState,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
  })));
}

async function main() {
  const reviewResponse = await fetch(`${baseUrl}/api/session`);
  assert.equal(reviewResponse.ok, true);
  const review = await reviewResponse.json();
  assert.equal(review.session_ready, true);
  assert.ok(Math.abs(Number(review.sony_fps) - Number(review.iphone_fps)) < 0.05);
  const event = review.events.find((item) => item.event_id === "e-001");
  assert.ok(event);
  const slope = Number(review.time_map.slope);
  const intercept = Number(review.time_map.intercept);
  assert.ok(Number.isFinite(slope) && slope > 0 && Number.isFinite(intercept));
  const expectedIphoneLocal = (sonyLocal) => (
    ((Number(event.sony_start_s) + sonyLocal - intercept) / slope)
      - Number(event.iphone_start_s)
  );
  const mappedDrift = (sample) => Math.abs(
    sample[1].currentTime - expectedIphoneLocal(sample[0].currentTime)
  );

  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE }
      : {}),
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    await page.goto(`${baseUrl}/?v=playback-sync-qa`, {
      waitUntil: "domcontentloaded",
      timeout: 15000,
    });
    await page.waitForFunction(
      () => document.querySelectorAll("#event-list button").length === 3,
      null,
      { timeout: 10000 },
    );
    await page.locator("#event-list button[data-event-id='e-001']").click();
    await page.waitForFunction(
      () => [...document.querySelectorAll("video")].every((video) => (
        video.readyState >= 3 && video.videoWidth > 0 && video.videoHeight > 0
      )),
      null,
      { timeout: 20000 },
    );
    await page.evaluate(() => {
      window.__iphoneSeekCount = 0;
      document.querySelector("#iphone-video").addEventListener("seeking", () => {
        window.__iphoneSeekCount += 1;
      });
    });

    const play = page.locator("[data-action='toggle-play']");
    await play.click();
    await page.waitForFunction(
      () => [...document.querySelectorAll("video")].every((video) => !video.paused),
      null,
      { timeout: 5000 },
    );
    const samples = [];
    for (let index = 0; index < 14; index += 1) {
      await page.waitForTimeout(150);
      samples.push(await mediaState(page));
    }
    const warmed = samples.slice(3);
    const drifts = warmed.map(mappedDrift);
    assert.ok(Math.max(...drifts) < 0.035, `playback drift: ${drifts.join(",")}`);
    assert.equal(await page.evaluate(() => window.__iphoneSeekCount), 0);
    assert.ok(warmed.some((sample) => Math.abs(sample[1].playbackRate - 1) > 0.0001));

    await play.click();
    await page.waitForFunction(
      () => [...document.querySelectorAll("video")].every((video) => video.paused),
      null,
      { timeout: 5000 },
    );
    const paused = await mediaState(page);
    assert.ok(mappedDrift(paused) < 0.02);

    await page.locator("#master-seek").evaluate((slider) => {
      slider.value = "130";
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForTimeout(100);
    const scrubbed = await mediaState(page);
    assert.ok(Math.abs(scrubbed[0].currentTime - 1.5) < 0.02);
    assert.ok(Math.abs(scrubbed[1].currentTime - expectedIphoneLocal(1.5)) < 0.02);

    await play.click();
    await page.waitForFunction(
      () => [...document.querySelectorAll("video")].every((video) => !video.paused),
      null,
      { timeout: 5000 },
    );
    await page.evaluate(() => {
      const sony = document.querySelector("#sony-video");
      document.querySelector("#iphone-video").currentTime = Math.max(0, sony.currentTime - 0.2);
    });
    await page.waitForFunction(
      (mapping) => {
        const sony = document.querySelector("#sony-video");
        const iphone = document.querySelector("#iphone-video");
        const globalSony = mapping.sonyStart + sony.currentTime;
        const expectedIphone = ((globalSony - mapping.intercept) / mapping.slope)
          - mapping.iphoneStart;
        return Math.abs(expectedIphone - iphone.currentTime) < 0.04;
      },
      {
        sonyStart: Number(event.sony_start_s),
        iphoneStart: Number(event.iphone_start_s),
        slope,
        intercept,
      },
      { timeout: 3000 },
    );
    assert.ok(await page.evaluate(() => window.__iphoneSeekCount) >= 2);
    await play.click();

    await page.locator("#master-seek").evaluate((slider) => {
      slider.value = "131.8";
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await play.click();
    await page.waitForFunction(
      () => [...document.querySelectorAll("video")].every((video) => !video.paused),
      null,
      { timeout: 1000 },
    );
    await page.waitForFunction(
      () => [...document.querySelectorAll("video")].every((video) => (
        video.ended || video.paused
      )),
      null,
      { timeout: 3000 },
    );
    const ended = await mediaState(page);
    assert.ok(mappedDrift(ended) < 0.04);

    process.stdout.write(`${JSON.stringify({
      sonyFps: review.sony_fps,
      iphoneFps: review.iphone_fps,
      timeMap: review.time_map,
      maxWarmDriftMs: Math.round(Math.max(...drifts) * 1000),
      playbackSync: true,
      pauseSync: true,
      scrubSync: true,
      hardSync: true,
      naturalEndSync: true,
    }, null, 2)}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
