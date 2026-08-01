(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const sony = $("#sony-video");
  const iphone = $("#iphone-video");
  const state = { review: null, selected: null, globalSonyTime: 0, syncing: false };
  const metricLabels = {
    brzina_ulaska_norm: "Brzina ulaska",
    rotacija_trupa_2d_dps: "Rotacija trupa (2D)",
    promena_visine_kukova_norm: "Visina kukova",
    vreme_oporavka_s: "Stabilnost",
    intenzitet_pokreta_0_100: "Intenzitet pokreta",
  };

  function status(message, error = false) {
    const node = $("#app-status");
    node.textContent = message || "";
    node.classList.toggle("error", error);
  }

  function injury(event) {
    return Boolean(event && (event.prijavljen_povredni_dogadjaj || event.iskljuceno_iz_statistike || event.status === "povreda"));
  }

  function clamp(value, lower, upper) {
    return Math.max(lower, Math.min(upper, value));
  }

  function sonyDuration() {
    const persisted = Number(state.review?.sony_duration_s);
    if (Number.isFinite(persisted) && persisted > 0) return persisted;
    const loaded = Number(sony.duration);
    return Number.isFinite(loaded) && loaded > 0 ? loaded : 0;
  }

  function inverseIphoneTime(sonyTime, review = state.review) {
    const map = review?.time_map || {};
    const slope = Number(map.slope);
    const intercept = Number(map.intercept);
    if (!Number.isFinite(slope) || slope <= 0 || !Number.isFinite(intercept)) return 0;
    return (sonyTime - intercept) / slope;
  }

  function iphoneTime(sonyTime) {
    return inverseIphoneTime(sonyTime, state.review);
  }

  function eventSpan(event, startKey, endKey) {
    const span = Number(event?.[endKey]) - Number(event?.[startKey]);
    return Number.isFinite(span) && span > 0 ? span : 0;
  }

  function localTimesForGlobal(globalSonyTime, event, review = state.review) {
    const duration = sonyDuration();
    const boundedGlobal = clamp(Number(globalSonyTime) || 0, 0, duration);
    const sonyStart = Number(event?.sony_start_s) || 0;
    const iphoneStart = Number(event?.iphone_start_s) || 0;
    const sonyLocalRaw = globalSonyTime - Number(event.sony_start_s || 0);
    const iphoneLocalRaw = iphoneTime(globalSonyTime) - Number(event.iphone_start_s || 0);
    const mappedIphoneTime = inverseIphoneTime(boundedGlobal, review);
    const mappedSonyLocal = boundedGlobal - sonyStart;
    const mappedIphoneLocal = mappedIphoneTime - iphoneStart;
    return {
      globalSonyTime: boundedGlobal,
      sonyLocalTime: clamp(review === state.review ? sonyLocalRaw : mappedSonyLocal, 0, eventSpan(event, "sony_start_s", "sony_end_s")),
      iphoneLocalTime: clamp(review === state.review ? iphoneLocalRaw : mappedIphoneLocal, 0, eventSpan(event, "iphone_start_s", "iphone_end_s")),
    };
  }

  function globalSonyTimeForLocal(localSonyTime, event) {
    return (Number(localSonyTime) || 0) + (Number(event?.sony_start_s) || 0);
  }

  function syncVideosFromGlobal(globalSonyTime) {
    const event = state.selected;
    if (!event) return;
    const times = localTimesForGlobal(globalSonyTime, event);
    state.globalSonyTime = times.globalSonyTime;
    state.syncing = true;
    sony.currentTime = times.sonyLocalTime;
    iphone.currentTime = times.iphoneLocalTime;
    state.syncing = false;
  }

  function seekSony(globalSonyTime) {
    syncVideosFromGlobal(globalSonyTime);
    $("#master-seek").value = String(state.globalSonyTime);
    drawCharts();
    updateReadout();
  }

  function updateReadout() {
    const time = Number(state.globalSonyTime || 0);
    $("#time-readout").textContent = `${time.toFixed(2)} s`;
    $("#master-seek").max = String(Math.max(1, sonyDuration()));
    $("#master-seek").value = String(clamp(time, 0, sonyDuration()));
  }

  function mediaFor(event, camera) {
    if (!event) return "";
    const media = event.media || {};
    return media[camera] || `/media/events/${encodeURIComponent(event.event_id)}/${camera}.mp4`;
  }

  function setEditorDisabled(disabled) {
    $("#confirmed-technique").disabled = disabled;
    $("#score").disabled = disabled;
    $("#note").disabled = disabled;
    $("#save-button").disabled = disabled;
  }

  function updateEditor(event) {
    const disabled = injury(event);
    $("#suggested-technique").value = event.predlog_tehnike || "";
    $("#confirmed-technique").value = event.potvrdena_tehnika || "";
    $("#score").value = event.ocena == null ? "" : String(event.ocena);
    $("#note").value = event.napomena || "";
    setEditorDisabled(disabled);
    const visibility = $("#visibility-state");
    visibility.textContent = disabled ? "Prijavljen povredni događaj · Nedovoljno vidljivo" : "";
    visibility.classList.toggle("warning", disabled);
  }

  setEditorDisabled(true);

  function selectEvent(event) {
    state.selected = event;
    state.globalSonyTime = Number(event.sony_start_s) || 0;
    document.querySelectorAll("#event-list button").forEach((button) => {
      button.setAttribute("aria-current", button.dataset.eventId === event.event_id ? "true" : "false");
    });
    sony.src = mediaFor(event, "sony");
    iphone.src = mediaFor(event, "iphone");
    sony.load();
    iphone.load();
    updateEditor(event);
    seekSony(state.globalSonyTime);
    updateReadout();
    drawCharts();
  }

  function renderEvents() {
    const list = $("#event-list");
    list.replaceChildren();
    const events = Array.isArray(state.review.events) ? state.review.events : [];
    $("#event-count").textContent = String(events.length);
    events.forEach((event) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.eventId = event.event_id;
      if (injury(event)) button.classList.add("injury");
      button.innerHTML = `<span class="event-name"></span><span class="event-meta"></span>${injury(event) ? '<span class="injury-label">Prijavljen povredni događaj</span>' : ""}`;
      button.querySelector(".event-name").textContent = event.predlog_tehnike || event.event_id;
      button.querySelector(".event-meta").textContent = `${Number(event.sony_start_s || 0).toFixed(2)}–${Number(event.sony_end_s || 0).toFixed(2)} s`;
      button.addEventListener("click", () => selectEvent(event));
      item.append(button);
      list.append(item);
    });
    if (events.length) selectEvent(events[0]);
  }

  function sonyFps() {
    const fps = Number(state.review?.sony_fps);
    return Number.isFinite(fps) && fps > 0 ? fps : null;
  }

  function updateFrameControls() {
    const disabled = sonyFps() === null;
    [$("#step-back"), $("#step-forward")].forEach((button) => {
      button.disabled = disabled;
      button.title = disabled ? "FPS Sony nije dostupan; kadriranje je onemogućeno" : button.title;
    });
  }

  function stepFrame(direction) {
    const fps = sonyFps();
    if (fps === null) {
      status("FPS Sony nije dostupan; kadriranje je onemogućeno", true);
      updateFrameControls();
      return;
    }
    seekSony(state.globalSonyTime + direction / fps);
  }

  function metricValue(point, key) {
    const value = point?.[key] ?? point?.metrics?.[key];
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }

  function drawChart(canvas) {
    const key = canvas.dataset.metric;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const domain = Math.max(sonyDuration(), 1e-9);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#d5dce1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 1);
    ctx.lineTo(width, height - 1);
    ctx.stroke();
    const frameSamples = state.review?.frame_metrics || [];
    const fallbackSamples = (state.review?.events || []).map((event) => ({
      timestamp_s: event.sony_start_s,
      ...event,
    }));
    const samples = (frameSamples.length ? frameSamples : fallbackSamples).map((point) => ({
      time: Number(point.timestamp_s || 0),
      value: metricValue(point, key),
    })).filter((point) => point.value !== null);
    if (samples.length > 1) {
      let min = Math.min(...samples.map((point) => point.value));
      let max = Math.max(...samples.map((point) => point.value));
      if (min === max) { min -= 1; max += 1; }
      ctx.strokeStyle = "#116466";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      samples.forEach((point, index) => {
        const x = clamp((point.time / domain) * width, 0, width);
        const y = clamp(height - ((point.value - min) / (max - min)) * (height - 4) - 2, 0, height);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    const cursorX = clamp((Number(state.globalSonyTime || 0) / domain) * width, 0, width);
    ctx.strokeStyle = "#9b2c2c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, height);
    ctx.stroke();
  }

  function drawCharts() {
    document.querySelectorAll("canvas[data-metric]").forEach(drawChart);
  }

  async function saveAnnotation(event) {
    if (injury(event)) {
      status("Prijavljen povredni događaj je samo za čitanje", true);
      return;
    }
    const payload = {
      potvrdena_tehnika: $("#confirmed-technique").value,
      ocena: Number($("#score").value),
      napomena: $("#note").value,
    };
    const response = await fetch(`/api/events/${encodeURIComponent(event.event_id)}/annotation`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Čuvanje nije uspelo");
    Object.assign(event, result);
    status("Sačuvano");
  }

  $("#annotation-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.selected || injury(state.selected)) return;
    try { await saveAnnotation(state.selected); } catch (error) { status(error.message, true); }
  });
  $("#master-seek").addEventListener("input", (event) => seekSony(event.target.value));
  $("[data-action='toggle-play']").addEventListener("click", () => {
    if (sony.paused) sony.play().catch(() => {}); else sony.pause();
  });
  $("[data-action='step-back']").addEventListener("click", () => stepFrame(-1));
  $("[data-action='step-forward']").addEventListener("click", () => stepFrame(1));
  $("[data-action='restart']").addEventListener("click", () => seekSony(Number(state.selected?.sony_start_s || 0)));
  sony.addEventListener("timeupdate", () => {
    if (!state.syncing && state.selected) {
      state.globalSonyTime = clamp(globalSonyTimeForLocal(sony.currentTime, state.selected), 0, sonyDuration());
      const times = localTimesForGlobal(state.globalSonyTime, state.selected);
      state.syncing = true;
      iphone.currentTime = times.iphoneLocalTime;
      state.syncing = false;
      updateReadout();
      drawCharts();
    }
  });
  sony.addEventListener("play", () => { if (iphone.paused) iphone.play().catch(() => {}); });
  sony.addEventListener("pause", () => { if (!iphone.paused) iphone.pause(); });
  $("#share-button").addEventListener("click", async () => {
    try {
      if (navigator.share) await navigator.share({ title: "Video pregled", url: location.href });
      else await navigator.clipboard.writeText(location.href);
      status("Veza je spremna za deljenje");
    } catch (error) { if (error.name !== "AbortError") status("Deljenje nije uspelo", true); }
  });
  $("#sync-button").addEventListener("click", async () => {
    try {
      const response = await fetch("/api/session/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ anchors: state.review.anchors, injury_cutoff_s: state.review.injury_cutoff_s }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Sinhronizacija nije uspela");
      state.review = result;
      updateFrameControls();
      status("Sinhronizacija je potvrđena");
      drawCharts();
    } catch (error) { status(error.message, true); }
  });
  document.querySelectorAll("canvas[data-metric]").forEach((canvas) => canvas.addEventListener("click", (event) => {
    const bounds = canvas.getBoundingClientRect();
    seekSony(((event.clientX - bounds.left) / Math.max(1, bounds.width)) * sonyDuration());
  }));
  window.addEventListener("resize", drawCharts);

  fetch("/api/session").then((response) => {
    if (!response.ok) throw new Error("Sesija nije dostupna");
    return response.json();
  }).then((review) => {
    state.review = review;
    $("#session-title").textContent = review.session_id || "Video pregled";
    updateFrameControls();
    renderEvents();
    drawCharts();
  }).catch((error) => status(error.message, true));
})();
