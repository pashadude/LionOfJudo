(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const sony = $("#sony-video");
  const iphone = $("#iphone-video");
  const state = {
    review: null,
    selected: null,
    globalSonyTime: 0,
    syncing: false,
  };

  function status(message, error = false) {
    const node = $("#app-status");
    node.textContent = message || "";
    node.classList.toggle("error", error);
  }

  function injury(event) {
    return Boolean(event && (
      event.prijavljen_povredni_dogadjaj
      || event.iskljuceno_iz_statistike
      || event.status === "povreda"
    ));
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
      sonyLocalTime: clamp(
        review === state.review ? sonyLocalRaw : mappedSonyLocal,
        0,
        eventSpan(event, "sony_start_s", "sony_end_s"),
      ),
      iphoneLocalTime: clamp(
        review === state.review ? iphoneLocalRaw : mappedIphoneLocal,
        0,
        eventSpan(event, "iphone_start_s", "iphone_end_s"),
      ),
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
    if (!state.selected) return;
    const start = Number(state.selected.sony_start_s);
    const end = Number(state.selected.sony_end_s);
    syncVideosFromGlobal(clamp(Number(globalSonyTime), start, end));
    $("#master-seek").value = String(state.globalSonyTime);
    drawCharts();
    updateReadout();
    updateCorrectionControls();
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
    visibility.textContent = disabled
      ? "Prijavljen povredni događaj · samo za čitanje"
      : (event.vidljivost === "nedovoljno_vidljivo" ? "Nedovoljno vidljivo" : "");
    visibility.classList.toggle("warning", disabled || Boolean(visibility.textContent));
  }

  setEditorDisabled(true);

  function normalEvents() {
    return (state.review?.events || [])
      .filter((event) => !injury(event))
      .slice()
      .sort((first, second) => Number(first.sony_start_s) - Number(second.sony_start_s));
  }

  function nextNormalEvent(event) {
    const events = normalEvents();
    const index = events.findIndex((item) => item.event_id === event?.event_id);
    return index >= 0 && index + 1 < events.length ? events[index + 1] : null;
  }

  function updateCorrectionControls() {
    const event = state.selected;
    const readOnly = !event || injury(event);
    $("#event-start").disabled = readOnly;
    $("#event-end").disabled = readOnly;
    $("#update-bounds-button").disabled = readOnly;
    $("#create-event-button").disabled = !state.review || readOnly;
    $("#delete-button").disabled = readOnly;
    $("#merge-button").disabled = readOnly || !nextNormalEvent(event);
    const start = Number(event?.sony_start_s);
    const end = Number(event?.sony_end_s);
    const cursor = Number(state.globalSonyTime);
    $("#split-button").disabled = readOnly
      || !Number.isFinite(cursor)
      || cursor <= start + 0.001
      || cursor >= end - 0.001;
  }

  function populateEventBounds(event) {
    $("#event-start").value = Number(event.sony_start_s).toFixed(3);
    $("#event-end").value = Number(event.sony_end_s).toFixed(3);
  }

  function updateSyncLock(review) {
    const locked = Boolean(review && review.sync_locked);
    const button = $("#sync-button");
    const note = $("#sync-lock-note");
    button.disabled = locked;
    button.title = locked
      ? "Sinhronizacija je zaključana; za promenu je potreban novi uvoz"
      : "Potvrdi sinhronizaciju";
    note.hidden = !locked;
  }

  function selectEvent(event) {
    state.selected = event;
    state.globalSonyTime = Number(event.sony_start_s) || 0;
    document.querySelectorAll("#event-list button").forEach((button) => {
      button.setAttribute(
        "aria-current",
        button.dataset.eventId === event.event_id ? "true" : "false",
      );
    });
    sony.src = mediaFor(event, "sony");
    iphone.src = mediaFor(event, "iphone");
    sony.load();
    iphone.load();
    updateEditor(event);
    populateEventBounds(event);
    seekSony(state.globalSonyTime);
    updateReadout();
    updateCorrectionControls();
    drawCharts();
  }

  function renderEvents(preferredEventId = null) {
    const list = $("#event-list");
    list.replaceChildren();
    const events = Array.isArray(state.review?.events) ? state.review.events : [];
    $("#event-count").textContent = String(events.length);
    events.forEach((event) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.eventId = event.event_id;
      if (injury(event)) button.classList.add("injury");
      button.innerHTML = `<span class="event-name"></span><span class="event-meta"></span>${injury(event) ? '<span class="injury-label">Prijavljen povredni događaj</span>' : ""}`;
      button.querySelector(".event-name").textContent = event.potvrdena_tehnika
        || event.predlog_tehnike
        || event.event_id;
      button.querySelector(".event-meta").textContent = `${Number(event.sony_start_s).toFixed(2)}–${Number(event.sony_end_s).toFixed(2)} s`;
      button.addEventListener("click", () => selectEvent(event));
      item.append(button);
      list.append(item);
    });
    const selected = events.find((event) => event.event_id === preferredEventId) || events[0];
    if (selected) {
      selectEvent(selected);
    } else {
      state.selected = null;
      setEditorDisabled(true);
      updateCorrectionControls();
      drawCharts();
    }
  }

  function sonyFps() {
    const fps = Number(state.review?.sony_fps);
    return Number.isFinite(fps) && fps > 0 ? fps : null;
  }

  function updateFrameControls() {
    const disabled = sonyFps() === null;
    [$("#step-back"), $("#step-forward")].forEach((button) => {
      button.disabled = disabled;
      button.title = disabled
        ? "FPS Sony nije dostupan; kadriranje je onemogućeno"
        : button.getAttribute("aria-label");
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
    const value = point?.[key];
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }

  function selectedFrameSamples(key) {
    const event = state.selected;
    if (!event) return [];
    const start = Number(event.sony_start_s);
    const end = Number(event.sony_end_s);
    return (state.review?.frame_metrics || [])
      .map((point) => ({
        time: Number(point.timestamp_s),
        value: metricValue(point, key),
      }))
      .filter((point) => (
        Number.isFinite(point.time)
        && point.time >= start
        && point.time <= end
        && point.value !== null
      ));
  }

  function drawChart(canvas) {
    const key = canvas.dataset.metric;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#d5dce1";
    ctx.lineWidth = 1;
    for (const fraction of [0.25, 0.5, 0.75]) {
      const y = Math.round(height * fraction) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    const event = state.selected;
    const samples = selectedFrameSamples(key);
    canvas.dataset.finiteSamples = String(samples.length);
    if (!event) return;
    const start = Number(event.sony_start_s);
    const end = Number(event.sony_end_s);
    const domain = Math.max(end - start, 1e-9);
    if (samples.length) {
      let min = Math.min(...samples.map((point) => point.value));
      let max = Math.max(...samples.map((point) => point.value));
      if (min === max) {
        min -= 0.5;
        max += 0.5;
      }
      ctx.strokeStyle = "#116466";
      ctx.fillStyle = "#116466";
      ctx.lineWidth = 2;
      ctx.beginPath();
      samples.forEach((point, index) => {
        const x = clamp(((point.time - start) / domain) * width, 1, width - 1);
        const y = clamp(
          height - ((point.value - min) / (max - min)) * (height - 12) - 6,
          2,
          height - 2,
        );
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      if (samples.length > 1) {
        ctx.stroke();
      } else {
        const point = samples[0];
        const x = clamp(((point.time - start) / domain) * width, 2, width - 2);
        ctx.arc(x, height / 2, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    const cursorX = clamp(((state.globalSonyTime - start) / domain) * width, 0, width);
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

  async function readResult(response, fallback) {
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || fallback);
    return result;
  }

  async function mutateReview(url, method, payload = undefined) {
    const options = { method, headers: {} };
    if (payload !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(payload);
    }
    const response = await fetch(url, options);
    return readResult(response, "Izmena događaja nije uspela");
  }

  function applyMutation(result, message) {
    state.review = result.review;
    updateSyncLock(state.review);
    updateFrameControls();
    renderEvents(result.selected_event_id);
    status(message);
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
    const result = await readResult(response, "Čuvanje nije uspelo");
    Object.assign(event, result);
    status("Sačuvano");
    renderEvents(event.event_id);
  }

  $("#annotation-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.selected || injury(state.selected)) return;
    try {
      await saveAnnotation(state.selected);
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#master-seek").addEventListener("input", (event) => seekSony(event.target.value));
  $("[data-action='toggle-play']").addEventListener("click", () => {
    if (sony.paused) sony.play().catch(() => {}); else sony.pause();
  });
  $("[data-action='step-back']").addEventListener("click", () => stepFrame(-1));
  $("[data-action='step-forward']").addEventListener("click", () => stepFrame(1));
  $("[data-action='restart']").addEventListener("click", () => {
    seekSony(Number(state.selected?.sony_start_s || 0));
  });

  sony.addEventListener("timeupdate", () => {
    if (!state.syncing && state.selected) {
      state.globalSonyTime = clamp(
        globalSonyTimeForLocal(sony.currentTime, state.selected),
        Number(state.selected.sony_start_s),
        Number(state.selected.sony_end_s),
      );
      const times = localTimesForGlobal(state.globalSonyTime, state.selected);
      state.syncing = true;
      iphone.currentTime = times.iphoneLocalTime;
      state.syncing = false;
      updateReadout();
      updateCorrectionControls();
      drawCharts();
    }
  });
  sony.addEventListener("play", () => { if (iphone.paused) iphone.play().catch(() => {}); });
  sony.addEventListener("pause", () => { if (!iphone.paused) iphone.pause(); });

  $("#sync-button").addEventListener("click", async () => {
    if (state.review?.sync_locked) return;
    try {
      const response = await fetch("/api/session/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          anchors: state.review.anchors,
          injury_cutoff_s: state.review.injury_cutoff_s,
        }),
      });
      state.review = await readResult(response, "Sinhronizacija nije uspela");
      updateSyncLock(state.review);
      status("Sinhronizacija je potvrđena");
      drawCharts();
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#create-event-button").addEventListener("click", async () => {
    try {
      const result = await mutateReview("/api/events", "POST", {
        sony_start_s: Number($("#event-start").value),
        sony_end_s: Number($("#event-end").value),
      });
      applyMutation(result, "Događaj je napravljen");
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#update-bounds-button").addEventListener("click", async () => {
    const event = state.selected;
    if (!event || injury(event)) return;
    try {
      const result = await mutateReview(
        `/api/events/${encodeURIComponent(event.event_id)}/bounds`,
        "PUT",
        {
          sony_start_s: Number($("#event-start").value),
          sony_end_s: Number($("#event-end").value),
        },
      );
      applyMutation(result, "Granice događaja su sačuvane");
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#split-button").addEventListener("click", async () => {
    const event = state.selected;
    if (!event || injury(event)) return;
    try {
      const result = await mutateReview(
        `/api/events/${encodeURIComponent(event.event_id)}/split`,
        "POST",
        { sony_split_s: Number(state.globalSonyTime) },
      );
      applyMutation(result, "Događaj je podeljen");
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#merge-button").addEventListener("click", async () => {
    const event = state.selected;
    const next = nextNormalEvent(event);
    if (!event || !next || injury(event)) return;
    try {
      const result = await fetch("/api/events/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_ids: [event.event_id, next.event_id] }),
      }).then((response) => readResult(response, "Spajanje nije uspelo"));
      applyMutation(result, "Događaji su spojeni");
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#delete-button").addEventListener("click", async () => {
    const event = state.selected;
    if (!event || injury(event)) return;
    if (!window.confirm(`Obrisati događaj ${event.event_id}?`)) return;
    try {
      const result = await mutateReview(
        `/api/events/${encodeURIComponent(event.event_id)}`,
        "DELETE",
      );
      applyMutation(result, "Događaj je obrisan");
    } catch (error) {
      status(error.message, true);
    }
  });

  document.querySelectorAll("canvas[data-metric]").forEach((canvas) => {
    canvas.addEventListener("click", (event) => {
      if (!state.selected) return;
      const bounds = canvas.getBoundingClientRect();
      const fraction = clamp(
        (event.clientX - bounds.left) / Math.max(1, bounds.width),
        0,
        1,
      );
      const start = Number(state.selected.sony_start_s);
      const end = Number(state.selected.sony_end_s);
      seekSony(start + fraction * (end - start));
    });
  });
  window.addEventListener("resize", drawCharts);

  fetch("/api/session")
    .then((response) => readResult(response, "Sesija nije dostupna"))
    .then((review) => {
      state.review = review;
      $("#session-title").textContent = review.session_id || "Video pregled";
      updateSyncLock(review);
      updateFrameControls();
      renderEvents();
      drawCharts();
    })
    .catch((error) => status(error.message, true));
})();
