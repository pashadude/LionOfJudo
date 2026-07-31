(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const sony = $("#sony-video");
  const iphone = $("#iphone-video");
  const state = { review: null, selected: null, syncing: false };
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

  function sonyDuration() {
    return Number(state.review?.sony_duration_s || sony.duration || 1);
  }

  function iphoneTime(sonyTime) {
    const map = state.review?.time_map || { slope: 1, intercept: 0 };
    return (sonyTime - Number(map.intercept || 0)) / Number(map.slope || 1);
  }

  function seekSony(value) {
    const clamped = Math.max(0, Math.min(sonyDuration(), Number(value) || 0));
    state.syncing = true;
    sony.currentTime = clamped;
    const target = Math.max(0, iphoneTime(clamped));
    if (Number.isFinite(target)) iphone.currentTime = target;
    $("#master-seek").value = String(clamped);
    state.syncing = false;
    drawCharts();
    updateReadout();
  }

  function updateReadout() {
    const time = Number(sony.currentTime || 0);
    $("#time-readout").textContent = `${time.toFixed(2)} s`;
    $("#master-seek").max = String(Math.max(1, sonyDuration()));
    $("#master-seek").value = String(Math.min(sonyDuration(), time));
  }

  function mediaFor(event, camera) {
    if (!event) return "";
    const media = event.media || {};
    return media[camera] || `/media/events/${encodeURIComponent(event.event_id)}/${camera}.mp4`;
  }

  function selectEvent(event) {
    state.selected = event;
    document.querySelectorAll("#event-list button").forEach((button) => {
      button.setAttribute("aria-current", button.dataset.eventId === event.event_id ? "true" : "false");
    });
    sony.src = mediaFor(event, "sony");
    iphone.src = mediaFor(event, "iphone");
    sony.load();
    iphone.load();
    seekSony(Number(event.sony_start_s || 0));
    $("#suggested-technique").value = event.predlog_tehnike || "";
    $("#confirmed-technique").value = event.potvrdena_tehnika || "";
    $("#score").value = event.ocena == null ? "" : String(event.ocena);
    $("#note").value = event.napomena || "";
    const disabled = injury(event);
    $("#confirmed-technique").disabled = disabled;
    $("#score").disabled = disabled;
    $("#save-button").disabled = false;
    const visibility = $("#visibility-state");
    visibility.textContent = disabled ? "Prijavljen povredni događaj · Nedovoljno vidljivo" : "";
    visibility.classList.toggle("warning", disabled);
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

  function metricValue(point, key) {
    const value = point?.[key] ?? point?.metrics?.[key];
    return Number.isFinite(Number(value)) ? Number(value) : null;
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
        const x = Math.max(0, Math.min(width, (point.time / sonyDuration()) * width));
        const y = Math.max(0, Math.min(height, height - ((point.value - min) / (max - min)) * (height - 4) - 2));
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    const cursorX = Math.max(0, Math.min(width, (Number(sony.currentTime || 0) / sonyDuration()) * width));
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
    const isInjury = injury(event);
    const payload = {
      potvrdena_tehnika: isInjury ? "" : $("#confirmed-technique").value,
      ocena: isInjury ? null : Number($("#score").value),
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
    if (!state.selected) return;
    try { await saveAnnotation(state.selected); } catch (error) { status(error.message, true); }
  });
  $("#master-seek").addEventListener("input", (event) => seekSony(event.target.value));
  $("[data-action='toggle-play']").addEventListener("click", () => {
    if (sony.paused) { sony.play(); iphone.play(); } else { sony.pause(); iphone.pause(); }
  });
  $("[data-action='step-back']").addEventListener("click", () => seekSony(sony.currentTime - 1 / Number(state.review?.sony_fps || 30)));
  $("[data-action='step-forward']").addEventListener("click", () => seekSony(sony.currentTime + 1 / Number(state.review?.sony_fps || 30)));
  $("[data-action='restart']").addEventListener("click", () => seekSony(Number(state.selected?.sony_start_s || 0)));
  sony.addEventListener("timeupdate", () => { if (!state.syncing) { iphone.currentTime = Math.max(0, iphoneTime(sony.currentTime)); updateReadout(); drawCharts(); } });
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
    renderEvents();
    drawCharts();
  }).catch((error) => status(error.message, true));
})();
