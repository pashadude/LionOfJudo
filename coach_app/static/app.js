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
    trainerCitations: [],
  };
  const MIN_EVENT_SPAN = 0.001;
  const DRAFT_EVENT_SPAN = 1;
  const HARD_SYNC_THRESHOLD_S = 0.12;
  const MAX_PLAYBACK_RATE_CORRECTION = 0.04;
  const PLAYBACK_RATE_GAIN = 0.8;

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
    iphone.playbackRate = 1;
    state.syncing = false;
  }

  function correctIphonePlayback(times, hardSync = false) {
    const target = Number(times?.iphoneLocalTime);
    const current = Number(iphone.currentTime);
    if (!Number.isFinite(target) || !Number.isFinite(current)) return;
    const drift = target - current;
    if (hardSync || iphone.paused || Math.abs(drift) >= HARD_SYNC_THRESHOLD_S) {
      if (Math.abs(drift) > 0.001) iphone.currentTime = target;
      iphone.playbackRate = 1;
      return;
    }
    iphone.playbackRate = clamp(
      1 + drift * PLAYBACK_RATE_GAIN,
      1 - MAX_PLAYBACK_RATE_CORRECTION,
      1 + MAX_PLAYBACK_RATE_CORRECTION,
    );
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
    const path = media[camera]
      || `/media/events/${encodeURIComponent(event.event_id)}/${camera}.mp4`;
    const mediaVersion = event.analysis_fingerprint || event.event_revision
      || state.review?.session_id || "1";
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}v=${encodeURIComponent(String(mediaVersion))}`;
  }

  function setChecked(name, value) {
    document.querySelectorAll(`input[name='${name}']`).forEach((input) => {
      input.checked = input.value === String(value ?? "");
    });
  }

  function checkedValue(name) {
    return document.querySelector(`input[name='${name}']:checked`)?.value || "";
  }

  function activeTrainerAssessment(event) {
    const revision = event?.aktivna_trener_revizija;
    return (event?.trener_procene || []).find((row) => row.revizija === revision) || null;
  }

  function hasPreAiAssessment(event) {
    return (event?.trener_procene || []).some((row) => (
      row.faza === "pre_ai"
      && row.event_revision === event.event_revision
      && row.analysis_fingerprint === event.analysis_fingerprint
    ));
  }

  function revealedAiEvaluation(event) {
    return (event?.ai_procene || []).find((row) => row.ai_otkriven_u) || null;
  }

  function activeAiFeedback(event) {
    const duel = event?.aktivni_duel;
    if (!duel) return null;
    return (event?.procene_ai_predloga || []).find((row) => (
      row.event_revision === duel.event_revision
      && row.analysis_fingerprint === duel.analysis_fingerprint
      && row.trener_revizija === duel.trener_revizija
      && row.evaluator_id === duel.evaluator_id
    )) || null;
  }

  function displayNumber(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "—";
  }

  function optionalScore(value) {
    if (value == null || value === "") return null;
    const score = Number(value);
    return Number.isFinite(score) ? score : null;
  }

  function updateCitationList() {
    const node = $("#citation-list");
    node.textContent = state.trainerCitations.length
      ? state.trainerCitations.map((second) => `${Number(second).toFixed(3)} s`).join(" · ")
      : "Nema citiranih trenutaka";
  }

  function setEditorDisabled(disabled) {
    $("#confirmed-technique").disabled = disabled;
    $("#trainer-reason").disabled = disabled;
    $("#add-current-second").disabled = disabled;
    $("#lock-assessment-button").disabled = disabled;
    document.querySelectorAll("input[name='visibility'], input[name='trainer-score']")
      .forEach((input) => { input.disabled = disabled; });
  }

  function renderEvidence(ai, feedback) {
    const list = $("#evidence-list");
    const controls = $("#evidence-feedback");
    list.replaceChildren();
    controls.replaceChildren();
    const savedRatings = new Map(
      (feedback?.procene_dokaza || []).map((row) => [row.metrika, row.odnos]),
    );
    (ai?.dokazi || []).forEach((row) => {
      const item = document.createElement("li");
      const metric = document.createElement("strong");
      const value = document.createElement("span");
      const time = document.createElement("span");
      metric.textContent = row.metrika;
      value.textContent = `${displayNumber(row.vrednost, 3)} ${row.jedinica}`;
      time.textContent = `${displayNumber(row.sony_s, 3)} s`;
      item.append(metric, value, time);
      list.append(item);

      const label = document.createElement("label");
      label.textContent = row.metrika;
      const select = document.createElement("select");
      select.dataset.evidenceMetric = row.metrika;
      [
        ["", "Bez ocene"],
        ["prihvatam", "Prihvatam"],
        ["nepotpun", "Nepotpun dokaz"],
        ["osporavam", "Osporavam"],
      ].forEach(([valueKey, text]) => {
        const option = document.createElement("option");
        option.value = valueKey;
        option.textContent = text;
        select.append(option);
      });
      select.value = savedRatings.get(row.metrika) || "";
      label.append(select);
      controls.append(label);
    });
  }

  function renderImu(imu) {
    const suffixes = {
      ugaona_brzina_trupa_dps: " °/s",
      proxy_ubrzanja_0_100: " / 100",
      proxy_impulsa_0_100: " / 100",
      intenzitet_0_100: " / 100",
      vrh_sony_s: " s",
    };
    document.querySelectorAll("#imu-panel .imu-value").forEach((node) => {
      const key = node.dataset.imu;
      const value = imu?.[key];
      node.textContent = typeof value === "number"
        ? `${displayNumber(value, key === "vrh_sony_s" ? 3 : 1)}${suffixes[key] || ""}`
        : (value || "—");
    });
  }

  function renderAiPanels(event, trainer) {
    const ai = revealedAiEvaluation(event);
    const revealed = Boolean(ai);
    $("#ai-duel").hidden = !revealed;
    $("#system-facts").hidden = !revealed;
    $("#imu-panel").hidden = !revealed;
    $("#ai-feedback").hidden = !revealed;
    if (!revealed) return;

    const trainerScore = optionalScore(trainer?.ocena);
    const aiScore = optionalScore(ai.predlozena_ocena);
    $("#trainer-duel-score").textContent = Number.isFinite(trainerScore) ? `${trainerScore} / 5` : "Bez ocene";
    $("#trainer-duel-technique").textContent = trainer?.potvrdena_tehnika || "Nedovoljno vidljivo";
    $("#trainer-duel-reason").textContent = trainer?.razlog || "Trener nije dao procenu.";
    $("#ai-duel-score").textContent = Number.isFinite(aiScore) ? `${aiScore} / 5` : "Bez ocene";
    $("#ai-duel-confidence").textContent = `Pouzdanost ${displayNumber(Number(ai.pouzdanost_0_1) * 100, 1)}%`;
    $("#ai-duel-reason").textContent = ai.razlog || "Sistem nema dovoljno podataka.";
    if (!Number.isFinite(aiScore)) {
      $("#duel-delta").textContent = "AI nema dovoljno podataka.";
    } else if (Number.isFinite(trainerScore)) {
      const delta = Math.abs(aiScore - trainerScore);
      $("#duel-delta").textContent = `AI odstupa za ${delta} poena. Odbrani procenu.`;
    } else {
      $("#duel-delta").textContent = "AI je dao ocenu. Trener nije ocenio vidljivi kvalitet.";
    }

    const feedback = activeAiFeedback(event);
    renderEvidence(ai, feedback);
    renderImu(event.imu_eksperimentalno);
    setChecked("ai-relation", feedback?.odnos);
    $("#feedback-reason").value = feedback?.razlog || "";
    const feedbackLocked = Boolean(feedback);
    document.querySelectorAll("input[name='ai-relation'], #ai-feedback textarea, #ai-feedback select")
      .forEach((control) => { control.disabled = feedbackLocked; });
    $("#save-feedback-button").disabled = feedbackLocked;
    $("#save-feedback-button").textContent = feedbackLocked
      ? "Odgovor je sačuvan"
      : "Sačuvaj odgovor";
  }

  function updateEditor(event) {
    const disabled = injury(event);
    const trainer = activeTrainerAssessment(event);
    const preAiLocked = hasPreAiAssessment(event);
    const revealed = Boolean(revealedAiEvaluation(event));
    $("#confirmed-technique").value = trainer?.potvrdena_tehnika || event.potvrdena_tehnika || "";
    $("#trainer-reason").value = trainer?.razlog || "";
    setChecked("visibility", trainer?.status_vidljivosti);
    setChecked("trainer-score", trainer?.ocena);
    state.trainerCitations = Array.isArray(trainer?.citirani_sony_trenuci_s)
      ? trainer.citirani_sony_trenuci_s.slice()
      : [];
    updateCitationList();
    setEditorDisabled(disabled);
    if (preAiLocked && !revealed) setEditorDisabled(true);
    $("#lock-assessment-button").textContent = revealed
      ? "Sačuvaj korekciju"
      : "Zaključaj procenu";
    $("#reveal-ai-button").hidden = disabled || !preAiLocked || revealed;
    $("#reveal-ai-button").disabled = disabled || !preAiLocked || revealed;
    renderAiPanels(event, trainer);
    const visibility = $("#visibility-state");
    visibility.textContent = disabled
      ? "Prijavljen povredni događaj · samo za čitanje"
      : revealed
        ? "AI otkriven · korekcije trenera ostaju verzionisane"
        : preAiLocked
          ? "Procena zaključana · AI još nije prikazan"
          : "AI je skriven dok trener ne zaključa procenu";
    visibility.classList.toggle("warning", disabled || Boolean(visibility.textContent));
  }

  setEditorDisabled(true);

  function normalEvents() {
    return (state.review?.events || [])
      .filter((event) => !injury(event))
      .slice()
      .sort((first, second) => Number(first.sony_start_s) - Number(second.sony_start_s));
  }

  function firstConfirmedSonyAnchor() {
    const confirmedSonyTimes = (state.review?.anchors || [])
      .filter((item) => item?.user_confirmed === true)
      .map((item) => Number(item?.sony_s))
      .filter((sonyTime) => Number.isFinite(sonyTime));
    return confirmedSonyTimes.length ? Math.min(...confirmedSonyTimes) : null;
  }

  function normalEventDraftBounds() {
    const firstAnchor = firstConfirmedSonyAnchor();
    const cutoff = Number(state.review?.injury_cutoff_s);
    if (
      firstAnchor === null
      || !Number.isFinite(cutoff)
      || cutoff - firstAnchor <= MIN_EVENT_SPAN
    ) return null;

    const gaps = [];
    let gapStart = firstAnchor;
    for (const event of normalEvents()) {
      const start = Number(event.sony_start_s);
      const end = Number(event.sony_end_s);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
      const boundedStart = clamp(start, firstAnchor, cutoff);
      const boundedEnd = clamp(end, firstAnchor, cutoff);
      if (boundedStart - gapStart > MIN_EVENT_SPAN) {
        gaps.push({ start: gapStart, end: boundedStart });
      }
      gapStart = Math.max(gapStart, boundedEnd);
    }
    if (cutoff - gapStart > MIN_EVENT_SPAN) gaps.push({ start: gapStart, end: cutoff });
    if (!gaps.length) return null;

    const rawCursor = Number(state.globalSonyTime);
    const cursor = Number.isFinite(rawCursor) ? rawCursor : cutoff;
    const preferredEnd = Math.min(cutoff, cursor);
    const boundedCursor = Math.max(firstAnchor, preferredEnd);
    const gap = gaps.reduce((closest, candidate) => {
      const distance = boundedCursor < candidate.start
        ? candidate.start - boundedCursor
        : boundedCursor > candidate.end
          ? boundedCursor - candidate.end
          : 0;
      return distance < closest.distance ? { candidate, distance } : closest;
    }, { candidate: gaps[0], distance: Number.POSITIVE_INFINITY }).candidate;
    const end = Math.min(gap.end, Math.max(gap.start + DRAFT_EVENT_SPAN, boundedCursor));
    const start = Math.max(gap.start, end - DRAFT_EVENT_SPAN);
    return end - start > MIN_EVENT_SPAN ? { start, end } : null;
  }

  function nextNormalEvent(event) {
    const events = normalEvents();
    const index = events.findIndex((item) => item.event_id === event?.event_id);
    return index >= 0 && index + 1 < events.length ? events[index + 1] : null;
  }

  function updateCorrectionControls() {
    const event = state.selected;
    const canEditSelectedNormal = Boolean(event && !injury(event));
    const readOnly = !canEditSelectedNormal;
    const canCreateEvent = Boolean(normalEventDraftBounds());
    $("#event-start").disabled = !(canEditSelectedNormal || canCreateEvent);
    $("#event-end").disabled = !(canEditSelectedNormal || canCreateEvent);
    $("#update-bounds-button").disabled = !canEditSelectedNormal;
    $("#create-event-button").disabled = !canCreateEvent;
    $("#delete-button").disabled = readOnly;
    $("#merge-button").disabled = readOnly || !nextNormalEvent(event);
    const start = Number(event?.sony_start_s);
    const end = Number(event?.sony_end_s);
    const cursor = Number(state.globalSonyTime);
    $("#split-button").disabled = readOnly
      || !Number.isFinite(cursor)
      || cursor <= start + 0.001
      || cursor >= end - 0.001;
    if (!canCreateEvent && event && injury(event)) {
      status("Nema dostupnog normalnog intervala između prvog ankera i preseka povrede.", true);
    }
  }

  function populateEventBounds(event) {
    $("#event-start").value = Number(event.sony_start_s).toFixed(3);
    $("#event-end").value = Number(event.sony_end_s).toFixed(3);
  }

  function populateDraftEventBounds() {
    const draft = normalEventDraftBounds();
    if (!draft) {
      $("#event-start").value = "";
      $("#event-end").value = "";
      return null;
    }
    $("#event-start").value = draft.start.toFixed(3);
    $("#event-end").value = draft.end.toFixed(3);
    return draft;
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
    if (injury(event)) populateDraftEventBounds();
    else populateEventBounds(event);
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

  function replaceSelectedEvent(event) {
    const index = (state.review?.events || []).findIndex((row) => row.event_id === event.event_id);
    if (index >= 0) state.review.events[index] = event;
    state.selected = event;
    updateEditor(event);
    const button = document.querySelector(`#event-list button[data-event-id='${CSS.escape(event.event_id)}']`);
    if (button) {
      button.querySelector(".event-name").textContent = event.potvrdena_tehnika
        || event.predlog_tehnike
        || event.event_id;
    }
  }

  function trainerAssessmentPayload() {
    const visibility = checkedValue("visibility");
    if (!visibility) throw new Error("Izaberite vidljivost izvođenja");
    if (visibility === "nedovoljno_vidljivo") {
      return {
        status_vidljivosti: visibility,
        potvrdena_tehnika: null,
        ocena: null,
        razlog: null,
        citirani_sony_trenuci_s: null,
      };
    }
    const score = checkedValue("trainer-score");
    const technique = $("#confirmed-technique").value.trim();
    const reason = $("#trainer-reason").value.trim();
    if (!technique || !score || !reason || !state.trainerCitations.length) {
      throw new Error("Za vidljivo izvođenje unesite tehniku, ocenu, razlog i Sony sekundu");
    }
    return {
      status_vidljivosti: visibility,
      potvrdena_tehnika: technique,
      ocena: Number(score),
      razlog: reason,
      citirani_sony_trenuci_s: state.trainerCitations.slice(),
    };
  }

  $("#add-current-second").addEventListener("click", () => {
    if (!state.selected || injury(state.selected)) return;
    const second = Number(Number(state.globalSonyTime).toFixed(3));
    if (!state.trainerCitations.some((value) => Number(value).toFixed(3) === second.toFixed(3))) {
      state.trainerCitations.push(second);
    }
    const marker = `[${second.toFixed(3)} s]`;
    const reason = $("#trainer-reason");
    if (!reason.value.includes(marker)) {
      reason.value = `${reason.value.trimEnd()}${reason.value.trim() ? " " : ""}${marker}`;
    }
    updateCitationList();
  });

  $("#trainer-assessment-form").addEventListener("submit", async (formEvent) => {
    formEvent.preventDefault();
    const selected = state.selected;
    if (!selected || injury(selected)) return;
    try {
      const response = await fetch(
        `/api/events/${encodeURIComponent(selected.event_id)}/trainer-assessments`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(trainerAssessmentPayload()),
        },
      );
      const result = await readResult(response, "Procena trenera nije sačuvana");
      replaceSelectedEvent(result.event);
      status(revealedAiEvaluation(result.event)
        ? "Korekcija trenera je sačuvana"
        : "Procena je zaključana. AI još nije prikazan.");
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#reveal-ai-button").addEventListener("click", async () => {
    const selected = state.selected;
    if (!selected || injury(selected)) return;
    try {
      const response = await fetch(
        `/api/events/${encodeURIComponent(selected.event_id)}/ai-reveal`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        },
      );
      const result = await readResult(response, "AI procena nije dostupna");
      replaceSelectedEvent(result.event);
      status("AI procena je otkrivena. Uporedite dokaze i odbranite trenersku procenu.");
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#ai-feedback").addEventListener("submit", async (formEvent) => {
    formEvent.preventDefault();
    const selected = state.selected;
    if (!selected || injury(selected)) return;
    const relation = checkedValue("ai-relation");
    if (!relation) {
      status("Izaberite odnos prema AI proceni", true);
      return;
    }
    const evidenceRatings = [...document.querySelectorAll("select[data-evidence-metric]")]
      .filter((select) => select.value)
      .map((select) => ({
        metrika: select.dataset.evidenceMetric,
        odnos: select.value,
      }));
    try {
      const response = await fetch(
        `/api/events/${encodeURIComponent(selected.event_id)}/ai-feedback`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            odnos: relation,
            razlog: $("#feedback-reason").value.trim() || null,
            procene_dokaza: evidenceRatings,
          }),
        },
      );
      const result = await readResult(response, "Odgovor trenera nije sačuvan");
      replaceSelectedEvent(result.event);
      status("Odgovor na AI procenu je sačuvan");
    } catch (error) {
      status(error.message, true);
    }
  });

  $("#master-seek").addEventListener("input", (event) => seekSony(event.target.value));
  $("[data-action='toggle-play']").addEventListener("click", async () => {
    if (!sony.paused) {
      sony.pause();
      iphone.pause();
      return;
    }
    try {
      await Promise.all([sony.play(), iphone.play()]);
    } catch (_error) {
      sony.pause();
      iphone.pause();
    }
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
      correctIphonePlayback(times);
      updateReadout();
      updateCorrectionControls();
      drawCharts();
    }
  });
  sony.addEventListener("pause", () => {
    if (!iphone.paused) iphone.pause();
    if (state.selected) {
      const times = localTimesForGlobal(
        globalSonyTimeForLocal(sony.currentTime, state.selected),
        state.selected,
      );
      correctIphonePlayback(times, true);
    }
  });

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
