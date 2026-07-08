const state = {
  scenario: null,
  selectedDrugs: [],
  selectedFactors: [],
  lastSessionId: null,
  lastResult: null,
};

function drugPickLimit() {
  return state.scenario?.drug_picks ?? 2;
}

function factorPickLimit() {
  return state.scenario?.factor_picks ?? 3;
}

const steps = {
  select: document.getElementById("step-select"),
  loading: document.getElementById("step-loading"),
  results: document.getElementById("step-results"),
};

const LOADING_MESSAGES = [
  "Loading crisis scenario",
  "Applying your factor weights",
  "Planning under uncertainty",
  "Learner refocusing attention",
  "Comparing downstream outcomes",
];

let loadingInterval = null;

function showStep(name) {
  Object.values(steps).forEach((el) => el.classList.remove("active"));
  steps[name].classList.add("active");

  const header = document.getElementById("page-header");
  if (header) {
    header.style.display = name === "select" ? "block" : "none";
  }

  document.getElementById("play").scrollIntoView({ behavior: "smooth", block: "start" });
}

function toggleSelection(list, value, max) {
  const idx = list.indexOf(value);
  if (idx >= 0) {
    list.splice(idx, 1);
    return;
  }
  if (list.length >= max) {
    list.shift();
  }
  list.push(value);
}

function fmtTrust(trust) {
  const t = Number(trust);
  if (Number.isNaN(t)) return "";
  return `<span class="cell-trust">trust ${t.toFixed(2)}</span>`;
}

function fmtValueWithTrust(value, trust) {
  if (trust == null || trust === undefined) {
    return `<span class="cell-value">${value}</span>`;
  }
  return `<span class="cell-value">${value}</span>${fmtTrust(trust)}`;
}

function fmtUnc(value, uncertainty) {
  return fmtValueWithTrust(value, uncertainty);
}

function drugLabel(id) {
  const drug = state.scenario?.drugs.find((d) => d.id === id);
  return drug ? `Drug ${drug.label}` : id;
}

function factorLabel(id) {
  const factor = state.scenario?.factors.find((f) => f.id === id);
  return factor ? factor.label : id;
}

function updatePickCounters() {
  const drugCounter = document.getElementById("drug-counter");
  const factorCounter = document.getElementById("factor-counter");
  const briefingPanel = document.getElementById("briefing-panel");
  const factorPanel = document.getElementById("factor-panel");

  drugCounter.textContent = `${state.selectedDrugs.length} / ${drugPickLimit()} drugs`;
  factorCounter.textContent = `${state.selectedFactors.length} / ${factorPickLimit()}`;

  const drugsDone = state.selectedDrugs.length === drugPickLimit();
  const factorsDone = state.selectedFactors.length === factorPickLimit();

  drugCounter.classList.toggle("complete", drugsDone);
  factorCounter.classList.toggle("complete", factorsDone);
  briefingPanel?.classList.toggle("complete", drugsDone);
  factorPanel?.classList.toggle("complete", factorsDone);
}

function renderPosterTable() {
  const thead = document.getElementById("poster-table-head");
  const tbody = document.getElementById("poster-table-body");
  const drugs = state.scenario.drugs;

  thead.innerHTML = `
    <th scope="col" class="corner-cell"></th>
    ${drugs.map((drug) => {
      const selected = state.selectedDrugs.includes(drug.id);
      return `
        <th scope="col" class="drug-col-head ${selected ? "selected" : ""}" data-drug-id="${drug.id}" type="button">
          <button type="button" class="drug-col-btn" data-drug-id="${drug.id}" aria-pressed="${selected}">
            ${drug.label}
          </button>
        </th>
      `;
    }).join("")}
  `;

  const rows = [
    { label: "Quantity on hand", values: drugs.map((d) => fmtUnc(d.qoh, d.qoh_uncertainty)) },
    { label: "Utilization rate", values: drugs.map((d) => fmtUnc(d.utz, d.utz_uncertainty)) },
    { label: "Clinical impact", values: drugs.map((d) => `<span class="cell-value">${d.clinical}</span>`) },
    { label: "ERD (weeks)", values: drugs.map((d) => fmtUnc(d.resupply_weeks, d.resupply_uncertainty)) },
  ];

  tbody.innerHTML = rows.map((row) => `
    <tr>
      <th scope="row">${row.label}</th>
      ${row.values.map((v, i) => {
        const drug = drugs[i];
        const selected = state.selectedDrugs.includes(drug.id);
        return `<td class="drug-col-cell ${selected ? "selected" : ""}" data-drug-id="${drug.id}">${v}</td>`;
      }).join("")}
    </tr>
  `).join("");

  thead.querySelectorAll(".drug-col-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      toggleSelection(state.selectedDrugs, btn.dataset.drugId, drugPickLimit());
      renderPosterTable();
      updatePickCounters();
      updateContinueButton();
    });
  });

  tbody.querySelectorAll(".drug-col-cell").forEach((cell) => {
    cell.addEventListener("click", () => {
      toggleSelection(state.selectedDrugs, cell.dataset.drugId, drugPickLimit());
      renderPosterTable();
      updatePickCounters();
      updateContinueButton();
    });
  });
}

function renderFactors() {
  const grid = document.getElementById("factor-grid");
  grid.innerHTML = "";

  state.scenario.factors.forEach((factor) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "factor-chip";
    btn.dataset.id = factor.id;
    const pickIdx = state.selectedFactors.indexOf(factor.id);
    const isSelected = pickIdx >= 0;
    btn.innerHTML = isSelected
      ? `<span class="factor-pick-rank">${pickIdx + 1}</span>${factor.label}`
      : factor.label;
    if (isSelected) {
      btn.classList.add("selected");
    }
    btn.addEventListener("click", () => {
      toggleSelection(state.selectedFactors, factor.id, factorPickLimit());
      renderFactors();
      updatePickCounters();
      updateContinueButton();
    });
    grid.appendChild(btn);
  });
}

function updateContinueButton() {
  const ready =
    state.selectedDrugs.length === drugPickLimit()
    && state.selectedFactors.length === factorPickLimit();
  const btn = document.getElementById("btn-run");
  const label = btn.querySelector(".btn-label");
  btn.disabled = !ready;
  if (ready) {
    label.textContent = "Face the Learner Agent";
    return;
  }
  const parts = [];
  const drugsLeft = drugPickLimit() - state.selectedDrugs.length;
  const factorsLeft = factorPickLimit() - state.selectedFactors.length;
  if (drugsLeft > 0) parts.push(`${drugsLeft} drug${drugsLeft > 1 ? "s" : ""}`);
  if (factorsLeft > 0) parts.push(`${factorsLeft} factor${factorsLeft > 1 ? "s" : ""}`);
  label.textContent = `Select ${parts.join(" & ")}`;
}

async function loadScenario() {
  const response = await fetch("/api/scenario");
  state.scenario = await response.json();
  renderPosterTable();
  renderFactors();
  updatePickCounters();
}

async function loadBackgrounds() {
  const response = await fetch("/api/backgrounds");
  const data = await response.json();
  const select = document.getElementById("background-select");
  data.options.forEach((opt) => {
    const option = document.createElement("option");
    option.value = opt.id;
    option.textContent = opt.label;
    select.appendChild(option);
  });
}

function formatOutcome(summary, weeks) {
  const stockouts = summary.total_stockouts;
  if (stockouts === 0) return "None";
  if (stockouts === 1) return "1 drug-week";
  return `${stockouts} drug-weeks`;
}

function fmtReward(n) {
  const v = Number(n);
  if (Number.isNaN(v)) return n;
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

function stockoutScore(summary) {
  return summary.total_stockouts;
}

function shortSessionId(sessionId) {
  return (sessionId || "").slice(0, 8);
}

function shareUrl(sessionId) {
  return `${window.location.origin}/r/${sessionId}`;
}

const PICK_WEIGHTS = [0.5, 0.35, 0.15];

/** Briefing-table factor order (all 7 poster rows). */
const POSTER_FACTORS = [
  { id: "qoh", weightKey: "qoh", label: "Quantity on hand" },
  { id: "qoh_uncertainty", weightKey: "qoh_uncertainty", label: "QOH uncertainty" },
  { id: "usage", weightKey: "usage", label: "Utilization rate" },
  { id: "usage_uncertainty", weightKey: "usage_uncertainty", label: "Utilization uncertainty" },
  { id: "clinical", weightKey: "clinical", label: "Clinical impact" },
  { id: "erd", weightKey: "runway", label: "ERD (weeks)" },
  { id: "erd_uncertainty", weightKey: "runway_uncertainty", label: "ERD uncertainty" },
];

/** Learner-only urgency signal (always in β vector, not a briefing pick). */
const LEARNER_EXTRA_FACTOR = {
  id: "reputation",
  weightKey: "reputation",
  label: "Shortage history",
};

const LEARNER_WEIGHT_LABELS = Object.fromEntries(
  [...POSTER_FACTORS, LEARNER_EXTRA_FACTOR].map((f) => [f.weightKey, f.label]),
);

function learnerFactorWeightMap(weights) {
  const w = weights || {};
  return Object.fromEntries(
    [...POSTER_FACTORS, LEARNER_EXTRA_FACTOR].map((f) => [f.id, w[f.weightKey] ?? 0]),
  );
}

function learnerTopFactors(weights, limit = 3) {
  const w = weights || {};
  return Object.entries(w)
    .filter(([, val]) => val > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([key, val]) => ({
      label: LEARNER_WEIGHT_LABELS[key] || key,
      w: val,
    }));
}

function fmtPct(weight) {
  const v = Number(weight);
  if (!v || Number.isNaN(v)) return "0%";
  return `${Math.round(v * 100)}%`;
}

function visitorFactorWeightMap(selectedFactors) {
  const map = Object.fromEntries(POSTER_FACTORS.map((f) => [f.id, 0]));
  (selectedFactors || []).forEach((id, i) => {
    if (id in map && i < PICK_WEIGHTS.length) {
      map[id] = PICK_WEIGHTS[i];
    }
  });
  return map;
}

function buildShareSummary(data) {
  const card = data.result_card;
  if (!card) return "Attention Policy Demo CogSci 2026";
  const title = (card.title || "").replace(/_/g, " ");
  const stock =
    card.your_stockouts === card.learner_stockouts
      ? `${card.your_stockouts} stockouts each`
      : `stockouts ${card.your_stockouts} vs ${card.learner_stockouts}`;
  return [
    title,
    `You ${fmtReward(card.your_score)}/wk vs Learner ${fmtReward(card.learner_score)}/wk`,
    `Week 1 focus: ${card.initial_focus}`,
    stock,
    "The first decision is where to look.",
  ].join(" · ");
}

function renderYouPriorityStack(selectedFactors) {
  const rankLabels = ["1st pick", "2nd pick", "3rd pick"];
  return (selectedFactors || []).map((id, i) => {
    const factor = POSTER_FACTORS.find((f) => f.id === id);
    return `
      <div class="priority-card you-side">
        <span class="priority-rank">${rankLabels[i]}</span>
        <span class="priority-name">${factor?.label || id}</span>
        <span class="priority-pct">${fmtPct(PICK_WEIGHTS[i])}</span>
      </div>
    `;
  }).join("");
}

function renderLearnerPriorityStack(learnerWeights) {
  return learnerTopFactors(learnerWeights, 3).map((f, i) => `
    <div class="priority-card learner-side">
      <span class="priority-rank">Top ${i + 1}</span>
      <span class="priority-name">${f.label}</span>
      <span class="priority-pct">${fmtPct(f.w)}</span>
    </div>
  `).join("");
}

function renderBetaLane(weight, sideClass, pickRank) {
  const pct = Number(weight) || 0;
  if (pct <= 0) {
    return `
      <div class="beta-lane ${sideClass} beta-lane-empty">
        <span class="beta-lane-off">not weighted</span>
      </div>
    `;
  }
  const fillPct = Math.max(4, pct * 100);
  const rank = pickRank
    ? `<span class="beta-pick-rank" title="Your pick ${pickRank}">${pickRank}</span>`
    : `<span class="beta-pick-spacer" aria-hidden="true"></span>`;
  return `
    <div class="beta-lane ${sideClass}">
      ${rank}
      <div class="beta-lane-track">
        <div class="beta-lane-fill" style="width:${fillPct}%"></div>
      </div>
      <span class="beta-lane-pct">${fmtPct(pct)}</span>
    </div>
  `;
}

function renderFactorPriorityTable(selectedFactors, youWeights, learnerWeights) {
  const youMap = visitorFactorWeightMap(selectedFactors);
  const learnerMap = learnerFactorWeightMap(learnerWeights);
  const compareFactors = [...POSTER_FACTORS, LEARNER_EXTRA_FACTOR];
  const pickRankById = Object.fromEntries(
    (selectedFactors || []).map((id, i) => [id, i + 1]),
  );

  const rows = compareFactors.map((factor) => {
    const youW = factor.id === LEARNER_EXTRA_FACTOR.id ? 0 : youMap[factor.id];
    const learnerW = learnerMap[factor.id];
    const active = youW > 0 || learnerW > 0.04;
    const learnerOnly = factor.id === LEARNER_EXTRA_FACTOR.id;
    return `
      <div class="beta-compare-row ${active ? "beta-row-active" : "beta-row-quiet"} ${learnerOnly ? "beta-row-learner-only" : ""}">
        <span class="beta-factor-name">${factor.label}${learnerOnly ? '<span class="beta-factor-note">learner only</span>' : ""}</span>
        ${renderBetaLane(youW, "you-side", pickRankById[factor.id])}
        ${renderBetaLane(learnerW, "learner-side", null)}
      </div>
    `;
  }).join("");

  return `
    <section class="story-priorities">
      <h3 class="story-section-label">What each policy weighted</h3>
      <p class="priority-legend">
        Attention weights turn signals into urgency scores. Your three picks were fixed at
        <strong>50% · 35% · 15%</strong> by click order. The Learner's weights adapt across
        all urgency signals (including shortage history).
      </p>

      <div class="priority-stacks">
        <div class="priority-stack">
          <span class="priority-stack-title you-side">Your attention</span>
          <div class="priority-stack-cards">${renderYouPriorityStack(selectedFactors)}</div>
        </div>
        <div class="priority-stack">
          <span class="priority-stack-title learner-side">Learner attention</span>
          <div class="priority-stack-cards">${renderLearnerPriorityStack(learnerWeights)}</div>
        </div>
      </div>

      <details class="beta-compare-details" open>
        <summary>All factors compared (7 initial factors + 1 learner signal)</summary>
        <div class="beta-compare-head">
          <span class="beta-head-factor">Factor</span>
          <span class="beta-head-you">You</span>
          <span class="beta-head-learner">Learner</span>
        </div>
        <div class="beta-compare-rows">${rows}</div>
      </details>
    </section>
  `;
}

function formatDrugShort(drug) {
  return (drug || "").replace("Drug ", "");
}

function formatDrugList(drugs) {
  return (drugs || []).map(formatDrugShort).join(" & ") || ", ";
}

function summarizeWeekActions(week) {
  const acts = (week.drug_details || [])
    .flatMap((d) => d.actions.filter((a) => a !== "Monitor" && a !== "No actions taken"))
    .map((a) => a.replace(/_/g, " ").toLowerCase());
  if (!acts.length) return "monitor only";
  return acts.slice(0, 3).join(" · ");
}

function focusSetsEqual(a, b) {
  const sa = [...(a || [])].map(formatDrugShort).sort().join(",");
  const sb = [...(b || [])].map(formatDrugShort).sort().join(",");
  return sa === sb;
}

function renderSpotlightJourney(drugLabels, weeksFocus, sideClass, title) {
  const cols = weeksFocus.map((focused, wi) => {
    const prev = wi > 0 ? weeksFocus[wi - 1] : null;
    const shifted = prev && !focusSetsEqual(focused, prev);
    return `
      <div class="spotlight-week ${shifted ? "shifted" : ""}">
        <span class="spotlight-wk">W${wi + 1}</span>
        <div class="spotlight-slots">
          ${drugLabels.map((label) => {
            const on = focused.includes(label);
            return `<span class="spotlight-slot ${on ? "lit" : ""}" aria-label="Drug ${label}${on ? " in focus" : ""}">${label}</span>`;
          }).join("")}
        </div>
        <span class="spotlight-focus-pair">${focused.join(" · ") || ", "}</span>
      </div>
    `;
  }).join("");

  return `
    <div class="spotlight-journey ${sideClass}">
      <span class="spotlight-journey-title">${title}</span>
      <div class="spotlight-drug-header">
        ${drugLabels.map((l) => `<span>${l}</span>`).join("")}
      </div>
      <div class="spotlight-weeks">${cols}</div>
    </div>
  `;
}

function renderWeekActionRows(youWeeks, learnerWeeks) {
  return youWeeks.map((yw, i) => {
    const lw = learnerWeeks[i];
    const diverged = !focusSetsEqual(
      yw.focused_drugs,
      lw?.focused_drugs,
    );
    return `
      <div class="story-action-row ${diverged ? "diverged" : ""}">
        <span class="story-action-wk">W${i + 1}</span>
        <div class="story-action-you">
          <span class="story-action-focus">${formatDrugList(yw.focused_drugs)}</span>
          <span class="story-action-acts">${summarizeWeekActions(yw)}</span>
        </div>
        <div class="story-action-learner">
          <span class="story-action-focus">${formatDrugList(lw?.focused_drugs)}</span>
          <span class="story-action-acts">${summarizeWeekActions(lw)}</span>
        </div>
        <div class="story-action-rewards">
          <span class="you">${fmtReward(yw.total_reward)}</span>
          <span class="learner">${fmtReward(lw?.total_reward)}</span>
        </div>
      </div>
    `;
  }).join("");
}

function outcomeVerdict(card) {
  const delta = card.reward_delta;
  const vStock = card.your_stockouts;
  const lStock = card.learner_stockouts;
  if (lStock < vStock) {
    return `Fewer stockouts for the Learner (${lStock} vs ${vStock} drug-weeks).`;
  }
  if (vStock < lStock) {
    return `You had fewer stockouts (${vStock} vs ${lStock} drug-weeks).`;
  }
  if (Math.abs(delta) < 5) {
    return "Similar outcomes, but focus paths still diverged.";
  }
  if (delta > 0) {
    return `Learner averaged +${fmtReward(delta)} more per week.`;
  }
  return `You averaged +${fmtReward(Math.abs(delta))} more per week.`;
}

function renderAttentionStory(data) {
  const el = document.getElementById("attention-story");
  const card = data.result_card;
  if (!el || !card) return;

  const drugLabels = card.drug_labels;
  const youWeeks = focusByWeek(data.your_policy.weeks);
  const learnerWeeks = focusByWeek(data.learner_agent.weeks);
  const stockoutYou = formatOutcome(data.your_policy.summary, card.weeks_count);
  const stockoutLearner = formatOutcome(data.learner_agent.summary, card.weeks_count);

  el.innerHTML = `
    <header class="story-thesis">
      <p class="story-thesis-main">${data.thesis_main || "The first decision is where to look."}</p>
      <p class="story-thesis-sub">${data.thesis_sub || "Attention weights turn inventory signals into urgency scores, only the two highest-urgency drugs received deep planning each week."}</p>
      ${data.interpretation ? `<p class="story-interpretation">${data.interpretation}</p>` : ""}
    </header>

    ${renderFactorPriorityTable(
      data.selected_factors,
      data.your_policy.weights,
      data.learner_agent.weights,
    )}

    <section class="story-journeys">
      <h3 class="story-section-label">Where attention went each week</h3>
      <div class="story-journeys-grid">
        ${renderSpotlightJourney(drugLabels, youWeeks, "you-side", "Your spotlight")}
        ${renderSpotlightJourney(drugLabels, learnerWeeks, "learner-side", "Learner spotlight")}
      </div>
      <p class="story-journey-note">Bright = deep planning. Dim = monitored only.</p>
    </section>

    <section class="story-actions">
      <h3 class="story-section-label">What each policy did on focused drugs</h3>
      <div class="story-action-head">
        <span></span>
        <span class="col-you">You</span>
        <span class="col-learner">Learner</span>
        <span class="col-reward">Reward</span>
      </div>
      <div class="story-action-rows">
        ${renderWeekActionRows(data.your_policy.weeks, data.learner_agent.weeks)}
      </div>
    </section>

    <footer class="story-outcome">
      <div class="outcome-stat you">
        <span class="outcome-val">${fmtReward(card.your_score)}</span>
        <span class="outcome-lbl">You · avg / wk</span>
      </div>
      <div class="outcome-stat learner">
        <span class="outcome-val">${fmtReward(card.learner_score)}</span>
        <span class="outcome-lbl">Learner · avg / wk</span>
      </div>
      <div class="outcome-stat neutral">
        <span class="outcome-val">${stockoutYou}</span>
        <span class="outcome-lbl">Your stockouts</span>
      </div>
      <div class="outcome-stat neutral">
        <span class="outcome-val">${stockoutLearner}</span>
        <span class="outcome-lbl">Learner stockouts</span>
      </div>
      <p class="outcome-verdict">${outcomeVerdict(card)}</p>
      <span class="story-session-id">#${shortSessionId(data.session_id)}</span>
    </footer>
  `;
}

function renderWeeklyBars(youSeries, learnerSeries) {
  const you = youSeries || [];
  const learner = learnerSeries || [];
  const weeks = you.length;
  const maxPositive = Math.max(1, ...you.filter((v) => v > 0), ...learner.filter((v) => v > 0));

  function rewardLane(value, side) {
    const cls = side === "you" ? "you" : "learner";
    const label = side === "you" ? "You" : "Learner";
    if (value < 0) {
      return `
        <div class="reward-lane ${cls}">
          <span class="reward-lane-label">${label}</span>
          <div class="reward-lane-track reward-lane-track-neg">
            <span class="reward-lane-neg">${fmtReward(value)}</span>
          </div>
        </div>
      `;
    }
    const pct = Math.max(4, (value / maxPositive) * 100);
    return `
      <div class="reward-lane ${cls}">
        <span class="reward-lane-label">${label}</span>
        <div class="reward-lane-track">
          <div class="reward-lane-bar" style="width:${pct}%"></div>
        </div>
        <span class="reward-lane-val">${fmtReward(value)}</span>
      </div>
    `;
  }

  const rows = Array.from({ length: weeks }, (_, i) => {
    const yv = you[i] ?? 0;
    const lv = learner[i] ?? 0;
    let rowClass = "";
    if (yv > lv) rowClass = "week-won-you";
    else if (lv > yv) rowClass = "week-won-learner";

    return `
      <div class="week-reward-row ${rowClass}">
        <span class="week-reward-wk">W${i + 1}</span>
        <div class="week-reward-lanes">
          ${rewardLane(yv, "you")}
          ${rewardLane(lv, "learner")}
        </div>
      </div>
    `;
  }).join("");

  const youAvg = you.reduce((sum, v) => sum + v, 0) / weeks;
  const learnerAvg = learner.reduce((sum, v) => sum + v, 0) / weeks;

  return `
    <div class="weekly-reward-chart">${rows}</div>
    <p class="week-chart-foot">5-wk avg · You ${fmtReward(youAvg)} · Learner ${fmtReward(learnerAvg)}</p>
    <div class="week-chart-legend">
      <span class="chart-key you">You</span>
      <span class="chart-key learner">Learner</span>
    </div>
  `;
}

function focusByWeek(weeks) {
  return weeks.map((w) =>
    w.focused_drugs.map((d) => d.replace("Drug ", ""))
  );
}

function setupShareActions(data) {
  state.lastSessionId = data.session_id;
  state.lastResult = data;
  const url = shareUrl(data.session_id);

  document.getElementById("btn-copy-link").onclick = async () => {
    try {
      await navigator.clipboard.writeText(url);
      document.getElementById("btn-copy-link").textContent = "Link copied!";
      setTimeout(() => {
        document.getElementById("btn-copy-link").textContent = "Copy link";
      }, 2000);
    } catch {
      prompt("Copy your result link:", url);
    }
  };

  const summary = buildShareSummary(data);

  document.getElementById("btn-share").onclick = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: "Attention Policy Demo",
          text: summary,
          url,
        });
        return;
      } catch {
        /* fall through */
      }
    }
    try {
      await navigator.clipboard.writeText(summary);
      document.getElementById("btn-share").textContent = "Summary copied!";
      setTimeout(() => {
        document.getElementById("btn-share").textContent = "Copy summary";
      }, 2000);
    } catch {
      prompt("Copy this summary:", summary);
    }
  };
}

function renderResults(data) {
  const card = data.result_card;

  renderAttentionStory(data);
  setupShareActions(data);

  if (card) {
    document.getElementById("weekly-bars").innerHTML = renderWeeklyBars(
      card.reward_series.you,
      card.reward_series.learner,
    );
  }
  renderStockoutHighlights(data.stockout_highlights || []);
}

function renderStockoutHighlights(highlights) {
  const el = document.getElementById("stockout-highlights");
  if (!el) return;
  if (!highlights.length) {
    el.innerHTML = "";
    el.classList.remove("visible");
    el.style.display = "none";
    return;
  }
  el.style.display = "";
  el.classList.add("visible");
  el.innerHTML = `
    <h3 class="pin-title">Stockouts</h3>
    <div class="stockout-pills">
      ${highlights.map((h) => `
        <div class="stockout-pill">
          <strong>Week ${h.week} · Drug ${h.drug}</strong>
          <p class="stockout-pill-copy">${h.your_reason}</p>
          ${h.learner_contrast ? `<p class="stockout-pill-contrast">${h.learner_contrast}</p>` : ""}
        </div>
      `).join("")}
    </div>
  `;
}

function renderWeekDrugDetails(details) {
  if (!details || !details.length) {
    return '<p class="week-quiet">No actions on focused or stockout drugs.</p>';
  }
  return `<div class="drug-action-list">${details.map((d) => `
    <div class="drug-action-row ${d.stockout ? "is-stockout" : ""}">
      <div class="drug-action-head">
        <span class="drug-action-name">${d.label}</span>
        ${d.focused
          ? '<span class="badge badge-focus">In focus</span>'
          : '<span class="badge badge-out">Outside focus</span>'}
      </div>
      <div class="drug-action-meta">
        <span class="runway-chip">${d.runway_start} to ${d.runway_end} wk runway</span>
      </div>
      <div class="action-tags">
        ${d.actions.map((a) => `<span class="action-tag${a === "No actions taken" ? " action-none" : ""}">${a}</span>`).join("")}
      </div>
      ${d.stockout_reason ? `<p class="stockout-reason">${d.stockout_reason}</p>` : ""}
    </div>
  `).join("")}</div>`;
}

function renderTags(items, extraClass = "") {
  if (!items || !items.length) return "none";
  return `<span class="tag-list">${items.map((item) =>
    `<span class="tag ${extraClass}">${item.replace("Drug ", "")}</span>`
  ).join("")}</span>`;
}

function startLoadingMessages() {
  const statusEl = document.getElementById("loading-status");
  let i = 0;
  statusEl.textContent = LOADING_MESSAGES[0];
  loadingInterval = setInterval(() => {
    i = (i + 1) % LOADING_MESSAGES.length;
    statusEl.textContent = LOADING_MESSAGES[i];
    statusEl.style.animation = "none";
    void statusEl.offsetWidth;
    statusEl.style.animation = "";
  }, 1400);
}

function stopLoadingMessages() {
  if (loadingInterval) {
    clearInterval(loadingInterval);
    loadingInterval = null;
  }
}

async function runSimulation() {
  const email = document.getElementById("email-input").value.trim();
  const name = document.getElementById("name-input").value.trim();
  if (email && !name) {
    alert("Please add your name so we know who to follow up with.");
    return;
  }

  showStep("loading");
  startLoadingMessages();

  const payload = {
    selected_drugs: state.selectedDrugs,
    selected_factors: state.selectedFactors,
    background: document.getElementById("background-select").value || null,
    name: name || null,
    email: email || null,
    consent: document.getElementById("consent-checkbox").checked,
  };

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Simulation failed");
    }

    const data = await response.json();
    stopLoadingMessages();
    renderResults(data);
    showStep("results");
    history.replaceState(null, "", shareUrl(data.session_id));
  } catch (error) {
    stopLoadingMessages();
    alert(error.message);
    showStep("select");
  }
}

function resetApp() {
  state.selectedDrugs = [];
  state.selectedFactors = [];
  document.getElementById("name-input").value = "";
  document.getElementById("email-input").value = "";
  document.getElementById("consent-checkbox").checked = false;
  document.getElementById("background-select").value = "";
  renderPosterTable();
  renderFactors();
  updatePickCounters();
  updateContinueButton();
  showStep("select");
}

document.getElementById("btn-run").addEventListener("click", () => runSimulation());
document.getElementById("btn-restart").addEventListener("click", () => {
  history.replaceState(null, "", "/");
  resetApp();
});

function getShareSessionId() {
  const match = window.location.pathname.match(/^\/r\/([0-9a-f-]{36})$/i);
  if (match) return match[1];
  return new URLSearchParams(window.location.search).get("r");
}

async function initApp() {
  await loadScenario();
  await loadBackgrounds();
  const shareId = getShareSessionId();
  if (shareId) {
    try {
      const res = await fetch(`/api/results/${shareId}`);
      if (res.ok) {
        const data = await res.json();
        renderResults(data);
        showStep("results");
        return;
      }
    } catch {
      /* play flow */
    }
  }
  showStep("select");
}

initApp();
