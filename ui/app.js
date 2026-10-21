/**
 * SmartRoute Presentation Engine & Live Audit Client
 * Powers the real-time pipeline stepper, feature extraction gauges,
 * feedback simulation, and SQLite WAL inspection.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const promptInput = document.getElementById("prompt-input");
  const charCounter = document.getElementById("char-counter");
  const btnDispatch = document.getElementById("btn-dispatch");
  const btnClear = document.getElementById("btn-clear");
  const responseContent = document.getElementById("response-content");
  const btnCopyResponse = document.getElementById("btn-copy-response");
  const toggleSimulateHook = document.getElementById("toggle-simulate-hook");
  const pipelineBadge = document.getElementById("pipeline-status-badge");

  // Pipeline Stepper Elements
  const stepClassify = document.getElementById("step-classify");
  const stepEngine = document.getElementById("step-engine");
  const stepDispatch = document.getElementById("step-dispatch");
  const stepStorage = document.getElementById("step-storage");
  const durClassify = document.getElementById("duration-classify");
  const durEngine = document.getElementById("duration-engine");
  const durDispatch = document.getElementById("duration-dispatch");
  const durStorage = document.getElementById("duration-storage");

  // Decision Meta Elements
  const metaModel = document.getElementById("meta-model-used");
  const metaTaskType = document.getElementById("meta-task-type");
  const metaComplexity = document.getElementById("meta-complexity");
  const metaLatency = document.getElementById("meta-latency");
  const metaCost = document.getElementById("meta-cost");
  const metaReason = document.getElementById("meta-reason");

  // Global Header Stats
  const statDecisions = document.getElementById("stat-total-decisions");
  const statMeanOverhead = document.getElementById("stat-mean-overhead");
  const statCostSaved = document.getElementById("stat-cost-saved");
  const btnRefreshStats = document.getElementById("btn-refresh-stats");

  // Terminal & Feedback
  const terminalOutput = document.getElementById("terminal-output");
  const btnClearTerminal = document.getElementById("btn-clear-terminal");
  const feedbackAlert = document.getElementById("feedback-alert");

  // State
  let lastDecisionId = null;
  let isExecuting = false;

  // Preset Scenarios
  const SCENARIOS = {
    general: "Explain Python decorators in simple terms with a 3-line code example.",
    code: "Write an optimal O(n) algorithm in Python to find the longest palindromic substring with unit tests.",
    reasoning: "Analyze the time complexity of the median-of-medians selection algorithm step-by-step and prove that worst-case recurrence is O(n).",
    override: "Incident: database deadlock detected on production primary cluster. Mitigate immediately."
  };

  // ------------------------------------------------------------------
  // Logging Utilities
  // ------------------------------------------------------------------

  function logTerminal(message, type = "info") {
    const time = new Date().toTimeString().split(" ")[0];
    const line = document.createElement("div");
    line.className = `log-line log-${type}`;
    line.textContent = `[${time}] [${type.toUpperCase()}] ${message}`;
    terminalOutput.appendChild(line);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
  }

  // ------------------------------------------------------------------
  // UI Helpers
  // ------------------------------------------------------------------

  function updateCharCount() {
    const text = promptInput.value || "";
    const chars = text.length;
    const estTokens = Math.ceil(chars / 4);
    charCounter.textContent = `${chars} chars | ~${estTokens} tokens`;
  }

  function resetPipelineSteps() {
    [stepClassify, stepEngine, stepDispatch, stepStorage].forEach(el => {
      el.className = "pipeline-step";
    });
    durClassify.textContent = "-- ms";
    durEngine.textContent = "-- ms";
    durDispatch.textContent = "-- ms";
    durStorage.textContent = "-- ms";
  }

  function setPipelineBadge(text, state) {
    pipelineBadge.textContent = text;
    pipelineBadge.className = `badge badge-${state}`;
  }

  // ------------------------------------------------------------------
  // Feature Gauges & Tables Rendering
  // ------------------------------------------------------------------

  function renderFeatures(feats, taskType, complexity, confidence) {
    if (!feats) return;
    document.getElementById("feat-tokens").textContent = feats.token_count || 0;
    document.getElementById("fill-tokens").style.width = `${Math.min(100, (feats.token_count / 500) * 100)}%`;

    document.getElementById("feat-code-ratio").textContent = feats.code_block_ratio || "0.00";
    document.getElementById("fill-code-ratio").style.width = `${Math.min(100, feats.code_block_ratio * 100)}%`;

    document.getElementById("feat-multi-step").textContent = feats.multi_step_count || 0;
    document.getElementById("fill-multi-step").style.width = `${Math.min(100, (feats.multi_step_count / 5) * 100)}%`;

    document.getElementById("feat-verbs").textContent = feats.instruction_verb_count || 0;
    document.getElementById("fill-verbs").style.width = `${Math.min(100, (feats.instruction_verb_count / 6) * 100)}%`;

    document.getElementById("feat-domain").textContent = feats.domain || "general";
    document.getElementById("feat-domain-ratio").textContent = `Ratio: ${feats.domain_match_ratio || "0.00"}`;

    document.getElementById("feat-confidence").textContent = confidence !== undefined ? confidence : "0.50";
    document.getElementById("fill-confidence").style.width = `${Math.min(100, (confidence || 0.5) * 100)}%`;
  }

  function renderReputationTable(reputation) {
    const tbody = document.getElementById("tbody-reputation");
    if (!reputation || reputation.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No reputation records in SQLite yet.</td></tr>';
      return;
    }

    tbody.innerHTML = reputation.map(r => {
      const isDegraded = r.ema_score < 0.30;
      const badgeClass = isDegraded ? "score-degraded" : "score-healthy";
      const statusText = isDegraded ? "Auto-Bump Triggered" : "Healthy";
      return `
        <tr>
          <td class="font-mono">${r.bucket_key}</td>
          <td><span class="badge badge-idle">${r.model_tier}</span></td>
          <td><span class="score-badge ${badgeClass}">${Number(r.ema_score).toFixed(3)}</span></td>
          <td class="font-mono">${r.call_count}</td>
          <td><span class="${isDegraded ? 'log-error' : 'log-success'}">${statusText}</span></td>
        </tr>
      `;
    }).join("");
  }

  function renderDecisionsTable(decisions) {
    const tbody = document.getElementById("tbody-decisions");
    if (!decisions || decisions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="empty-cell">No decisions logged.</td></tr>';
      return;
    }

    tbody.innerHTML = decisions.map(d => {
      const shortHash = d.prompt_hash ? d.prompt_hash.substring(0, 16) + "..." : "hash_err";
      const preview = d.prompt_preview ? (d.prompt_preview.length > 28 ? d.prompt_preview.substring(0, 28) + "..." : d.prompt_preview) : "";
      return `
        <tr>
          <td title="${d.prompt_hash}" style="color: #60a5fa;">${shortHash}</td>
          <td style="color: #94a3b8;">${preview}</td>
          <td><span class="badge badge-idle">${d.model_used || d.provider_key}</span></td>
          <td>${d.latency_ms}ms</td>
        </tr>
      `;
    }).join("");
  }

  // ------------------------------------------------------------------
  // API Calls
  // ------------------------------------------------------------------

  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) throw new Error("stats network failure");
      const data = await res.json();

      if (data.report) {
        statDecisions.textContent = data.report.total_decisions || 0;
        statCostSaved.textContent = `$${Number(data.report.total_cost_usd || 0).toFixed(4)}`;
      }
      renderReputationTable(data.reputation);
      renderDecisionsTable(data.decisions);
      logTerminal("SQLite state synchronized with active WAL database.", "info");
    } catch (err) {
      logTerminal(`Stats fetch failed: ${err.message}`, "error");
    }
  }

  async function executeRouting() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a query prompt to route.");
      return;
    }

    if (isExecuting) return;
    isExecuting = true;
    btnDispatch.disabled = true;
    setPipelineBadge("Processing", "processing");
    resetPipelineSteps();

    logTerminal(`Ingress prompt (${prompt.length} chars) received.`, "router");

    // Stage 1: Classify Animation
    stepClassify.classList.add("step-active");
    logTerminal("Evaluating 9 deterministic heuristic features (no LLM latency penalty)...", "router");

    const payload = {
      prompt: prompt,
      conversation_id: "presentation_session",
      override_provider: toggleSimulateHook.checked ? "anthropic" : null
    };

    const startTime = performance.now();

    try {
      const res = await fetch("/api/route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const meta = data.meta;

      lastDecisionId = meta.decision_id;

      // Update Stage 1: Classify Complete
      stepClassify.classList.remove("step-active");
      stepClassify.classList.add("step-complete");
      durClassify.textContent = `${meta.classification_latency_ms} ms`;
      document.getElementById("detail-classify").textContent = `Complexity: ${meta.complexity} (${meta.complexity_bucket.toUpperCase()})`;
      logTerminal(`Classification complete in ${meta.classification_latency_ms}ms -> Task: ${meta.task_type}, Tier: ${meta.complexity_bucket}`, "success");

      // Update Stage 2: Hierarchy Engine Complete
      stepEngine.classList.add("step-complete");
      durEngine.textContent = "< 1 ms";
      const overrideTag = meta.override_applied ? "Level 1 (Override)" : "Level 4 (Default Mapping)";
      document.getElementById("detail-engine").textContent = `${overrideTag} -> ${meta.model}`;
      logTerminal(`Hierarchy decision resolved -> ${meta.reason}`, "router");

      // Update Stage 3: Dispatch Complete
      stepDispatch.classList.add("step-complete");
      durDispatch.textContent = `${meta.provider_latency_ms} ms`;
      document.getElementById("detail-dispatch").textContent = `Provider returned 200 OK (${meta.model})`;
      logTerminal(`Dispatched to ${meta.model} in ${meta.provider_latency_ms}ms.`, "success");

      // Update Stage 4: Storage Complete
      stepStorage.classList.add("step-complete");
      durStorage.textContent = "< 2 ms";
      document.getElementById("detail-storage").textContent = `SHA-256: ${meta.decision_id.substring(0, 8)}...`;
      logTerminal(`Persisted decision record into SQLite WAL with SHA-256 hash.`, "info");

      // Populate Response & Metadata
      responseContent.textContent = data.text;
      metaModel.textContent = meta.model;
      metaTaskType.textContent = `${meta.task_type} (${meta.complexity_bucket})`;
      metaComplexity.textContent = meta.complexity.toFixed(2);
      metaLatency.textContent = `${meta.total_latency_ms} ms`;
      metaCost.textContent = `$${Number(meta.estimated_cost_usd || 0).toFixed(6)}`;
      metaReason.textContent = meta.reason;

      // Render Features & Tables
      renderFeatures(data.features, meta.task_type, meta.complexity, meta.confidence);
      if (data.db_records) {
        renderReputationTable(data.db_records.reputation);
        renderDecisionsTable(data.db_records.decisions);
      }

      setPipelineBadge("Complete", "success");
      fetchStats();

    } catch (err) {
      logTerminal(`Routing failure: ${err.message}`, "error");
      setPipelineBadge("Error", "idle");
      responseContent.textContent = `Error executing route: ${err.message}`;
    } finally {
      isExecuting = false;
      btnDispatch.disabled = false;
    }
  }

  async function sendFeedbackSignal(signalType, numericValue) {
    if (!lastDecisionId) {
      alert("Please dispatch at least one query before submitting a feedback signal.");
      return;
    }

    logTerminal(`Injecting user feedback signal: ${signalType} (${numericValue > 0 ? '+' : ''}${numericValue})`, "signal");

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          decision_id: lastDecisionId,
          signal_type: signalType
        })
      });

      const data = await res.json();
      if (data.status === "success") {
        logTerminal(`Feedback recorded! SQLite reputation table updated via EMA formula.`, "success");
        if (data.db_records) {
          renderReputationTable(data.db_records.reputation);
        }

        // Show Visual Feedback Banner
        feedbackAlert.className = "feedback-alert alert-success";
        feedbackAlert.innerHTML = `Signal <strong>${signalType}</strong> recorded (${numericValue}). EMA updated: <code>EMA_new = 0.3*(${numericValue}) + 0.7*EMA_old</code>.`;
        feedbackAlert.classList.remove("hidden");

        setTimeout(() => {
          feedbackAlert.classList.add("hidden");
        }, 5000);
      }
    } catch (err) {
      logTerminal(`Feedback error: ${err.message}`, "error");
    }
  }

  // ------------------------------------------------------------------
  // Event Listeners
  // ------------------------------------------------------------------

  promptInput.addEventListener("input", updateCharCount);

  // Scenario Chips Click
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const scenarioKey = chip.getAttribute("data-scenario");
      if (SCENARIOS[scenarioKey]) {
        promptInput.value = SCENARIOS[scenarioKey];
        updateCharCount();
        logTerminal(`Loaded scenario: "${scenarioKey}"`, "info");
      }
    });
  });

  // Tab Switching
  document.querySelectorAll(".tab-btn").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      tab.classList.add("active");
      const tabId = tab.getAttribute("data-tab");
      const targetContent = document.getElementById(`tab-${tabId}`);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  // Action Buttons
  btnDispatch.addEventListener("click", executeRouting);
  btnClear.addEventListener("click", () => {
    promptInput.value = "";
    updateCharCount();
    resetPipelineSteps();
    responseContent.innerHTML = '<div class="placeholder-text">Ready for prompt execution.</div>';
    setPipelineBadge("System Ready", "idle");
  });

  btnCopyResponse.addEventListener("click", () => {
    const text = responseContent.textContent;
    if (text) {
      navigator.clipboard.writeText(text);
      btnCopyResponse.textContent = "Copied!";
      setTimeout(() => { btnCopyResponse.textContent = "Copy Response"; }, 1800);
    }
  });

  // Feedback Signal Buttons
  document.getElementById("btn-sim-hard-regen").addEventListener("click", () => {
    sendFeedbackSignal("hard_regen", -0.30);
  });
  document.getElementById("btn-sim-correction").addEventListener("click", () => {
    sendFeedbackSignal("explicit_correction", -0.20);
  });
  document.getElementById("btn-sim-acceptance").addEventListener("click", () => {
    sendFeedbackSignal("acceptance", 0.05);
  });

  // Reset Reputation
  document.getElementById("btn-reset-reputation").addEventListener("click", async () => {
    if (confirm("Reset all learned EMA weights across all buckets back to baseline?")) {
      try {
        await fetch("/api/reset", { method: "POST" });
        logTerminal("All learned EMA reputations reset to baseline.", "warning");
        fetchStats();
      } catch (err) {
        logTerminal(`Reset failed: ${err.message}`, "error");
      }
    }
  });

  btnRefreshStats.addEventListener("click", fetchStats);
  btnClearTerminal.addEventListener("click", () => {
    terminalOutput.innerHTML = "";
    logTerminal("Terminal cleared by user.", "info");
  });

  // Initialize
  promptInput.value = SCENARIOS.general;
  updateCharCount();
  fetchStats();
});
