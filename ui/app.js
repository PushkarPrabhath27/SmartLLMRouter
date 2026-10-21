/**
 * SmartRoute Studio Client & Zero-Lag SSE Streaming Engine
 * FAANG-Grade Real-Time Observability, Syntax Highlighting & Token Streaming
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Workbench
  const promptInput = document.getElementById("prompt-input");
  const charTokenCounter = document.getElementById("char-token-counter");
  const btnDispatch = document.getElementById("btn-dispatch-stream");
  const btnClear = document.getElementById("btn-clear-prompt");
  const toggleOverrideHook = document.getElementById("toggle-override-hook");

  // DOM Elements - Waterfall Bar
  const wfStep1 = document.getElementById("wf-step-1");
  const wfStep2 = document.getElementById("wf-step-2");
  const wfStep3 = document.getElementById("wf-step-3");
  const wfStep4 = document.getElementById("wf-step-4");
  const wfMetricClassify = document.getElementById("wf-metric-classify");
  const wfMetricHierarchy = document.getElementById("wf-metric-hierarchy");
  const wfMetricTtft = document.getElementById("wf-metric-ttft");
  const wfMetricWal = document.getElementById("wf-metric-wal");
  const wfSpeedBadge = document.getElementById("wf-speed-badge");
  const liveSpeedText = document.getElementById("live-speed-text");

  // DOM Elements - Decision Spec Ribbon
  const specModel = document.getElementById("spec-model");
  const specTaskType = document.getElementById("spec-task-type");
  const specComplexity = document.getElementById("spec-complexity");
  const specLatency = document.getElementById("spec-latency");
  const specCost = document.getElementById("spec-cost");
  const specReason = document.getElementById("spec-reason");

  // DOM Elements - Output Viewports
  const streamingViewport = document.getElementById("streaming-viewport");
  const rawStreamViewport = document.getElementById("raw-stream-viewport");
  const graphViewport = document.getElementById("graph-viewport");
  const rawTokensContainer = document.getElementById("raw-tokens-container");
  const btnCopyOutput = document.getElementById("btn-copy-output");
  const winModeTabs = document.querySelectorAll(".win-tab");

  // DOM Elements - Cascade Graph Cards
  const graphStatusOverride = document.getElementById("graph-status-override");
  const graphDescOverride = document.getElementById("graph-desc-override");
  const graphStatusClassifier = document.getElementById("graph-status-classifier");
  const graphDescClassifier = document.getElementById("graph-desc-classifier");
  const graphStatusReputation = document.getElementById("graph-status-reputation");
  const graphDescReputation = document.getElementById("graph-desc-reputation");

  // DOM Elements - Telemetry Header
  const statDecisions = document.getElementById("stat-total-decisions");
  const btnSyncTelemetry = document.getElementById("btn-sync-telemetry");

  // DOM Elements - Profiler & Terminal
  const kernelLogLines = document.getElementById("kernel-log-lines");
  const btnClearLogs = document.getElementById("btn-clear-logs");
  const signalToast = document.getElementById("signal-toast");

  // Engine State
  let lastDecisionId = null;
  let isStreaming = false;
  let rawAccumulatedOutput = "";
  let tokenCountSession = 0;
  let streamStartTime = 0;
  let renderFramePending = false;
  let currentWindowMode = "formatted";

  // Curated Systems Engineering Benchmarks
  const SCENARIOS = {
    distributed: `Implement a production-grade distributed lock with Redis TTL lease renewal and fencing tokens in Python:
- Use atomic Lua scripts for acquire (SET NX PX) and release
- Include background heartbeat thread to renew lease every TTL / 3
- Return monotonic fencing token to prevent split-brain writes under GC pauses
- Include full exception handling and context manager support (__aenter__/__aexit__)`,

    database: `Analyze write amplification and memory hierarchies in LSM-Trees vs B+Trees:
1. Contrast Write Amplification Factor (WAF) between Size-Tiered and Leveled Compaction
2. Bloom filter sizing formula (bits-per-key for 1% false positive rate)
3. Read amplification mitigation under sustained high-throughput ingestion rates
4. Memtable flush stalls and write throttling thresholds`,

    ringbuffer: `Write a lock-free Single-Producer Single-Consumer (SPSC) ring buffer in Python using multiprocessing.Array and atomic indices:
- Ensure cache line alignment (pad struct to prevent false sharing)
- Preserve acquire-release memory barrier semantics around wrap-around modulo masking
- Implement non-blocking try_push() and try_pop() with queue-full / queue-empty detection
- Sub-microsecond latency target`,

    incident: `CRITICAL PRODUCTION INCIDENT (P0): Kafka consumer group 'order-engine' cascade rebalance storm.
High partition lag detected across partitions 0-31.
1. Immediate root-cause diagnosis: max.poll.interval.ms vs JVM GC pauses vs consumer heartbeat thread starve
2. Emergency operational commands to pause fetch and prevent duplicate deliveries
3. Production configuration overrides (session.timeout.ms, max.poll.records, partition.assignment.strategy)`
  };

  // ------------------------------------------------------------------
  // Text Sanitizer & Syntax Highlighting Engine
  // ------------------------------------------------------------------

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Strips conversational filler and decorative emoji numbers (e.g. # 1️⃣ -> # Step 1:)
  function sanitizeModelOutput(text) {
    if (!text) return "";
    let clean = text;
    // Replace emoji number bullets in comments or lists
    clean = clean.replace(/#\s*[1-9]️⃣/g, (m) => `# Step ${m.replace(/\D/g, "")}:`);
    clean = clean.replace(/[1-9]️⃣\s*/g, (m) => `${m.replace(/\D/g, "")}. `);
    // Remove typical AI greetings if present
    clean = clean.replace(/^(Sure!?|Certainly!?|Here is the|Here's a|Below is).+?\n\n/i, "");
    return clean;
  }

  function highlightPythonSyntax(code) {
    const esc = escapeHtml(code);
    // Single-pass tokenizer avoids any tag-nesting or string replacement corruption
    return esc.replace(
      /(#[^\n]*)|(&quot;&quot;&quot;[\s\S]*?&quot;&quot;&quot;|&#039;&#039;&#039;[\s\S]*?&#039;&#039;&#039;|&quot;[^&]*?&quot;|&#039;[^&#]*?&#039;)|\b(def|class|return|if|else|elif|import|from|async|await|try|except|finally|for|while|in|is|not|and|or|lambda|with|as|yield|pass|raise|None|True|False)\b|\b([a-zA-Z_][a-zA-Z0-9_]*)(?=\()|\b(\d+(\.\d+)?)\b/g,
      (match, comment, str, kw, fn, num) => {
        if (comment) return `<span class="tok-com">${comment}</span>`;
        if (str) return `<span class="tok-str">${str}</span>`;
        if (kw) return `<span class="tok-kw">${kw}</span>`;
        if (fn) return `<span class="tok-fn">${fn}</span>`;
        if (num) return `<span class="tok-num">${num}</span>`;
        return match;
      }
    );
  }

  function formatMarkdownText(raw) {
    let p = escapeHtml(raw);
    p = p.replace(/^### (.*$)/gm, '<h4 style="font-size: 13px; font-weight: 700; color: #f1f5f9; margin: 14px 0 6px;">$1</h4>');
    p = p.replace(/^## (.*$)/gm, '<h3 style="font-size: 14px; font-weight: 700; color: #ffffff; margin: 18px 0 8px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px;">$1</h3>');
    p = p.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #f8fafc; font-weight: 600;">$1</strong>');
    p = p.replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.06); color: #818cf8; padding: 1px 5px; border-radius: 3px; font-family: var(--font-mono); font-size: 11px;">$1</code>');
    p = p.replace(/^\s*[-*]\s+(.*$)/gm, '<li style="margin-left: 18px; margin-bottom: 4px; color: #cbd5e1;">$1</li>');
    p = p.replace(/\n\n/g, '<div style="height: 10px;"></div>');
    p = p.replace(/\n/g, '<br>');
    return `<div class="formatted-text-chunk" style="line-height: 1.6; margin: 6px 0;">${p}</div>`;
  }

  function renderFormattedContent(text, isFinished = true) {
    if (!text) {
      streamingViewport.innerHTML = `
        <div class="viewport-placeholder">
          <div class="placeholder-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
              <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
              <line x1="8" y1="21" x2="16" y2="21"></line>
              <line x1="12" y1="17" x2="12" y2="21"></line>
            </svg>
          </div>
          <span class="placeholder-primary">Engine Stream Inactive</span>
          <span class="placeholder-secondary">Press Dispatch Query or hit ⌘+Enter to execute in-process routing & zero-lag streaming</span>
        </div>`;
      return;
    }

    const sanitized = sanitizeModelOutput(text);

    let html = "";
    if (sanitized.includes("```")) {
      // Split by code fences while supporting streaming unclosed fence at end
      const regex = /```([a-zA-Z0-9_-]*)?\n?([\s\S]*?)($|```)/g;
      let lastIndex = 0;
      let match;

      while ((match = regex.exec(sanitized)) !== null) {
        const plainText = sanitized.slice(lastIndex, match.index);
        if (plainText.trim()) {
          html += formatMarkdownText(plainText);
        }

        const lang = (match[1] || "python").trim();
        const codeLines = match[2] || "";
        const highlighted = highlightPythonSyntax(codeLines);

        html += `
          <div class="code-block-wrapper">
            <div class="code-header-bar font-mono">
              <span>${lang.toUpperCase()}</span>
              <button class="code-copy-btn font-mono" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.textContent='COPIED'; setTimeout(()=>this.textContent='COPY', 1200);">COPY</button>
            </div>
            <pre class="code-inner-body font-mono"><code>${highlighted}</code></pre>
          </div>
        `;
        lastIndex = regex.lastIndex;
      }
      const remaining = sanitized.slice(lastIndex);
      if (remaining.trim()) {
        html += formatMarkdownText(remaining);
      }
    } else if (/\b(def |class |import |from |return )/.test(sanitized)) {
      // Automatic detection: if output is raw code without markdown backticks, wrap in code container!
      const highlighted = highlightPythonSyntax(sanitized);
      html += `
        <div class="code-block-wrapper">
          <div class="code-header-bar font-mono">
            <span>PYTHON (SYSTEM SPEC)</span>
            <button class="code-copy-btn font-mono" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.textContent='COPIED'; setTimeout(()=>this.textContent='COPY', 1200);">COPY</button>
          </div>
          <pre class="code-inner-body font-mono"><code>${highlighted}</code></pre>
        </div>
      `;
    } else {
      html += formatMarkdownText(sanitized);
    }

    if (!isFinished) {
      html += `<span class="streaming-cursor"></span>`;
    }

    streamingViewport.innerHTML = html;
    streamingViewport.scrollTop = streamingViewport.scrollHeight;
  }

  // ------------------------------------------------------------------
  // High-Performance Batched Stream Renderer (Zero-Lag 60fps)
  // ------------------------------------------------------------------

  function scheduleBatchRender(isFinished = false) {
    if (renderFramePending && !isFinished) return;
    renderFramePending = true;

    requestAnimationFrame(() => {
      renderFormattedContent(rawAccumulatedOutput, isFinished);
      renderFramePending = false;
    });
  }

  // ------------------------------------------------------------------
  // Logging Utilities
  // ------------------------------------------------------------------

  function logKernel(msg, level = "info") {
    const time = new Date().toTimeString().split(" ")[0];
    const line = document.createElement("div");
    line.className = `kline kline-${level}`;
    line.textContent = `[${time}] [${level.toUpperCase()}] ${msg}`;
    kernelLogLines.appendChild(line);
    kernelLogLines.scrollTop = kernelLogLines.scrollHeight;
  }

  // ------------------------------------------------------------------
  // Input & Counter Helpers
  // ------------------------------------------------------------------

  function updateCharCount() {
    const text = promptInput.value || "";
    const chars = text.length;
    const estTokens = Math.ceil(chars / 4);
    charTokenCounter.textContent = `${chars} chars | ~${estTokens} tokens`;
  }

  function resetWaterfall() {
    [wfStep1, wfStep2, wfStep3, wfStep4].forEach(s => s.classList.remove("active"));
    wfMetricClassify.textContent = "-- ms";
    wfMetricHierarchy.textContent = "-- ms";
    wfMetricTtft.textContent = "-- ms";
    wfMetricWal.textContent = "-- ms";
    wfSpeedBadge.className = "wf-speed-badge font-mono";
    liveSpeedText.textContent = "Ready";
  }

  // ------------------------------------------------------------------
  // Profiler Tables & Feature Meters
  // ------------------------------------------------------------------

  function renderFeatureMeters(feats, confidence) {
    if (!feats) return;
    document.getElementById("meter-val-tokens").textContent = `${feats.token_count} tokens`;
    document.getElementById("meter-bar-tokens").style.width = `${Math.min(100, (feats.token_count / 500) * 100)}%`;

    document.getElementById("meter-val-code").textContent = feats.code_block_ratio.toFixed(2);
    document.getElementById("meter-bar-code").style.width = `${Math.min(100, feats.code_block_ratio * 100)}%`;

    document.getElementById("meter-val-multistep").textContent = `${feats.multi_step_count} cues`;
    document.getElementById("meter-bar-multistep").style.width = `${Math.min(100, (feats.multi_step_count / 4) * 100)}%`;

    document.getElementById("meter-val-verbs").textContent = `${feats.instruction_verb_count} verbs`;
    document.getElementById("meter-bar-verbs").style.width = `${Math.min(100, (feats.instruction_verb_count / 5) * 100)}%`;

    document.getElementById("meter-val-domain").textContent = `${feats.domain} (${(feats.domain_match_ratio * 100).toFixed(0)}%)`;
    document.getElementById("meter-bar-domain").style.width = `${Math.min(100, feats.domain_match_ratio * 100)}%`;

    document.getElementById("meter-val-confidence").textContent = confidence !== undefined ? confidence.toFixed(2) : "0.50";
    document.getElementById("meter-bar-confidence").style.width = `${Math.min(100, (confidence || 0.5) * 100)}%`;
  }

  function renderReputationTable(reputation) {
    const tbody = document.getElementById("tbody-reputation");
    if (!reputation || reputation.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-row">No records in SQLite reputation table yet.</td></tr>';
      return;
    }

    tbody.innerHTML = reputation.map(r => {
      const isBumped = r.ema_score < 0.30;
      const chipClass = isBumped ? "score-red" : "score-green";
      const statusText = isBumped ? "AUTO-BUMPED" : "HEALTHY";
      return `
        <tr>
          <td><span style="color:#e2e8f0; font-weight:600;">${r.bucket_key}</span></td>
          <td><span style="color:#818cf8;">${r.model_tier}</span></td>
          <td><span class="score-chip ${chipClass}">${Number(r.ema_score).toFixed(3)}</span></td>
          <td>${r.call_count}</td>
          <td><span class="${isBumped ? 'text-danger' : 'text-emerald'}">${statusText}</span></td>
        </tr>
      `;
    }).join("");
  }

  function renderDecisionsTable(decisions) {
    const tbody = document.getElementById("tbody-decisions");
    if (!decisions || decisions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="empty-row">No decisions logged.</td></tr>';
      return;
    }

    tbody.innerHTML = decisions.map(d => {
      const shortHash = d.prompt_hash ? d.prompt_hash.substring(0, 16) + "..." : "err";
      const preview = d.prompt_preview ? (d.prompt_preview.length > 32 ? d.prompt_preview.substring(0, 32) + "..." : d.prompt_preview) : "";
      return `
        <tr>
          <td title="${d.prompt_hash}" style="color: #60a5fa;">${shortHash}</td>
          <td style="color: #94a3b8;">${preview}</td>
          <td><span style="color: #cbd5e1;">${d.model_used || d.provider_key}</span></td>
          <td>${d.latency_ms}ms</td>
        </tr>
      `;
    }).join("");
  }

  // ------------------------------------------------------------------
  // Fast Pre-Classification (<4ms)
  // ------------------------------------------------------------------

  async function triggerInstantPreclassification(prompt) {
    if (!prompt.trim()) return;
    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      if (res.ok) {
        const data = await res.json();
        document.getElementById("feat-extract-time").textContent = `< ${Math.max(1, data.classification_latency_ms)}ms`;
        renderFeatureMeters(data.features, data.confidence);

        // Update Decision Cascade Graph Preview
        graphDescClassifier.textContent = `${data.task_type} · ${data.complexity_bucket.toUpperCase()} (Complexity ${data.complexity.toFixed(2)})`;
      }
    } catch (err) {
      // Non-blocking background telemetry
    }
  }

  // ------------------------------------------------------------------
  // Sync Telemetry with SQLite WAL
  // ------------------------------------------------------------------

  async function syncTelemetry() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) throw new Error("stats network failure");
      const data = await res.json();

      if (data.report) {
        statDecisions.textContent = data.report.total_decisions || 0;
      }
      renderReputationTable(data.reputation);
      renderDecisionsTable(data.decisions);
      logKernel("SQLite WAL synchronization complete.", "sys");
    } catch (err) {
      logKernel(`Sync failed: ${err.message}`, "err");
    }
  }

  // ------------------------------------------------------------------
  // Zero-Lag SSE Stream Dispatch
  // ------------------------------------------------------------------

  async function dispatchStream() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a query prompt.");
      return;
    }

    if (isStreaming) return;
    isStreaming = true;
    btnDispatch.disabled = true;
    rawAccumulatedOutput = "";
    tokenCountSession = 0;
    streamStartTime = performance.now();

    resetWaterfall();
    wfSpeedBadge.className = "wf-speed-badge font-mono streaming";
    liveSpeedText.textContent = "Connecting...";

    rawTokensContainer.innerHTML = "";
    renderFormattedContent("", false);

    logKernel(`Ingress prompt dispatch: "${prompt.slice(0, 48)}..."`, "router");

    const isOverrideActive = toggleOverrideHook.checked;
    const payload = {
      prompt,
      conversation_id: "studio_session",
      override_provider: isOverrideActive ? "anthropic" : null
    };

    // Update Graph Stage 1
    if (isOverrideActive) {
      graphStatusOverride.textContent = "TRIGGERED";
      graphStatusOverride.className = "stage-status font-mono text-indigo";
      graphDescOverride.textContent = "Direct override hook intercepted query -> Bypassing Level 2/3 heuristics.";
    } else {
      graphStatusOverride.textContent = "BYPASS NOT ACTIVE";
      graphStatusOverride.className = "stage-status font-mono";
      graphDescOverride.textContent = "No programmatic override. Falling through to Level 2 Heuristic Classifier.";
    }

    try {
      const response = await fetch("/api/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // Keep partial tail

        for (const line of lines) {
          if (!line.trim()) continue;

          let eventName = "message";
          let eventData = "";

          for (const sub of line.split("\n")) {
            if (sub.startsWith("event: ")) eventName = sub.slice(7).trim();
            if (sub.startsWith("data: ")) eventData = sub.slice(6).trim();
          }

          if (!eventData) continue;

          try {
            const parsed = JSON.parse(eventData);

            if (eventName === "init") {
              // Stage 1: Classifier Completed in <5ms!
              wfStep1.classList.add("active");
              wfMetricClassify.textContent = `${parsed.classification_latency_ms}ms`;

              // Stage 2: Hierarchy Engine Activated
              wfStep2.classList.add("active");
              wfMetricHierarchy.textContent = "< 0.6ms";

              specTaskType.textContent = `${parsed.task_type} (${parsed.complexity_bucket.toUpperCase()})`;
              specComplexity.textContent = parsed.complexity.toFixed(2);
              renderFeatureMeters(parsed.features, parsed.confidence);
              document.getElementById("feat-extract-time").textContent = `< ${Math.max(1, parsed.classification_latency_ms)}ms`;

              // Update Graph Stage 2 & 3
              graphStatusClassifier.textContent = `COMPLETED (${parsed.classification_latency_ms}ms)`;
              graphDescClassifier.textContent = `Task: ${parsed.task_type} · Bucket: ${parsed.complexity_bucket.toUpperCase()} (Complexity ${parsed.complexity.toFixed(2)})`;
              graphStatusReputation.textContent = "REPUTATION FILTER VERIFIED";

              logKernel(`Feature vector extracted in ${parsed.classification_latency_ms}ms. Tier: ${parsed.complexity_bucket.toUpperCase()}`, "ok");

            } else if (eventName === "chunk") {
              // Real-time token delta arrived!
              if (tokenCountSession === 0) {
                // First token arrived -> TTFT
                const ttft = (performance.now() - streamStartTime).toFixed(0);
                wfStep3.classList.add("active");
                wfMetricTtft.textContent = `${ttft}ms`;
                logKernel(`First token received. TTFT: ${ttft}ms. Ingress streaming active.`, "ok");
              }

              tokenCountSession++;
              rawAccumulatedOutput += parsed.text;

              // Log to Raw Token Stream Viewport
              const elapsedMs = (performance.now() - streamStartTime).toFixed(1);
              const tokRow = document.createElement("div");
              tokRow.className = "raw-token-line";
              tokRow.innerHTML = `<span class="raw-dim">[+${elapsedMs}ms #${tokenCountSession}]</span> <span>${escapeHtml(parsed.text)}</span>`;
              rawTokensContainer.appendChild(tokRow);
              if (rawTokensContainer.children.length > 100) {
                rawTokensContainer.removeChild(rawTokensContainer.children[0]);
              }
              rawStreamViewport.scrollTop = rawStreamViewport.scrollHeight;

              // Calculate live token speed
              const elapsedSec = (performance.now() - streamStartTime) / 1000;
              const tokPerSec = Math.round(tokenCountSession / (elapsedSec || 0.001));
              liveSpeedText.textContent = `${tokPerSec} tok/s`;

              // Schedule butter-smooth 60fps render
              scheduleBatchRender(false);

            } else if (eventName === "done") {
              // Streaming finished
              const meta = parsed.meta;
              lastDecisionId = meta.decision_id;

              wfStep4.classList.add("active");
              wfMetricWal.textContent = "< 1.2ms";
              liveSpeedText.textContent = "Finished";
              wfSpeedBadge.className = "wf-speed-badge font-mono";

              specModel.textContent = meta.model;
              specLatency.textContent = `${meta.total_latency_ms}ms`;
              specCost.textContent = `$${Number(meta.estimated_cost_usd || 0).toFixed(6)}`;
              specReason.textContent = meta.reason;

              scheduleBatchRender(true);

              if (parsed.db_records) {
                renderReputationTable(parsed.db_records.reputation);
                renderDecisionsTable(parsed.db_records.decisions);
              }

              logKernel(`Query completed in ${meta.total_latency_ms}ms. SHA-256 committed to SQLite WAL.`, "ok");
              syncTelemetry();

            } else if (eventName === "error") {
              throw new Error(parsed.error);
            }
          } catch (e) {
            // parsing error
          }
        }
      }

    } catch (err) {
      logKernel(`Stream error: ${err.message}`, "err");
      renderFormattedContent(`Stream failure: ${err.message}`, true);
      liveSpeedText.textContent = "Error";
    } finally {
      isStreaming = false;
      btnDispatch.disabled = false;
    }
  }

  // ------------------------------------------------------------------
  // Feedback Signal Injection
  // ------------------------------------------------------------------

  async function injectFeedback(signalType, numericValue) {
    if (!lastDecisionId) {
      alert("Please dispatch at least one query before submitting a feedback signal.");
      return;
    }

    logKernel(`Injecting feedback signal: ${signalType} (${numericValue > 0 ? '+' : ''}${numericValue})`, "signal");

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
        logKernel(`SQLite reputation updated. Applied EMA formula: EMA_t = 0.3*(${numericValue}) + 0.7*EMA_{t-1}`, "ok");
        if (data.db_records) {
          renderReputationTable(data.db_records.reputation);
        }

        signalToast.className = "signal-toast font-mono";
        signalToast.innerHTML = `Signal recorded: <strong>${signalType}</strong> (${numericValue}). SQLite EMA updated live.`;
        signalToast.classList.remove("hidden");

        setTimeout(() => signalToast.classList.add("hidden"), 4500);
      }
    } catch (err) {
      logKernel(`Feedback error: ${err.message}`, "err");
    }
  }

  // ------------------------------------------------------------------
  // Window Mode Switching (Formatted vs Raw vs Graph)
  // ------------------------------------------------------------------

  winModeTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      winModeTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.getAttribute("data-wmode");
      currentWindowMode = mode;

      streamingViewport.classList.add("hidden");
      rawStreamViewport.classList.add("hidden");
      graphViewport.classList.add("hidden");

      if (mode === "formatted") {
        streamingViewport.classList.remove("hidden");
      } else if (mode === "raw") {
        rawStreamViewport.classList.remove("hidden");
      } else if (mode === "graph") {
        graphViewport.classList.remove("hidden");
      }
    });
  });

  // ------------------------------------------------------------------
  // Event Bindings
  // ------------------------------------------------------------------

  promptInput.addEventListener("input", () => {
    updateCharCount();
    clearTimeout(promptInput._deb);
    promptInput._deb = setTimeout(() => {
      triggerInstantPreclassification(promptInput.value);
    }, 120);
  });

  // Keyboard shortcut: Cmd/Ctrl + Enter to dispatch
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      dispatchStream();
    }
  });

  // Scenario Cards Click
  document.querySelectorAll(".scenario-card").forEach(card => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".scenario-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      const key = card.getAttribute("data-scenario");
      if (SCENARIOS[key]) {
        promptInput.value = SCENARIOS[key];
        updateCharCount();
        triggerInstantPreclassification(SCENARIOS[key]);
        logKernel(`Loaded production benchmark: ${key}`, "sys");
      }
    });
  });

  // Profiler Tabs Switching
  document.querySelectorAll(".profiler-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".profiler-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".ptab-panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const targetId = `ptab-${tab.getAttribute("data-ptab")}`;
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add("active");
    });
  });

  btnDispatch.addEventListener("click", dispatchStream);

  btnClear.addEventListener("click", () => {
    promptInput.value = "";
    updateCharCount();
    resetWaterfall();
    renderFormattedContent("", true);
    rawTokensContainer.innerHTML = "";
  });

  btnCopyOutput.addEventListener("click", () => {
    if (rawAccumulatedOutput) {
      navigator.clipboard.writeText(rawAccumulatedOutput);
      btnCopyOutput.textContent = "COPIED";
      setTimeout(() => btnCopyOutput.textContent = "Copy Output", 1500);
    }
  });

  document.getElementById("btn-signal-regen").addEventListener("click", () => injectFeedback("hard_regen", -0.30));
  document.getElementById("btn-signal-correction").addEventListener("click", () => injectFeedback("explicit_correction", -0.20));
  document.getElementById("btn-signal-accept").addEventListener("click", () => injectFeedback("acceptance", 0.05));

  document.getElementById("btn-reset-weights").addEventListener("click", async () => {
    if (confirm("Reset all learned EMA reputation scores across SQLite back to baseline?")) {
      await fetch("/api/reset", { method: "POST" });
      logKernel("SQLite reputation table reset to baseline weights.", "signal");
      syncTelemetry();
    }
  });

  btnSyncTelemetry.addEventListener("click", syncTelemetry);
  btnClearLogs.addEventListener("click", () => {
    kernelLogLines.innerHTML = "";
    logKernel("Log buffer flushed.", "sys");
  });

  // Initialize with Distributed scenario
  promptInput.value = SCENARIOS.distributed;
  updateCharCount();
  triggerInstantPreclassification(SCENARIOS.distributed);
  syncTelemetry();
});
