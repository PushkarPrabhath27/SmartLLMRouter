/**
 * SmartRoute — Clean, High-Visual Interactive Frontend
 * Dead-simple to understand, stunning visuals, zero lag.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Inputs & Presets
  const promptInput = document.getElementById("prompt-input");
  const charCounter = document.getElementById("char-counter");
  const instantPreview = document.getElementById("instant-preview");
  const btnDispatch = document.getElementById("btn-dispatch");
  const presetPills = document.querySelectorAll(".preset-pill");

  // DOM Elements - Visual Pipeline Diagram
  const pipePromptType = document.getElementById("pipe-prompt-type");
  const pipePulse = document.getElementById("pipe-pulse");
  const pipeSpeed = document.getElementById("pipe-speed");
  const tierFast = document.getElementById("tier-fast");
  const tierCode = document.getElementById("tier-code");
  const tierHeavy = document.getElementById("tier-heavy");

  // DOM Elements - Decision Banner & Output
  const decisionPill = document.getElementById("decision-pill");
  const decisionTierName = document.getElementById("decision-tier-name");
  const decisionReason = document.getElementById("decision-reason");
  const resOverhead = document.getElementById("res-overhead");
  const resModel = document.getElementById("res-model");
  const resCost = document.getElementById("res-cost");
  const outputViewport = document.getElementById("output-viewport");

  // DOM Elements - Feedback & Copy
  const btnGood = document.getElementById("btn-good");
  const btnBad = document.getElementById("btn-bad");
  const feedbackToast = document.getElementById("feedback-toast");
  const btnCopy = document.getElementById("btn-copy");

  // DOM Elements - Technical Drawer
  const btnTechToggle = document.getElementById("btn-tech-toggle");
  const techDrawer = document.getElementById("tech-drawer");
  const toggleArrow = document.getElementById("toggle-arrow");
  const techFeatCode = document.getElementById("tech-feat-code");
  const techFeatReasoning = document.getElementById("tech-feat-reasoning");
  const techFeatDomain = document.getElementById("tech-feat-domain");
  const techFeatComplexity = document.getElementById("tech-feat-complexity");
  const techShaDisplay = document.getElementById("tech-sha-display");
  const hdrRoutingTime = document.getElementById("hdr-routing-time");

  // State
  let lastDecisionId = null;
  let isStreaming = false;
  let rawAccumulatedText = "";
  let renderPending = false;

  // Simple, recognizable presets
  const PRESETS = {
    simple: {
      text: "What is the capital of France and what is it famous for?",
      tier: "fast",
      label: "Simple Question",
      reason: "Everyday general inquiry with low complexity. Routed to Fast Tier to answer instantly with 95% cost savings."
    },
    code: {
      text: "Write a clean Python function to check whether a given string is a palindrome. Include docstring and test cases.",
      tier: "code",
      label: "Programming Task",
      reason: "Contains Python coding keywords and syntax requirements. Routed to Code Tier for high technical accuracy."
    },
    reasoning: {
      text: "Compare B-Tree vs LSM-Tree storage engines for write-heavy database workloads. Explain write amplification, memory footprint, and compactions.",
      tier: "heavy",
      label: "Complex Architecture",
      reason: "High complexity multi-step reasoning task with deep domain concepts. Routed to Reasoning Tier for comprehensive analysis."
    }
  };

  // ------------------------------------------------------------------
  // Visual Pipeline Animation & Highlighting
  // ------------------------------------------------------------------

  function highlightPipelineModel(tier) {
    [tierFast, tierCode, tierHeavy].forEach(c => {
      c.classList.remove("active", "active-code", "active-heavy");
    });

    if (tier === "fast" || tier === "low") {
      tierFast.classList.add("active");
      pipePromptType.textContent = "Simple Task";
      instantPreview.innerHTML = 'Estimated: <strong class="text-emerald">Fast Tier (Llama 8B) · Saves ~95%</strong>';
    } else if (tier === "code" || tier === "medium") {
      tierCode.classList.add("active-code");
      pipePromptType.textContent = "Coding Task";
      instantPreview.innerHTML = 'Estimated: <strong style="color: var(--color-purple);">Code Tier (Llama 70B) · High Accuracy</strong>';
    } else {
      tierHeavy.classList.add("active-heavy");
      pipePromptType.textContent = "Complex Reasoning";
      instantPreview.innerHTML = 'Estimated: <strong style="color: var(--color-blue);">Reasoning Tier · Deep Logic</strong>';
    }
  }

  // ------------------------------------------------------------------
  // Fast Local/API Pre-Classification
  // ------------------------------------------------------------------

  async function updatePreview(text) {
    const chars = text.length;
    charCounter.textContent = `${chars} characters`;

    if (!text.trim()) {
      [tierFast, tierCode, tierHeavy].forEach(c => c.classList.remove("active", "active-code", "active-heavy"));
      pipePromptType.textContent = "Waiting for input...";
      instantPreview.textContent = "Type prompt to preview route";
      pipePulse.classList.remove("pulsing");
      return;
    }

    pipePulse.classList.add("pulsing");

    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text })
      });
      if (res.ok) {
        const data = await res.json();
        const latency = Math.max(1, Math.round(data.classification_latency_ms));
        pipeSpeed.textContent = `< ${latency}ms`;
        hdrRoutingTime.textContent = `< ${latency}ms`;

        let tierKey = "fast";
        if (data.task_type === "code" || data.complexity_bucket === "high") {
          tierKey = "code";
        } else if (data.task_type === "reasoning" || data.complexity > 0.6) {
          tierKey = "heavy";
        } else if (data.complexity > 0.35) {
          tierKey = "code";
        }
        highlightPipelineModel(tierKey);

        // Update Technical Drawer values
        if (data.features) {
          techFeatCode.textContent = data.features.code_block_ratio.toFixed(2);
          techFeatReasoning.textContent = `${data.features.multi_step_count} markers`;
          techFeatDomain.textContent = data.features.domain;
          techFeatComplexity.textContent = `${data.complexity.toFixed(2)} / 1.00`;
        }
      }
    } catch {
      // Local fallback heuristic if server not ready
      if (/\b(def|class|function|python|code|bug|sql)\b/i.test(text)) {
        highlightPipelineModel("code");
      } else if (text.length > 200 || /\b(compare|analyze|why|architecture)\b/i.test(text)) {
        highlightPipelineModel("heavy");
      } else {
        highlightPipelineModel("fast");
      }
    }
  }

  // ------------------------------------------------------------------
  // High-Quality Code & Markdown Formatter
  // ------------------------------------------------------------------

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function highlightPython(code) {
    const esc = escapeHtml(code);
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

  function renderCleanOutput(text, isFinished = true) {
    if (!text) {
      outputViewport.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">💡</div>
          <div class="empty-title">Ready to Route</div>
          <div class="empty-desc">Click "Route & Run" above to see SmartRoute analyze the prompt in 2ms and stream the answer from the chosen model.</div>
        </div>`;
      return;
    }

    // Strip emoji bullets and conversational greeting if present
    let clean = text
      .replace(/#\s*[1-9]️⃣/g, (m) => `# Step ${m.replace(/\D/g, "")}:`)
      .replace(/[1-9]️⃣\s*/g, (m) => `${m.replace(/\D/g, "")}. `)
      .replace(/^(Sure!?|Certainly!?|Here is the|Here's a|Below is).+?\n\n/i, "");

    let html = "";
    if (clean.includes("```")) {
      const regex = /```([a-zA-Z0-9_-]*)?\n?([\s\S]*?)($|```)/g;
      let lastIndex = 0;
      let match;

      while ((match = regex.exec(clean)) !== null) {
        const plain = clean.slice(lastIndex, match.index).trim();
        if (plain) {
          html += formatPlainMarkdown(plain);
        }

        const lang = (match[1] || "python").toUpperCase();
        const code = match[2] || "";
        const highlighted = highlightPython(code);

        html += `
          <div class="code-block-wrapper">
            <div class="code-header-bar font-mono">
              <span>${lang}</span>
              <button class="code-copy-btn font-mono" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.textContent='COPIED'; setTimeout(()=>this.textContent='COPY', 1200);">COPY</button>
            </div>
            <pre class="code-inner-body font-mono"><code>${highlighted}</code></pre>
          </div>
        `;
        lastIndex = regex.lastIndex;
      }
      const rest = clean.slice(lastIndex).trim();
      if (rest) {
        html += formatPlainMarkdown(rest);
      }
    } else if (/\b(def |class |import |from |return )/.test(clean)) {
      const highlighted = highlightPython(clean);
      html += `
        <div class="code-block-wrapper">
          <div class="code-header-bar font-mono">
            <span>PYTHON</span>
            <button class="code-copy-btn font-mono" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.textContent='COPIED'; setTimeout(()=>this.textContent='COPY', 1200);">COPY</button>
          </div>
          <pre class="code-inner-body font-mono"><code>${highlighted}</code></pre>
        </div>
      `;
    } else {
      html += formatPlainMarkdown(clean);
    }

    if (!isFinished) {
      html += `<span class="streaming-cursor"></span>`;
    }

    outputViewport.innerHTML = html;
    outputViewport.scrollTop = outputViewport.scrollHeight;
  }

  function formatPlainMarkdown(raw) {
    let p = escapeHtml(raw);
    p = p.replace(/^### (.*$)/gm, '<h4 style="font-size: 14px; font-weight: 700; color: #f8fafc; margin: 12px 0 6px;">$1</h4>');
    p = p.replace(/^## (.*$)/gm, '<h3 style="font-size: 15px; font-weight: 700; color: #ffffff; margin: 16px 0 8px;">$1</h3>');
    p = p.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #ffffff; font-weight: 600;">$1</strong>');
    p = p.replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.08); color: #818cf8; padding: 1px 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 12px;">$1</code>');
    p = p.replace(/^\s*[-*]\s+(.*$)/gm, '<li style="margin-left: 20px; margin-bottom: 4px; color: #cbd5e1;">$1</li>');
    p = p.replace(/\n\n/g, '<div style="height: 12px;"></div>');
    p = p.replace(/\n/g, '<br>');
    return `<div style="margin: 8px 0; line-height: 1.65;">${p}</div>`;
  }

  function scheduleBatchRender(isFinished = false) {
    if (renderPending && !isFinished) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderCleanOutput(rawAccumulatedText, isFinished);
      renderPending = false;
    });
  }

  // ------------------------------------------------------------------
  // Stream Dispatcher
  // ------------------------------------------------------------------

  async function runQuery() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a question or click an example above.");
      return;
    }

    if (isStreaming) return;
    isStreaming = true;
    btnDispatch.disabled = true;
    btnDispatch.innerHTML = `<span>Routing...</span>`;
    rawAccumulatedText = "";

    decisionPill.className = "decision-status-pill";
    decisionTierName.textContent = "Analyzing in <3ms...";
    decisionReason.textContent = "SmartRoute is calculating prompt complexity and checking model reputations...";
    resOverhead.textContent = "...";
    resModel.textContent = "...";
    resCost.textContent = "...";

    renderCleanOutput("Connecting to model stream...", false);

    try {
      const response = await fetch("/api/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, conversation_id: "demo_user" })
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
        buffer = lines.pop();

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
            const data = JSON.parse(eventData);

            if (eventName === "init") {
              const latency = Math.max(1, Math.round(data.classification_latency_ms));
              resOverhead.textContent = `${latency}ms`;
              pipeSpeed.textContent = `${latency}ms`;

              // Determine tier styling
              let tierKey = "fast";
              let pillClass = "routed-fast";
              let tierDisplay = "🟢 Fast Tier (Llama 8B)";

              if (data.task_type === "code" || data.complexity_bucket === "high") {
                tierKey = "code";
                pillClass = "routed-code";
                tierDisplay = "🟣 Code Tier (Llama 70B)";
              } else if (data.task_type === "reasoning" || data.complexity > 0.6) {
                tierKey = "heavy";
                pillClass = "routed-heavy";
                tierDisplay = "🔵 Reasoning Tier (GPT-4o)";
              }

              highlightPipelineModel(tierKey);
              decisionPill.className = `decision-status-pill ${pillClass}`;
              decisionTierName.textContent = tierDisplay;

              decisionReason.innerHTML = `<strong>Selected in ${latency}ms:</strong> Detected <em>${data.task_type}</em> task (complexity ${data.complexity.toFixed(2)}). Evaluated locally without costly API overhead.`;
              renderCleanOutput("", false);

            } else if (eventName === "chunk") {
              rawAccumulatedText += data.text;
              scheduleBatchRender(false);

            } else if (eventName === "done") {
              const meta = data.meta;
              lastDecisionId = meta.decision_id;

              resModel.textContent = meta.model ? meta.model.replace("openai/", "").replace("groq/", "") : "Llama-3";
              resCost.textContent = `$${Number(meta.estimated_cost_usd || 0.00002).toFixed(5)}`;
              resOverhead.textContent = `${meta.classification_latency_ms || 2.1}ms`;

              scheduleBatchRender(true);

              // Update SHA-256 privacy audit
              if (meta.decision_id) {
                techShaDisplay.textContent = `Decision ID: ${meta.decision_id} (Zero prompt text stored to disk)`;
              }
            }
          } catch {
            // parse error
          }
        }
      }

    } catch (err) {
      renderCleanOutput(`Error during routing: ${err.message}`, true);
      decisionReason.textContent = "Error communicating with server.";
    } finally {
      isStreaming = false;
      btnDispatch.disabled = false;
      btnDispatch.innerHTML = `
        <span>Route & Run</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      `;
    }
  }

  // ------------------------------------------------------------------
  // Feedback Buttons (Learning Demonstration)
  // ------------------------------------------------------------------

  async function submitFeedback(signalType, label) {
    if (!lastDecisionId) {
      alert("Please run at least one query before submitting feedback.");
      return;
    }

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision_id: lastDecisionId, signal_type: signalType })
      });
      if (res.ok) {
        feedbackToast.textContent = `Feedback recorded (${label})! Model reputation updated in SQLite WAL.`;
        feedbackToast.classList.remove("hidden");
        setTimeout(() => feedbackToast.classList.add("hidden"), 4000);
      }
    } catch {
      // ignore
    }
  }

  // ------------------------------------------------------------------
  // Event Listeners
  // ------------------------------------------------------------------

  promptInput.addEventListener("input", () => {
    clearTimeout(promptInput._deb);
    promptInput._deb = setTimeout(() => {
      updatePreview(promptInput.value);
    }, 150);
  });

  promptInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runQuery();
    }
  });

  presetPills.forEach(pill => {
    pill.addEventListener("click", () => {
      presetPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");

      const key = pill.getAttribute("data-preset");
      const preset = PRESETS[key];
      if (preset) {
        promptInput.value = preset.text;
        updatePreview(preset.text);
      }
    });
  });

  btnDispatch.addEventListener("click", runQuery);

  btnGood.addEventListener("click", () => submitFeedback("acceptance", "👍 Helpful"));
  btnBad.addEventListener("click", () => submitFeedback("explicit_correction", "👎 Needs better tier"));

  btnCopy.addEventListener("click", () => {
    if (rawAccumulatedText) {
      navigator.clipboard.writeText(rawAccumulatedText);
      btnCopy.textContent = "Copied!";
      setTimeout(() => btnCopy.textContent = "Copy Answer", 1500);
    }
  });

  btnTechToggle.addEventListener("click", () => {
    const isHidden = techDrawer.classList.contains("hidden");
    if (isHidden) {
      techDrawer.classList.remove("hidden");
      toggleArrow.classList.add("open");
    } else {
      techDrawer.classList.add("hidden");
      toggleArrow.classList.remove("open");
    }
  });

  // Initialize with Simple preset
  const defaultPreset = PRESETS.simple;
  promptInput.value = defaultPreset.text;
  updatePreview(defaultPreset.text);
});
