/**
 * Responsive Agent Platform — Frontend Application Script
 * Orchestrates Chat SSE streaming, Ingestion, MCP registry, Agent config, and Traces.
 */

// ── State ────────────────────────────────────────────────────────────────────
let currentSessionId = "sess_" + Math.random().toString(36).substring(2, 9);
let currentEvidenceMap = {}; // Maps source citations to evidence objects
let pollInterval = null;

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTenantAndSession();
  initTabs();
  initChat();
  initIngestion();
  initMcp();
  initAgentConfig();
  initTraces();
});

function getTenantId() {
  const input = document.getElementById("tenantInput");
  return (input && input.value.trim()) || "default";
}

function initTenantAndSession() {
  document.getElementById("sidebarSessionId").innerText = currentSessionId;
  document.getElementById("activeSessionDisplay").innerText = `Session: ${currentSessionId}`;

  document.getElementById("btnNewSession").addEventListener("click", () => {
    currentSessionId = "sess_" + Math.random().toString(36).substring(2, 9);
    document.getElementById("sidebarSessionId").innerText = currentSessionId;
    document.getElementById("activeSessionDisplay").innerText = `Session: ${currentSessionId}`;
    appendSystemNotice(`Started new session: ${currentSessionId}`);
  });

  document.getElementById("btnClearHistory").addEventListener("click", () => {
    const chatHistory = document.getElementById("chatHistory");
    chatHistory.innerHTML = "";
    appendSystemNotice("Chat history cleared for current display.");
  });
}

// ── Tab Switching ────────────────────────────────────────────────────────────
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      // Auto-fetch data for active tab
      if (targetId === "tab-ingest") loadSources();
      else if (targetId === "tab-mcp") loadMcpServers();
      else if (targetId === "tab-config") loadAgentConfig();
      else if (targetId === "tab-traces") loadTraces();
    });
  });
}

// ── Chat & Real-Time SSE Streaming ───────────────────────────────────────────
function getShowSourcesFlag() {
  const val = localStorage.getItem("show_evidence_sources");
  return val === "true"; // Default is false (OFF)
}

function setShowSourcesFlag(enabled) {
  localStorage.setItem("show_evidence_sources", enabled ? "true" : "false");
  const cbHeader = document.getElementById("toggleShowSources");
  const cbSidebar = document.getElementById("toggleShowSourcesSidebar");
  if (cbHeader) cbHeader.checked = enabled;
  if (cbSidebar) cbSidebar.checked = enabled;

  // Dynamically toggle visibility on all sources panels in chat history
  document.querySelectorAll(".sources-panel").forEach((panel) => {
    if (enabled) {
      panel.classList.remove("sources-hidden");
    } else {
      panel.classList.add("sources-hidden");
    }
  });

  // Update per-message toggle buttons text
  document.querySelectorAll(".sources-toggle-btn").forEach((btn) => {
    const count = btn.getAttribute("data-count") || "";
    if (enabled) {
      btn.innerHTML = `📚 Hide Sources`;
    } else {
      btn.innerHTML = `📚 Show Sources (${count})`;
    }
  });
}

window.toggleSingleMessageSources = function(btnElem) {
  const card = btnElem.closest(".message-card") || btnElem.parentElement.parentElement;
  if (!card) return;
  const panel = card.querySelector(".sources-panel");
  if (!panel) return;

  const isHidden = panel.classList.contains("sources-hidden");
  const count = btnElem.getAttribute("data-count") || "";
  if (isHidden) {
    panel.classList.remove("sources-hidden");
    btnElem.innerHTML = `📚 Hide Sources`;
  } else {
    panel.classList.add("sources-hidden");
    btnElem.innerHTML = `📚 Show Sources (${count})`;
  }
};

function initChat() {
  const chatInput = document.getElementById("chatInput");
  const btnSend = document.getElementById("btnSendChat");

  btnSend.addEventListener("click", handleSendChat);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendChat();
    }
  });

  // Wire up Show Sources toggle (Default: OFF)
  const isSourcesEnabled = getShowSourcesFlag();
  const cbHeader = document.getElementById("toggleShowSources");
  const cbSidebar = document.getElementById("toggleShowSourcesSidebar");

  if (cbHeader) {
    cbHeader.checked = isSourcesEnabled;
    cbHeader.addEventListener("change", (e) => setShowSourcesFlag(e.target.checked));
  }
  if (cbSidebar) {
    cbSidebar.checked = isSourcesEnabled;
    cbSidebar.addEventListener("change", (e) => setShowSourcesFlag(e.target.checked));
  }
}

async function handleSendChat() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  appendUserMessage(message);

  const breadcrumbs = document.getElementById("liveBreadcrumbs");
  const breadcrumbText = document.getElementById("breadcrumbText");
  breadcrumbs.style.display = "block";
  breadcrumbText.innerText = "Analysing your query...";

  // Prepare streaming assistant message placeholder
  const assistantCard = createAssistantCard();
  const bodyElem = assistantCard.querySelector(".message-body");
  const senderBadge = assistantCard.querySelector(".badge");

  let fullAnswerText = "";
  let receivedDone = false;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": getTenantId(),
      },
      body: JSON.stringify({
        message: message,
        session_id: currentSessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}: ${await response.text()}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // Keep incomplete frame

      for (const block of lines) {
        if (!block.trim()) continue;
        const eventMatch = block.match(/^event:\s*(\w+)/m);
        const dataMatch = block.match(/^data:\s*(.*)$/m);

        const eventType = eventMatch ? eventMatch[1] : "message";
        const rawData = dataMatch ? dataMatch[1] : "";

        let parsedData = {};
        try {
          parsedData = JSON.parse(rawData);
        } catch {
          parsedData = { text: rawData };
        }

        if (eventType === "status") {
          breadcrumbText.innerText = parsedData.message || parsedData.stage || parsedData.text || "Thinking...";
        } else if (eventType === "answer") {
          const chunk = parsedData.answer || parsedData.chunk || parsedData.text || "";
          if (chunk) {
            fullAnswerText = chunk;
            bodyElem.innerHTML = renderMarkdown(fullAnswerText);
          }
        } else if (eventType === "done") {
          receivedDone = true;
          breadcrumbs.style.display = "none";
          if (parsedData.error) {
            senderBadge.innerText = "⚠️ Service Notice";
            senderBadge.className = "badge badge-pending";
            const cleanErr = formatAppError(parsedData.error);
            bodyElem.innerHTML = `
              <div class="error-notice" style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-left: 3px solid #f59e0b; padding: 0.75rem 1rem; border-radius: 6px; margin-top: 0.5rem;">
                <div style="font-weight: 600; color: #fbbf24; margin-bottom: 0.25rem;">Service Notice</div>
                <div style="color: #e5e7eb; font-size: 0.95rem; line-height: 1.5;">${escapeHtml(cleanErr)}</div>
              </div>`;
          } else {
            senderBadge.innerText = parsedData.grounding_passed ? "🛡️ Grounded & Verified" : "⚠️ Synthesis Complete";
            senderBadge.className = parsedData.grounding_passed ? "badge badge-ready" : "badge badge-pending";

            // If answer was supplied in done payload
            if (parsedData.final_answer && !fullAnswerText) {
              fullAnswerText = parsedData.final_answer;
            }

            // Process sources and inline citations
            const sources = parsedData.sources || [];
            renderAssistantDone(bodyElem, fullAnswerText, sources, parsedData.total_latency_ms);
          }
        } else if (eventType === "error") {
          const raw = parsedData.message || parsedData.detail || "Request could not be completed";
          const cleanErr = formatAppError(raw);
          bodyElem.innerHTML = `
            <div class="error-notice" style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-left: 3px solid #f59e0b; padding: 0.75rem 1rem; border-radius: 6px; margin-top: 0.5rem;">
              <div style="font-weight: 600; color: #fbbf24; margin-bottom: 0.25rem;">Service Notice</div>
              <div style="color: #e5e7eb; font-size: 0.95rem; line-height: 1.5;">${escapeHtml(cleanErr)}</div>
            </div>`;
          senderBadge.innerText = "⚠️ Service Notice";
          senderBadge.className = "badge badge-pending";
          breadcrumbs.style.display = "none";
        }
      }
    }

    if (!receivedDone) {
      breadcrumbs.style.display = "none";
      if (!fullAnswerText) {
        bodyElem.innerHTML = "<em>(Completed with empty response)</em>";
      }
    }
  } catch (err) {
    breadcrumbs.style.display = "none";
    const cleanErr = formatAppError(err.message);
    bodyElem.innerHTML = `
      <div class="error-notice" style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-left: 3px solid #f59e0b; padding: 0.75rem 1rem; border-radius: 6px; margin-top: 0.5rem;">
        <div style="font-weight: 600; color: #fbbf24; margin-bottom: 0.25rem;">Service Notice</div>
        <div style="color: #e5e7eb; font-size: 0.95rem; line-height: 1.5;">${escapeHtml(cleanErr)}</div>
      </div>`;
    senderBadge.innerText = "⚠️ Service Notice";
    senderBadge.className = "badge badge-pending";
  }
}

function appendUserMessage(text) {
  const history = document.getElementById("chatHistory");
  const div = document.createElement("div");
  div.className = "message-card message-user";
  div.innerHTML = `
    <div class="message-sender">You</div>
    <div class="message-body">${escapeHtml(text)}</div>
  `;
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
}

function createAssistantCard() {
  const history = document.getElementById("chatHistory");
  const div = document.createElement("div");
  div.className = "message-card message-assistant";
  div.innerHTML = `
    <div class="message-sender">
      <span>AI Research Agent</span>
      <span class="badge badge-processing">Processing...</span>
    </div>
    <div class="message-body"><span style="color: var(--text-muted);">Generating response...</span></div>
  `;
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
  return div;
}

function appendSystemNotice(text) {
  const history = document.getElementById("chatHistory");
  const div = document.createElement("div");
  div.style.textAlign = "center";
  div.style.fontSize = "0.75rem";
  div.style.color = "var(--text-muted)";
  div.style.padding = "0.4rem";
  div.innerText = text;
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
}

function renderAssistantDone(bodyElem, text, sources, latencyMs) {
  if (!text || !text.trim()) {
    text = "<em>I could not synthesize an answer from the retrieved evidence. Please review the sources below.</em>";
  }

  // Store evidence objects in currentEvidenceMap
  sources.forEach((src, idx) => {
    const key = `ref_${idx + 1}`;
    currentEvidenceMap[key] = src;
  });

  // Convert citations like [1] or [terms.pdf, p.2] into clickable badges
  let formatted = renderMarkdown(text);
  formatted = formatted.replace(/\[(\d+)\]/g, (match, num) => {
    const key = `ref_${num}`;
    return `<span class="citation-badge" onclick="inspectCitation('${key}')">[${num}]</span>`;
  });

  const showSources = getShowSourcesFlag();

  // Sources footer
  let sourcesHtml = "";
  if (sources && sources.length > 0) {
    sourcesHtml = `
      <div class="sources-panel ${showSources ? "" : "sources-hidden"}">
        <div class="sources-panel-title">📚 Retrieved Evidence Sources (${sources.length})</div>
        ${sources.map((s, idx) => `
          <div class="source-item" onclick="inspectCitation('ref_${idx + 1}')">
            <span><strong>[${idx + 1}]</strong> ${escapeHtml(s.source_name || "Evidence")} ${s.page_number ? `(p. ${s.page_number})` : ""}</span>
            <span style="font-size: 0.75rem; color: var(--text-secondary);">${s.rerank_score ? `Score: ${s.rerank_score.toFixed(3)}` : "Verified"}</span>
          </div>
        `).join("")}
      </div>
    `;
  }

  const sourcesTogglePill = (sources && sources.length > 0) ? `
    <span class="sources-toggle-btn" data-count="${sources.length}" onclick="toggleSingleMessageSources(this)">
      ${showSources ? "📚 Hide Sources" : `📚 Show Sources (${sources.length})`}
    </span>
  ` : "";

  const metaHtml = `
    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.6rem; display: flex; justify-content: space-between; align-items: center;">
      <div>${sourcesTogglePill}</div>
      <div>${latencyMs ? `Latency: ${latencyMs} ms &bull; Deterministic Reranked` : ""}</div>
    </div>
  `;

  bodyElem.innerHTML = formatted + sourcesHtml + metaHtml;
}

// ── Citation Provenance Inspector ────────────────────────────────────────────
window.inspectCitation = function(refKey) {
  const inspector = document.getElementById("citationInspector");
  const evidence = currentEvidenceMap[refKey];
  if (!evidence) {
    inspector.innerHTML = `<p style="color: var(--warning);">Evidence details not available for ${refKey}.</p>`;
    return;
  }

  inspector.innerHTML = `
    <div style="background: var(--bg-elevated); padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border-color);">
      <h4 style="color: var(--text-primary); margin-bottom: 0.4rem;">${escapeHtml(evidence.source_name || "Evidence Item")}</h4>
      <p style="margin-bottom: 0.3rem;"><strong>Type / URI:</strong> <span style="word-break: break-all;">${escapeHtml(evidence.source_uri || "local://knowledge-base")}</span></p>
      ${evidence.page_number ? `<p style="margin-bottom: 0.3rem;"><strong>Page Number:</strong> ${evidence.page_number}</p>` : ""}
      ${evidence.section_heading ? `<p style="margin-bottom: 0.3rem;"><strong>Section:</strong> ${escapeHtml(evidence.section_heading)}</p>` : ""}
      ${evidence.rerank_score ? `<p style="margin-bottom: 0.3rem;"><strong>Cross-Encoder Relevance:</strong> ${(evidence.rerank_score * 100).toFixed(1)}%</p>` : ""}
      <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--border-color);">
        <strong style="color: var(--text-muted);">Retrieved Excerpt:</strong>
        <p style="margin-top: 0.25rem; font-family: var(--font-mono); font-size: 0.75rem; background: var(--bg-primary); padding: 0.5rem; border-radius: 4px; max-height: 140px; overflow-y: auto;">
          ${escapeHtml(evidence.text || "No preview snippet available.")}
        </p>
      </div>
    </div>
  `;
};

// ── Ingestion (Files & Web) ──────────────────────────────────────────────────
function initIngestion() {
  const dropzone = document.getElementById("fileDropzone");
  const fileInput = document.getElementById("fileInput");
  const btnRefresh = document.getElementById("btnRefreshSources");
  const urlForm = document.getElementById("urlIngestForm");

  // Drag-and-drop
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => { dropzone.classList.remove("dragover"); });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) uploadFiles(e.target.files);
  });

  // URL form
  urlForm.addEventListener("submit", handleRegisterUrl);
  btnRefresh.addEventListener("click", loadSources);
}

async function uploadFiles(files) {
  const feedback = document.getElementById("uploadFeedback");
  feedback.innerHTML = `<span style="color: var(--accent);">Uploading ${files.length} file(s)...</span>`;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  try {
    const res = await fetch(`/api/ingest/files?tenant_id=${encodeURIComponent(getTenantId())}`, {
      method: "POST",
      headers: { "X-Tenant-ID": getTenantId() },
      body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const count = data.queued_count ?? (Array.isArray(data.files) ? data.files.length : (Array.isArray(data) ? data.length : files.length));
    feedback.innerHTML = `<span style="color: var(--success);">Uploaded ${count} file(s) successfully! Indexing queued.</span>`;
    loadSources();
  } catch (err) {
    feedback.innerHTML = `<span style="color: var(--danger);">Upload failed: ${err.message}</span>`;
  }
}

async function handleRegisterUrl(e) {
  e.preventDefault();
  const feedback = document.getElementById("urlFeedback");
  const url = document.getElementById("urlInput").value.trim();
  const val = parseInt(document.getElementById("refreshValue").value, 10);
  const unit = document.getElementById("refreshUnit").value;

  feedback.innerHTML = `<span style="color: var(--accent);">Registering & crawling URL...</span>`;

  try {
    const res = await fetch("/api/ingest/url", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": getTenantId(),
      },
      body: JSON.stringify({
        url: url,
        refresh_value: val,
        refresh_interval: val,
        refresh_unit: unit,
        refresh_enabled: true,
        tenant_id: getTenantId(),
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    feedback.innerHTML = `<span style="color: var(--success);">URL registered successfully! Crawling queued.</span>`;
    document.getElementById("urlInput").value = "";
    loadSources();
  } catch (err) {
    feedback.innerHTML = `<span style="color: var(--danger);">Registration failed: ${err.message}</span>`;
  }
}

async function loadSources() {
  const tbody = document.getElementById("sourcesTableBody");
  try {
    const res = await fetch(`/api/ingest/sources?tenant_id=${encodeURIComponent(getTenantId())}`, {
      headers: { "X-Tenant-ID": getTenantId() },
    });
    if (!res.ok) throw new Error(await res.text());
    const sources = await res.json();

    if (!sources || sources.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No documents or web pages ingested yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = sources.map(s => {
      const hasError = Boolean(s.error || s.ingestion_error || (s.chunk_count === 0 && (s.status === "completed" || s.status === "failed")));
      const rawStatus = (hasError ? "failed" : (s.status || s.ingestion_status || "completed")).toLowerCase();
      let badgeClass = "badge-ready";
      if (rawStatus === "completed" || rawStatus === "ready") {
        badgeClass = "badge-ready";
      } else if (rawStatus === "pending" || rawStatus === "processing" || rawStatus === "queued") {
        badgeClass = "badge-processing";
      } else if (rawStatus === "failed" || rawStatus === "rejected" || rawStatus === "error") {
        badgeClass = "badge-failed";
      }

      const displayStatus = rawStatus.toUpperCase();
      const refreshText = s.refresh_schedule || (s.refresh_interval ? `${s.refresh_interval} ${s.refresh_unit || "hour"}(s)` : "Manual");
      const dateVal = s.created_at || s.last_ingested_at;
      const errorMsg = s.error || s.ingestion_error;

      return `
        <tr class="${hasError ? 'row-failed' : ''}">
          <td>
            <strong>${escapeHtml(s.name || s.url || "Source")}</strong>
            ${s.url && s.url !== s.name ? `<br><a href="${escapeHtml(s.url)}" target="_blank" style="font-size: 0.75rem; color: var(--accent);">${escapeHtml(s.url)}</a>` : ""}
            ${hasError ? `<div style="font-size: 0.73rem; color: var(--danger); margin-top: 4px; line-height: 1.3;">⚠️ <strong>Error:</strong> ${escapeHtml(errorMsg || "Extracted 0 chunks - needs reindexing")}</div>` : ""}
          </td>
          <td><span class="badge" style="background: var(--bg-elevated);">${escapeHtml(s.source_type || "document")}</span></td>
          <td><span class="badge ${badgeClass}">${displayStatus}</span></td>
          <td><strong style="${s.chunk_count === 0 ? 'color: var(--danger);' : ''}">${s.chunk_count || 0}</strong> chunks</td>
          <td>${escapeHtml(refreshText)}</td>
          <td>${formatDate(dateVal)}</td>
          <td>
            <button class="${hasError ? 'btn-primary' : 'btn-secondary'} btn-sm" onclick="reingestSource('${s.id}')" title="Reindex this source">🔄 Reindex</button>
            <button class="btn-danger btn-sm" onclick="deleteSource('${s.id}')">🗑️</button>
          </td>
        </tr>
      `;
    }).join("");

    // Auto-poll if any source is pending or processing
    const hasActive = sources.some(s => {
      const st = (s.status || s.ingestion_status || "").toUpperCase();
      return st === "PENDING" || st === "PROCESSING" || st === "QUEUED";
    });
    if (hasActive && !pollInterval) {
      pollInterval = setInterval(loadSources, 4000);
    } else if (!hasActive && pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--danger);">Failed to load sources: ${err.message}</td></tr>`;
  }
}

window.reingestSource = async function(id) {
  try {
    await fetch(`/api/ingest/reingest/${id}`, { method: "POST", headers: { "X-Tenant-ID": getTenantId() } });
    loadSources();
  } catch (err) {
    alert("Reingest error: " + err.message);
  }
};

window.deleteSource = async function(id) {
  if (!confirm("Delete this document and all indexed vectors?")) return;
  try {
    await fetch(`/api/ingest/sources/${id}`, { method: "DELETE", headers: { "X-Tenant-ID": getTenantId() } });
    loadSources();
  } catch (err) {
    alert("Delete error: " + err.message);
  }
};

// ── MCP Registry ────────────────────────────────────────────────────────────
function initMcp() {
  const modal = document.getElementById("mcpModal");
  const btnOpen = document.getElementById("btnOpenMcpModal");
  const btnClose = document.getElementById("btnCloseMcpModal");
  const form = document.getElementById("mcpRegisterForm");
  const transportSel = document.getElementById("mcpTransport");

  btnOpen.addEventListener("click", () => modal.classList.add("active"));
  btnClose.addEventListener("click", () => modal.classList.remove("active"));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("active"); });

  transportSel.addEventListener("change", () => {
    const isStdio = transportSel.value === "stdio";
    document.getElementById("mcpEndpointGroup").style.display = isStdio ? "none" : "block";
    document.getElementById("mcpStdioGroup").style.display = isStdio ? "block" : "none";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      name: document.getElementById("mcpName").value.trim(),
      description: document.getElementById("mcpDesc").value.trim(),
      transport_type: transportSel.value,
      endpoint_url: document.getElementById("mcpEndpoint").value.trim() || null,
      stdio_command: document.getElementById("mcpStdio").value.trim() || null,
      action_scope: document.getElementById("mcpScope").value,
      auth_type: document.getElementById("mcpAuthType").value,
      auth_secret: document.getElementById("mcpSecret").value || null,
      tenant_id: getTenantId(),
    };

    try {
      const res = await fetch("/api/admin/mcp/servers", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Tenant-ID": getTenantId() },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      modal.classList.remove("active");
      form.reset();
      loadMcpServers();
    } catch (err) {
      alert("Registration failed: " + err.message);
    }
  });
}

async function loadMcpServers() {
  const tbody = document.getElementById("mcpTableBody");
  try {
    const res = await fetch(`/api/admin/mcp/servers?tenant_id=${encodeURIComponent(getTenantId())}`, {
      headers: { "X-Tenant-ID": getTenantId() },
    });
    if (!res.ok) throw new Error(await res.text());
    const servers = await res.json();

    if (!servers || servers.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No MCP servers registered yet. Connect one to enable live external tools.</td></tr>`;
      return;
    }

    tbody.innerHTML = servers.map(s => `
      <tr>
        <td><strong>${escapeHtml(s.name)}</strong><br><small style="color: var(--text-muted);">${escapeHtml(s.description || "")}</small></td>
        <td><span class="badge" style="background: var(--bg-elevated);">${s.transport_type}</span></td>
        <td><code style="font-size: 0.75rem;">${escapeHtml(s.endpoint_url || s.stdio_command || "-")}</code></td>
        <td><span class="badge badge-ready">${s.action_scope}</span></td>
        <td><span class="badge ${s.enabled ? 'badge-ready' : 'badge-pending'}">${s.enabled ? 'Active' : 'Disabled'}</span></td>
        <td><strong>${s.tool_count || 0} tools</strong></td>
        <td>
          <button class="btn-secondary btn-sm" onclick="testMcpServer('${s.id}')">⚡ Test & Probe</button>
          <button class="btn-danger btn-sm" onclick="deleteMcpServer('${s.id}')">🗑️</button>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--danger);">Failed to load MCP servers: ${err.message}</td></tr>`;
  }
}

window.testMcpServer = async function(id) {
  try {
    const res = await fetch(`/api/admin/mcp/servers/${id}/test`, {
      method: "POST",
      headers: { "X-Tenant-ID": getTenantId() },
    });
    const data = await res.json();
    alert(`Discovered ${data.tools_discovered} tools: \n` + (data.tool_names || []).join(", "));
    loadMcpServers();
  } catch (err) {
    alert("Test probe failed: " + err.message);
  }
};

window.deleteMcpServer = async function(id) {
  if (!confirm("Delete this MCP server?")) return;
  try {
    await fetch(`/api/admin/mcp/servers/${id}`, { method: "DELETE", headers: { "X-Tenant-ID": getTenantId() } });
    loadMcpServers();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
};

// ── Agent Configuration ──────────────────────────────────────────────────────
function initAgentConfig() {
  document.getElementById("btnSaveAgentConfig").addEventListener("click", saveAgentConfig);
}

async function loadAgentConfig() {
  const container = document.getElementById("agentConfigContainer");
  try {
    const res = await fetch("/api/admin/config/agents", {
      headers: { "X-Tenant-ID": getTenantId() },
    });
    if (!res.ok) throw new Error(await res.text());
    const agents = await res.json();

    container.innerHTML = Object.entries(agents).map(([agentKey, cfg]) => `
      <div class="agent-card" data-agent="${agentKey}">
        <div class="agent-card-header">
          <span class="agent-name">${formatAgentName(agentKey)}</span>
          <span class="badge ${cfg.enabled ? 'badge-ready' : 'badge-pending'}">${cfg.enabled ? 'Enabled' : 'Disabled'}</span>
        </div>
        <div>
          <label style="font-size: 0.75rem; color: var(--text-secondary);">Model:</label>
          <select class="agent-model">
            <option value="gemini-2.5-pro" ${cfg.model === 'gemini-2.5-pro' ? 'selected' : ''}>gemini-2.5-pro (Deep reasoning)</option>
            <option value="gemini-2.5-flash" ${cfg.model === 'gemini-2.5-flash' ? 'selected' : ''}>gemini-2.5-flash (Fast & low-latency)</option>
            <option value="gemini-1.5-pro" ${cfg.model === 'gemini-1.5-pro' ? 'selected' : ''}>gemini-1.5-pro</option>
            <option value="gemini-1.5-flash" ${cfg.model === 'gemini-1.5-flash' ? 'selected' : ''}>gemini-1.5-flash</option>
          </select>
        </div>
        <div>
          <label style="font-size: 0.75rem; color: var(--text-secondary);">Temperature: <span class="temp-val">${cfg.temperature}</span></label>
          <input type="range" class="agent-temp" min="0" max="1" step="0.05" value="${cfg.temperature}" oninput="this.parentElement.querySelector('.temp-val').innerText = this.value">
        </div>
        <div>
          <label style="font-size: 0.75rem; color: var(--text-secondary);">Max Output Tokens:</label>
          <input type="number" class="agent-tokens" value="${cfg.max_output_tokens || 4096}" min="256" max="16384">
        </div>
        <div>
          <label style="font-size: 0.75rem; color: var(--text-secondary);">System Instruction:</label>
          <textarea class="agent-instruction" rows="5" style="font-size: 0.8rem; font-family: var(--font-mono);">${escapeHtml(cfg.instruction || "")}</textarea>
        </div>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<div style="color: var(--danger);">Failed to load agent configurations: ${err.message}</div>`;
  }
}

async function saveAgentConfig() {
  const cards = document.querySelectorAll(".agent-card");
  const payload = {};

  cards.forEach(card => {
    const agentKey = card.getAttribute("data-agent");
    payload[agentKey] = {
      model: card.querySelector(".agent-model").value,
      temperature: parseFloat(card.querySelector(".agent-temp").value),
      max_output_tokens: parseInt(card.querySelector(".agent-tokens").value, 10),
      instruction: card.querySelector(".agent-instruction").value,
    };
  });

  try {
    const res = await fetch("/api/admin/config/agents", {
      method: "PATCH",
      headers: { "Content-Type": "application/json", "X-Tenant-ID": getTenantId() },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    alert("✅ Agent configurations updated successfully without redeploying code!");
  } catch (err) {
    alert("Save failed: " + err.message);
  }
}

// ── Observability & Traces ───────────────────────────────────────────────────
function initTraces() {
  document.getElementById("btnRefreshTraces").addEventListener("click", loadTraces);
  const modal = document.getElementById("traceModal");
  document.getElementById("btnCloseTraceModal").addEventListener("click", () => modal.classList.remove("active"));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("active"); });
}

function renderQueryCell(text, isRewritten = false) {
  if (!text) return '<span style="color: var(--text-muted); font-size: 0.8rem; font-style: italic;">(None)</span>';
  const clean = escapeHtml(text);
  const color = isRewritten ? '#60a5fa' : 'var(--text-primary)';
  const font = isRewritten ? 'font-family: var(--font-mono);' : '';
  const icon = isRewritten ? '<span style="color: var(--accent); margin-right: 0.25rem;">⚡</span>' : '';

  if (text.length > 200) {
    return `
      <div class="fixed-query-box" style="color: ${color}; ${font}">
        ${icon}${clean}
      </div>
    `;
  }

  return `
    <div style="max-width: 240px; word-break: break-word; font-size: 0.85rem; color: ${color}; ${font}">
      ${icon}<strong>${clean}</strong>
    </div>
  `;
}

async function loadTraces() {
  const tbody = document.getElementById("tracesTableBody");
  try {
    const res = await fetch(`/api/admin/traces?tenant_id=${encodeURIComponent(getTenantId())}&limit=25`, {
      headers: { "X-Tenant-ID": getTenantId() },
    });
    if (!res.ok) throw new Error(await res.text());
    const traces = await res.json();

    if (!traces || traces.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No execution traces recorded yet. Submit queries in chat to log traces.</td></tr>`;
      return;
    }

    tbody.innerHTML = traces.map(t => {
      const userQ = t.user_query || "";
      const rewrittenQ = t.rewritten_query || "";
      const isTransformed = rewrittenQ && rewrittenQ.trim() !== userQ.trim();
      return `
      <tr>
        <td><code style="font-size: 0.75rem;">${(t.id || "").substring(0, 8)}...</code></td>
        <td style="max-width: 250px;">
          ${renderQueryCell(userQ, false)}
        </td>
        <td style="max-width: 260px;">
          ${isTransformed
            ? renderQueryCell(rewrittenQ, true)
            : `<span style="color: var(--text-muted); font-size: 0.8rem; font-style: italic;">(Direct / Unchanged)</span>`
          }
        </td>
        <td><span class="badge badge-processing">${t.total_latency_ms || 0} ms</span></td>
        <td>${t.input_tokens || 0} / ${t.output_tokens || 0}</td>
        <td><span class="badge ${t.grounding_passed ? 'badge-ready' : 'badge-failed'}">${t.grounding_passed ? 'PASSED' : 'FLAGGED'}</span></td>
        <td>${formatDate(t.created_at)}</td>
        <td><button class="btn-secondary btn-sm" onclick="viewTraceDetails('${t.id}')">Inspect 🔍</button></td>
      </tr>
    `}).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--danger);">Failed to load traces: ${err.message}</td></tr>`;
  }
}

window.viewTraceDetails = async function(traceId) {
  const modal = document.getElementById("traceModal");
  const body = document.getElementById("traceModalBody");
  body.innerHTML = `<div>Loading trace details...</div>`;
  modal.classList.add("active");

  try {
    const res = await fetch(`/api/admin/traces/${traceId}`, {
      headers: { "X-Tenant-ID": getTenantId() },
    });
    if (!res.ok) throw new Error(await res.text());
    const t = await res.json();

    const userQ = t.user_query || "";
    const rewrittenQ = t.rewritten_query || "";
    const isTransformed = rewrittenQ && rewrittenQ.trim() !== userQ.trim();

    body.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
        <div><strong>Trace ID:</strong> <code>${t.id}</code></div>
        <div><strong>Session ID:</strong> <code>${t.session_id}</code></div>
        <div><strong>Total Latency:</strong> ${t.total_latency_ms} ms</div>
        <div><strong>Grounding Verified:</strong> ${t.grounding_passed ? '✅ Yes' : '❌ Flagged'}</div>
      </div>

      <!-- Side-by-side Query Comparison -->
      <div style="margin-top: 0.75rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
        <div style="background: var(--bg-primary); padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border-color);">
          <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.35rem;">
            👤 Original User Sent <span style="font-size: 0.75rem; color: var(--text-muted);">(${userQ.length} chars)</span>
          </div>
          <div style="font-size: 0.88rem; color: var(--text-primary); max-height: 140px; overflow-x: auto; overflow-y: auto; white-space: pre-wrap; word-break: break-word; background: rgba(0, 0, 0, 0.2); padding: 0.5rem; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.05); scrollbar-width: thin;">
            ${escapeHtml(userQ)}
          </div>
        </div>
        <div style="background: var(--bg-primary); padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border-color);">
          <div style="font-size: 0.8rem; font-weight: 600; color: var(--accent); margin-bottom: 0.35rem;">
            🤖 AI Rewritten Query <span style="font-size: 0.75rem; color: var(--text-muted);">(${isTransformed ? `${rewrittenQ.length} chars` : 'Direct'})</span>
          </div>
          <div style="font-size: 0.88rem; color: ${isTransformed ? '#60a5fa' : 'var(--text-muted)'}; font-family: ${isTransformed ? 'var(--font-mono)' : 'inherit'}; max-height: 140px; overflow-x: auto; overflow-y: auto; white-space: pre-wrap; word-break: break-word; background: rgba(0, 0, 0, 0.2); padding: 0.5rem; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.05); scrollbar-width: thin;">
            ${isTransformed ? escapeHtml(rewrittenQ) : "<em>(Passed directly without rewriting)</em>"}
          </div>
        </div>
      </div>

      ${t.reranked_evidence ? `
        <div style="margin-top: 0.75rem;">
          <strong>Reranked Evidence (${Array.isArray(t.reranked_evidence) ? t.reranked_evidence.length : 0} items):</strong>
          <pre style="background: var(--bg-primary); padding: 0.5rem; border-radius: 4px; max-height: 150px; overflow-y: auto; font-size: 0.75rem; border: 1px solid var(--border-color);">${escapeHtml(JSON.stringify(t.reranked_evidence, null, 2))}</pre>
        </div>
      ` : ""}

      <div style="margin-top: 0.75rem;">
        <strong>Final Grounded Answer:</strong>
        <div style="background: var(--bg-primary); padding: 0.5rem; border-radius: 4px; margin-top: 0.25rem; max-height: 180px; overflow-y: auto; border: 1px solid var(--border-color);">
          ${renderMarkdown(t.final_answer || "")}
        </div>
      </div>
    `;
  } catch (err) {
    body.innerHTML = `<div style="color: var(--danger);">Failed to load trace: ${err.message}</div>`;
  }
};

// ── Utilities ────────────────────────────────────────────────────────────────
function formatAppError(rawMsg) {
  if (!rawMsg) return "We encountered a temporary issue while processing your request. Please try again.";
  const str = String(rawMsg);
  if (str.includes("503") || str.includes("UNAVAILABLE") || /high demand/i.test(str)) {
    return "The AI service is currently experiencing heavy traffic and high demand. Please try asking your question again in a few moments.";
  }
  if (str.includes("429") || str.includes("RESOURCE_EXHAUSTED") || /rate limit|quota/i.test(str)) {
    return "The system is currently receiving a high volume of requests. Please wait a moment before sending another query.";
  }
  if (str.includes("401") || str.includes("403") || /api_key|permission_denied/i.test(str)) {
    return "The service could not authenticate with the AI provider. Please verify your system settings or contact support.";
  }
  if (/timeout|connection|network/i.test(str)) {
    return "A temporary network issue occurred while connecting to the service. Please try again.";
  }
  if (str.includes("{") && str.includes("}")) {
    return "We encountered a temporary service issue while processing your request. Please try again in a moment.";
  }
  return str;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderMarkdown(md) {
  if (!md) return "";
  // Basic markdown formatting: bold, italics, code, bullet lists, newlines
  let html = escapeHtml(md);
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\n\n/g, "<br><br>");
  html = html.replace(/\n- (.*?)(?=(\n|$))/g, "<br>&bull; $1");
  html = html.replace(/\n\d+\. (.*?)(?=(\n|$))/g, "<br><strong>&bull;</strong> $1");
  return html;
}

function formatDate(isoStr) {
  if (!isoStr) return "-";
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + " " + d.toLocaleDateString();
  } catch {
    return isoStr;
  }
}

function formatAgentName(key) {
  return key.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}
