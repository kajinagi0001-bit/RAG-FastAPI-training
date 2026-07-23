def render_chat_ui_html() -> str:
    return """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --line: #d9e2ec;
      --muted: #627d98;
      --text: #1f2933;
      --accent: #0b7285;
      --accent-strong: #095c6a;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1, h2, h3 {
      margin: 0;
      letter-spacing: 0;
    }
    h1 {
      font-size: 24px;
    }
    h2 {
      font-size: 18px;
      margin-bottom: 12px;
    }
    h3 {
      font-size: 15px;
      margin-bottom: 8px;
    }
    a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
      gap: 18px;
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px;
    }
    section {
      min-width: 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    label {
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
    }
    textarea, select, input {
      width: 100%;
      border: 1px solid #bcccdc;
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      background: #ffffff;
      color: var(--text);
    }
    textarea {
      min-height: 180px;
      resize: vertical;
      line-height: 1.5;
    }
    .field {
      margin-bottom: 14px;
    }
    .divider {
      height: 1px;
      margin: 18px 0;
      background: var(--line);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 92px;
      gap: 10px;
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      font: inherit;
      font-weight: 700;
      color: #ffffff;
      background: var(--accent);
      cursor: pointer;
    }
    button:hover {
      background: var(--accent-strong);
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .answer {
      white-space: pre-wrap;
      line-height: 1.65;
      min-height: 140px;
    }
    .upload-result {
      min-height: 72px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      line-height: 1.5;
      font-size: 14px;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 13px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f8fafc;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #ffffff;
    }
    .item p {
      margin: 0;
      line-height: 1.55;
    }
    .muted {
      color: var(--muted);
    }
    .error {
      color: #b42318;
      font-weight: 700;
    }
    code {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
    }
    @media (max-width: 820px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      main {
        grid-template-columns: 1fr;
        padding: 12px;
      }
      .row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>RAG Chat</h1>
    <nav><a href="/dashboard">Evaluation Dashboard</a></nav>
  </header>
  <main>
    <section class="panel">
      <h2>Document Upload</h2>
      <form id="upload-form">
        <div class="field">
          <label for="document-file">File</label>
          <input id="document-file" name="file" type="file" accept=".md,.txt,.pdf" required>
        </div>
        <button id="upload-button" type="submit">Upload</button>
      </form>
      <div id="upload-result" class="upload-result muted">No document uploaded yet.</div>
      <div class="divider"></div>
      <form id="chat-form">
        <div class="field">
          <label for="question">Question</label>
          <textarea id="question" name="question" required></textarea>
        </div>
        <div class="row">
          <div class="field">
            <label for="mode">Mode</label>
            <select id="mode" name="mode">
              <option value="/chat">normal RAG</option>
              <option value="/agent">agent</option>
              <option value="/agent/tool-calling">tool-calling agent</option>
            </select>
          </div>
          <div class="field">
            <label for="top-k">top_k</label>
            <input id="top-k" name="top_k" type="number" min="1" max="10" value="3">
          </div>
        </div>
        <button id="send-button" type="submit">Send</button>
      </form>
    </section>
    <section class="stack">
      <div class="panel">
        <h2>Answer</h2>
        <div id="meta" class="meta"></div>
        <div id="answer" class="answer muted">No answer yet.</div>
      </div>
      <div class="panel">
        <h2>Timing</h2>
        <div id="timings" class="stack muted">No timing data yet.</div>
      </div>
      <div class="panel">
        <h2>Sources</h2>
        <div id="sources" class="stack muted">No sources yet.</div>
      </div>
      <div class="panel">
        <h2>Agent Steps</h2>
        <div id="steps" class="stack muted">No steps yet.</div>
      </div>
    </section>
  </main>
  <script>
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("document-file");
    const uploadButton = document.getElementById("upload-button");
    const uploadResultEl = document.getElementById("upload-result");
    const form = document.getElementById("chat-form");
    const questionInput = document.getElementById("question");
    const modeInput = document.getElementById("mode");
    const topKInput = document.getElementById("top-k");
    const sendButton = document.getElementById("send-button");
    const answerEl = document.getElementById("answer");
    const timingsEl = document.getElementById("timings");
    const sourcesEl = document.getElementById("sources");
    const stepsEl = document.getElementById("steps");
    const metaEl = document.getElementById("meta");

    const runTypeByPath = {
      "/chat": "chat",
      "/agent": "agent",
      "/agent/tool-calling": "tool_calling_agent"
    };

    uploadForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const file = fileInput.files[0];
      if (!file) {
        return;
      }

      const formData = new FormData();
      formData.append("file", file);
      uploadButton.disabled = true;
      uploadResultEl.className = "upload-result muted";
      uploadResultEl.textContent = "Uploading...";

      try {
        const response = await fetch("/documents/upload", {
          method: "POST",
          body: formData
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Upload failed.");
        }
        uploadResultEl.className = "upload-result";
        uploadResultEl.innerHTML = `
          <strong>${escapeHtml(payload.title)}</strong><br>
          document_id=${payload.id}<br>
          created_at=${escapeHtml(payload.created_at)}<br>
          <span class="muted">${escapeHtml(shorten(payload.content || "", 180))}</span>
        `;
      } catch (error) {
        uploadResultEl.className = "upload-result error";
        uploadResultEl.textContent = String(error.message || error);
      } finally {
        uploadButton.disabled = false;
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = questionInput.value.trim();
      const endpoint = modeInput.value;
      const topK = Number(topKInput.value || 3);
      if (!question) {
        return;
      }

      sendButton.disabled = true;
      answerEl.className = "answer muted";
      answerEl.textContent = "Loading...";
      sourcesEl.className = "stack muted";
      sourcesEl.textContent = "Loading...";
      timingsEl.className = "stack muted";
      timingsEl.textContent = "Loading...";
      stepsEl.className = "stack muted";
      stepsEl.textContent = "Loading...";
      metaEl.innerHTML = "";

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question, top_k: topK})
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Request failed.");
        }
        renderResponse(payload);
        await renderLatestRunLink(question, runTypeByPath[endpoint]);
      } catch (error) {
        answerEl.className = "answer error";
        answerEl.textContent = String(error.message || error);
        timingsEl.className = "stack muted";
        timingsEl.textContent = "No timing data.";
        sourcesEl.className = "stack muted";
        sourcesEl.textContent = "No sources.";
        stepsEl.className = "stack muted";
        stepsEl.textContent = "No steps.";
      } finally {
        sendButton.disabled = false;
      }
    });

    function renderResponse(payload) {
      answerEl.className = "answer";
      answerEl.textContent = payload.answer || "";
      renderTimings(payload.timings || {});
      renderSources(payload.sources || []);
      renderSteps(payload.steps || []);
    }

    function renderTimings(timings) {
      const entries = Object.entries(timings);
      if (!entries.length) {
        timingsEl.className = "stack muted";
        timingsEl.textContent = "No timing data.";
        return;
      }
      timingsEl.className = "stack";
      timingsEl.innerHTML = `
        <article class="item">
          ${entries.map(([name, seconds]) => `
            <p><strong>${escapeHtml(name)}</strong>: ${formatSeconds(seconds)}</p>
          `).join("")}
        </article>
      `;
    }

    function renderSources(sources) {
      if (!sources.length) {
        sourcesEl.className = "stack muted";
        sourcesEl.textContent = "No sources.";
        return;
      }
      sourcesEl.className = "stack";
      sourcesEl.innerHTML = sources.map((source) => `
        <article class="item">
          <h3>${escapeHtml(source.title)} · score ${Number(source.score).toFixed(3)}</h3>
          <p class="muted">document_id=${source.document_id} chunk_id=${source.chunk_id} chunk_index=${source.chunk_index}</p>
          <p>${escapeHtml(source.content)}</p>
        </article>
      `).join("");
    }

    function renderSteps(steps) {
      if (!steps.length) {
        stepsEl.className = "stack muted";
        stepsEl.textContent = "No steps.";
        return;
      }
      stepsEl.className = "stack";
      stepsEl.innerHTML = steps.map((step) => `
        <article class="item">
          <h3>${step.step}. ${escapeHtml(step.action)}</h3>
          <p>${escapeHtml(step.observation)}</p>
        </article>
      `).join("");
    }

    async function renderLatestRunLink(question, runType) {
      const response = await fetch("/rag-runs");
      if (!response.ok) {
        return;
      }
      const runs = await response.json();
      const run = runs.find((item) => item.question === question && item.run_type === runType);
      if (!run) {
        return;
      }
      metaEl.innerHTML = `
        <span class="pill">run_id: <a href="/rag-runs/${run.id}">${run.id}</a></span>
        <span class="pill">${escapeHtml(run.run_type)}</span>
        <span class="pill"><a href="/rag-runs/${run.id}/feedback">feedback</a></span>
        <span class="pill"><a href="/rag-runs/${run.id}/tool-calls">tool calls</a></span>
      `;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function shorten(value, maxLength) {
      if (value.length <= maxLength) {
        return value;
      }
      return value.slice(0, maxLength - 3) + "...";
    }

    function formatSeconds(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) {
        return "-";
      }
      return `${number.toFixed(3)}s`;
    }
  </script>
</body>
</html>"""
