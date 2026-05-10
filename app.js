const allowedExtensions = new Set(["pdf", "doc", "docx", "csv"]);
// Swap this to http://127.0.0.1:8000 when running the backend locally.
const API_BASE_URL = "https://notebookllm-clone-genai-assignment-production-6fb9.up.railway.app/";

const documentInput = document.querySelector("#documentInput");
const uploadZone = document.querySelector("#uploadZone");
const sourceCard = document.querySelector("#sourceCard");
const sourceName = document.querySelector("#sourceName");
const sourceMeta = document.querySelector("#sourceMeta");
const removeFileButton = document.querySelector("#removeFileButton");
const statusPill = document.querySelector("#statusPill");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const sendButton = document.querySelector("#sendButton");

let currentFile = null;
let currentDocumentId = null;
let isThinking = false;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getExtension(fileName) {
  return fileName.split(".").pop().toLowerCase();
}

function setChatEnabled(enabled) {
  chatInput.disabled = !enabled;
  sendButton.disabled = !enabled || isThinking;
  chatInput.placeholder = enabled
    ? "Ask something about the uploaded document"
    : "Upload a document to start chatting";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(value) {
  const codeSpans = [];
  // Escape first, then allow a small Markdown subset so model output stays safe to render.
  let rendered = escapeHtml(value).replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE_SPAN_${codeSpans.length}@@`;
    codeSpans.push(`<code>${code}</code>`);
    return token;
  });

  rendered = rendered
    .replace(/\*\*([\s\S]+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__([\s\S]+?)__/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+?)\*/g, "<em>$1</em>")
    .replace(/_([^_\n]+?)_/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  codeSpans.forEach((code, index) => {
    rendered = rendered.replace(`@@CODE_SPAN_${index}@@`, code);
  });

  return rendered;
}

function renderMarkdown(content) {
  const lines = content.trim().split(/\r?\n/);
  const blocks = [];
  let paragraph = [];
  let list = null;
  let codeBlock = null;

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!list) return;
    blocks.push(`<${list.type}>${list.items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${list.type}>`);
    list = null;
  }

  lines.forEach((line) => {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      flushParagraph();
      flushList();

      if (codeBlock) {
        blocks.push(`<pre><code>${escapeHtml(codeBlock.join("\n"))}</code></pre>`);
        codeBlock = null;
      } else {
        codeBlock = [];
      }
      return;
    }

    if (codeBlock) {
      codeBlock.push(line);
      return;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length + 2;
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }

    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const type = unordered ? "ul" : "ol";
      if (!list || list.type !== type) {
        flushList();
        list = { type, items: [] };
      }
      list.items.push(unordered ? unordered[1] : ordered[1]);
      return;
    }

    flushList();
    paragraph.push(trimmed);
  });

  flushParagraph();
  flushList();
  if (codeBlock) {
    blocks.push(`<pre><code>${escapeHtml(codeBlock.join("\n"))}</code></pre>`);
  }

  return blocks.join("");
}

function updateSource(file) {
  currentFile = file;
  if (!file) currentDocumentId = null;
  sourceCard.classList.toggle("is-empty", !file);
  removeFileButton.hidden = !file;
  statusPill.classList.toggle("is-ready", Boolean(file));
  statusPill.textContent = file ? "Source ready" : "Waiting for source";

  if (!file) {
    sourceName.textContent = "No document uploaded";
    sourceMeta.textContent = "Chat unlocks after a file is added.";
    setChatEnabled(false);
    return;
  }

  const extension = getExtension(file.name).toUpperCase();
  sourceName.textContent = file.name;
  sourceMeta.textContent = `${extension} · ${formatBytes(file.size)}`;
  setChatEnabled(true);
  chatInput.focus();
}

function addMessage(role, content) {
  const message = document.createElement("article");
  message.className = `message ${role}`;
  message.innerHTML = role === "assistant" ? renderMarkdown(content) : `<p>${escapeHtml(content)}</p>`;
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

function addTypingMessage() {
  const message = document.createElement("article");
  message.className = "message assistant";
  message.setAttribute("aria-label", "Assistant is typing");
  message.innerHTML = `
    <div class="typing" aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `;
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

async function uploadDocument(file) {
  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "X-Filename": encodeURIComponent(file.name),
    },
    body: file,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Upload failed.");
  }

  return response.json();
}

async function askDocument(question) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      document_id: currentDocumentId,
      message: question,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Chat request failed.");
  }

  return response.json();
}

function validateFile(file) {
  if (!file) return false;

  const extension = getExtension(file.name);
  if (allowedExtensions.has(extension)) return true;

  addMessage("assistant", "That file type is not supported yet. Please upload a PDF, DOC, DOCX, or CSV.");
  return false;
}

async function handleSelectedFile(file) {
  if (!validateFile(file)) return;

  updateSource(file);
  setChatEnabled(false);
  statusPill.textContent = "Uploading source";

  try {
    const result = await uploadDocument(file);
    currentDocumentId = result.document_id;
    statusPill.textContent = "Source ready";
    setChatEnabled(true);
    addMessage("assistant", `${result.filename} is uploaded. Ask me what you want to know from it.`);
  } catch (error) {
    documentInput.value = "";
    updateSource(null);
    addMessage("assistant", error.message);
  }
}

documentInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  handleSelectedFile(file);
});

removeFileButton.addEventListener("click", () => {
  documentInput.value = "";
  updateSource(null);
  addMessage("assistant", "Source removed. Upload another document whenever you are ready.");
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();

  if (!currentFile || !currentDocumentId || !question || isThinking) return;

  addMessage("user", question);
  chatInput.value = "";
  isThinking = true;
  setChatEnabled(true);

  const typingMessage = addTypingMessage();
  askDocument(question)
    .then((result) => {
      typingMessage.remove();
      addMessage("assistant", result.answer);
    })
    .catch((error) => {
      typingMessage.remove();
      addMessage("assistant", error.message);
    })
    .finally(() => {
      isThinking = false;
      setChatEnabled(true);
    });
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, () => {
    uploadZone.classList.remove("is-dragging");
  });
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  handleSelectedFile(file);
});

updateSource(null);
