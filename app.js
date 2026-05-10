const allowedExtensions = new Set(["pdf", "doc", "docx", "csv"]);
// Swap this to http://127.0.0.1:8000 when running the backend locally.
// const API_BASE_URL = "https://notebookllm-clone-genai-assignment-production-6fb9.up.railway.app";
const API_BASE_URL = "http://127.0.0.1:8000";

const documentInput = document.querySelector("#documentInput");
const uploadZone = document.querySelector("#uploadZone");
const uploadProgress = document.querySelector("#uploadProgress");
const uploadProgressTitle = document.querySelector("#uploadProgressTitle");
const uploadProgressDetail = document.querySelector("#uploadProgressDetail");
const documentList = document.querySelector("#documentList");
const documentCount = document.querySelector("#documentCount");
const statusPill = document.querySelector("#statusPill");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const sendButton = document.querySelector("#sendButton");

let documents = [];
let selectedDocumentIds = new Set();
let isThinking = false;
let isUploading = false;
let uploadProgressTimer = null;
let openSourcesMessage = null;

const uploadSteps = [
  ["Uploading source", "Sending the file to the workspace"],
  ["Reading document", "Extracting text from the uploaded file"],
  ["Building index", "Chunking the text and preparing retrieval"],
  ["Almost ready", "Saving the searchable document context"],
];

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
    ? "Ask something across your uploaded documents"
    : "Upload documents to start chatting";
}

function setUploadProgress(visible, stepIndex = 0) {
  uploadProgress.hidden = !visible;
  uploadZone.classList.toggle("is-uploading", visible);
  documentInput.disabled = visible;

  if (!visible) {
    clearInterval(uploadProgressTimer);
    uploadProgressTimer = null;
    return;
  }

  const safeStepIndex = Math.min(stepIndex, uploadSteps.length - 1);
  const [title, detail] = uploadSteps[safeStepIndex];
  uploadProgressTitle.textContent = title;
  uploadProgressDetail.textContent = detail;
}

function startUploadProgress() {
  let stepIndex = 0;
  setUploadProgress(true, stepIndex);
  clearInterval(uploadProgressTimer);
  uploadProgressTimer = setInterval(() => {
    stepIndex = Math.min(stepIndex + 1, uploadSteps.length - 1);
    setUploadProgress(true, stepIndex);
    if (stepIndex === uploadSteps.length - 1) {
      clearInterval(uploadProgressTimer);
      uploadProgressTimer = null;
    }
  }, 1800);
}

function getSelectedDocuments() {
  return documents.filter((document) => selectedDocumentIds.has(document.id));
}

function renderDocumentList() {
  documentCount.textContent = `${selectedDocumentIds.size}/${documents.length}`;
  documentList.innerHTML = "";

  if (!documents.length) {
    documentList.innerHTML = '<p class="empty-library">Uploaded sources will appear here.</p>';
    return;
  }

  documents.forEach((item) => {
    const documentItem = document.createElement("article");
    const isSelected = selectedDocumentIds.has(item.id);
    documentItem.className = `document-item${isSelected ? " is-selected" : ""}`;
    documentItem.dataset.documentId = item.id;
    documentItem.innerHTML = `
      <label class="document-main">
        <input class="document-check" type="checkbox" value="${escapeHtml(item.id)}" ${isSelected ? "checked" : ""} />
        <span class="document-icon" aria-hidden="true">${escapeHtml(item.extension)}</span>
        <span class="document-copy">
          <strong>${escapeHtml(item.filename)}</strong>
          <small>${escapeHtml(item.extension)} · ${formatBytes(item.sizeBytes)} · ${item.chunkCount.toLocaleString()} chunks</small>
        </span>
      </label>
      <span class="document-actions">
        <button class="document-chunks-toggle" type="button" data-document-id="${escapeHtml(item.id)}">
          ${item.chunksVisible ? "Hide" : "Inspect"}
        </button>
        <button class="document-remove" type="button" data-document-id="${escapeHtml(item.id)}">
          Remove
        </button>
      </span>
      ${renderChunkBrowser(item)}
    `;
    documentList.append(documentItem);
  });
}

function renderChunkBrowser(item) {
  if (!item.chunksVisible) return "";

  if (item.chunksLoading) {
    return '<div class="chunk-browser"><p>Loading chunks...</p></div>';
  }

  if (item.chunksError) {
    return `<div class="chunk-browser"><p>${escapeHtml(item.chunksError)}</p></div>`;
  }

  const chunks = item.chunks || [];
  const selectedChunk = chunks.find((chunk) => chunk.chunk_index === item.selectedChunkIndex);
  return `
    <div class="chunk-browser">
      <div class="chunk-browser-header">
        <span>${chunks.length.toLocaleString()} chunks</span>
      </div>
      <div class="chunk-number-list">
        ${chunks.map((chunk) => `
          <button class="chunk-number${chunk.chunk_index === item.selectedChunkIndex ? " is-active" : ""}" type="button" data-document-id="${escapeHtml(item.id)}" data-chunk-index="${chunk.chunk_index}">
            ${chunk.chunk_index + 1}
          </button>
        `).join("")}
      </div>
      <div class="chunk-preview">
        ${selectedChunk ? `<strong>Chunk ${selectedChunk.chunk_index + 1}</strong><p>${escapeHtml(selectedChunk.text)}</p>` : "<p>Select a chunk to preview its text.</p>"}
      </div>
    </div>
  `;
}

function updateSourceSummary() {
  const selectedCount = selectedDocumentIds.size;
  statusPill.classList.toggle("is-ready", Boolean(selectedCount));
  statusPill.textContent = selectedCount
    ? `${selectedCount} of ${documents.length} source${documents.length === 1 ? "" : "s"} selected`
    : "Waiting for source";
  setChatEnabled(Boolean(selectedCount));
  renderDocumentList();
  if (selectedCount) chatInput.focus();
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

function renderSources(sources) {
  return `
    <div class="message-sources" hidden>
      ${sources.map((source) => `
        <section class="source-snippet">
          <div class="source-snippet-header">
            <strong>${escapeHtml(source.filename)} · Chunk ${source.chunk_index + 1}</strong>
            <button class="source-text-toggle" type="button">Show text</button>
          </div>
          <p hidden>${escapeHtml(source.text)}</p>
        </section>
      `).join("")}
    </div>
  `;
}

function addMessage(role, content, sources = []) {
  const message = document.createElement("article");
  message.className = `message ${role}`;
  message.innerHTML = role === "assistant" ? renderMarkdown(content) : `<p>${escapeHtml(content)}</p>`;
  if (role === "assistant" && sources.length) {
    message.innerHTML += `
      <button class="sources-toggle" type="button">Show sources</button>
      ${renderSources(sources)}
    `;
  }
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
      document_ids: Array.from(selectedDocumentIds),
      message: question,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Chat request failed.");
  }

  return response.json();
}

async function deleteDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not remove the document.");
  }

  return response.json();
}

async function fetchDocumentChunks(documentId) {
  const response = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}/chunks`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not load chunks.");
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

async function uploadSingleFile(file, totalFiles, fileIndex) {
  if (!validateFile(file)) return false;

  setChatEnabled(false);
  startUploadProgress();
  statusPill.textContent = totalFiles > 1 ? `Uploading ${fileIndex + 1} of ${totalFiles}` : "Uploading source";

  try {
    const result = await uploadDocument(file);
    const uploadedDocument = {
      id: result.document_id,
      filename: result.filename,
      extension: result.extension.toUpperCase(),
      sizeBytes: result.size_bytes,
      characterCount: result.character_count,
      chunkCount: result.chunk_count,
      chunks: null,
      chunksError: "",
      chunksLoading: false,
      chunksVisible: false,
      selectedChunkIndex: null,
    };
    documents = [uploadedDocument, ...documents];
    selectedDocumentIds.add(uploadedDocument.id);
    setUploadProgress(false);
    updateSourceSummary();
    addMessage("assistant", `${result.filename} is uploaded and checked for chat.`);
    return true;
  } catch (error) {
    setUploadProgress(false);
    updateSourceSummary();
    addMessage("assistant", error.message);
    return false;
  }
}

async function handleSelectedFiles(fileList) {
  if (isUploading) return;

  const files = Array.from(fileList || []);
  if (!files.length) return;

  isUploading = true;
  try {
    for (const [index, file] of files.entries()) {
      await uploadSingleFile(file, files.length, index);
    }
  } finally {
    isUploading = false;
  }

  documentInput.value = "";
  updateSourceSummary();
}

documentInput.addEventListener("change", (event) => {
  handleSelectedFiles(event.target.files);
});

documentList.addEventListener("change", (event) => {
  if (!event.target.classList.contains("document-check")) return;

  const documentId = event.target.value;
  if (event.target.checked) {
    selectedDocumentIds.add(documentId);
  } else {
    selectedDocumentIds.delete(documentId);
  }
  updateSourceSummary();
});

documentList.addEventListener("click", async (event) => {
  const removeButton = event.target.closest(".document-remove");
  const chunksButton = event.target.closest(".document-chunks-toggle");
  const chunkNumberButton = event.target.closest(".chunk-number");
  if (!removeButton && !chunksButton && !chunkNumberButton) return;

  event.preventDefault();

  if (chunksButton) {
    const documentId = chunksButton.dataset.documentId;
    const documentItem = documents.find((document) => document.id === documentId);
    if (!documentItem) return;

    documentItem.chunksVisible = !documentItem.chunksVisible;
    if (documentItem.chunksVisible && !documentItem.chunks && !documentItem.chunksLoading) {
      documentItem.chunksLoading = true;
      documentItem.chunksError = "";
      renderDocumentList();
      try {
        const result = await fetchDocumentChunks(documentId);
        documentItem.chunks = result.chunks;
        documentItem.selectedChunkIndex = result.chunks[0]?.chunk_index ?? null;
      } catch (error) {
        documentItem.chunksError = error.message;
      } finally {
        documentItem.chunksLoading = false;
      }
    }
    renderDocumentList();
    return;
  }

  if (chunkNumberButton) {
    const documentId = chunkNumberButton.dataset.documentId;
    const documentItem = documents.find((document) => document.id === documentId);
    if (!documentItem) return;

    documentItem.selectedChunkIndex = Number(chunkNumberButton.dataset.chunkIndex);
    renderDocumentList();
    return;
  }

  const documentId = removeButton.dataset.documentId;
  const documentToRemove = documents.find((document) => document.id === documentId);
  if (!documentToRemove) return;

  removeButton.disabled = true;
  try {
    await deleteDocument(documentId);
    selectedDocumentIds.delete(documentId);
    documents = documents.filter((document) => document.id !== documentId);
    updateSourceSummary();
    addMessage("assistant", `${documentToRemove.filename} was removed.`);
  } catch (error) {
    removeButton.disabled = false;
    addMessage("assistant", error.message);
  }
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();

  if (!selectedDocumentIds.size || !question || isThinking) return;

  addMessage("user", question);
  chatInput.value = "";
  isThinking = true;
  setChatEnabled(Boolean(selectedDocumentIds.size));

  const typingMessage = addTypingMessage();
  askDocument(question)
    .then((result) => {
      typingMessage.remove();
      addMessage("assistant", result.answer, result.sources || []);
    })
    .catch((error) => {
      typingMessage.remove();
      addMessage("assistant", error.message);
    })
    .finally(() => {
      isThinking = false;
      updateSourceSummary();
    });
});

messages.addEventListener("click", (event) => {
  const sourceTextToggle = event.target.closest(".source-text-toggle");
  if (sourceTextToggle) {
    const sourceText = sourceTextToggle.closest(".source-snippet")?.querySelector("p");
    if (!sourceText) return;

    const isOpening = sourceText.hidden;
    sourceText.hidden = !isOpening;
    sourceTextToggle.textContent = isOpening ? "Hide text" : "Show text";
    return;
  }

  const toggle = event.target.closest(".sources-toggle");
  if (!toggle) return;

  const sourcePanel = toggle.nextElementSibling;
  const isOpening = sourcePanel.hidden;
  if (openSourcesMessage && openSourcesMessage !== sourcePanel) {
    openSourcesMessage.hidden = true;
    openSourcesMessage.previousElementSibling.textContent = "Show sources";
  }
  sourcePanel.hidden = !isOpening;
  toggle.textContent = isOpening ? "Hide sources" : "Show sources";
  openSourcesMessage = isOpening ? sourcePanel : null;
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
  handleSelectedFiles(event.dataTransfer.files);
});

updateSourceSummary();
