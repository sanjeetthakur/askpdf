const state = {
  docId: null,
  filename: null,
};

const pdfInput = document.querySelector("#pdfInput");
const uploadButton = document.querySelector("#uploadButton");
const docMeta = document.querySelector("#docMeta");
const messages = document.querySelector("#messages");
const questionForm = document.querySelector("#questionForm");
const questionInput = document.querySelector("#questionInput");
const modelStatus = document.querySelector("#modelStatus");

fetch("/api/health")
  .then((response) => response.json())
  .then((data) => {
    modelStatus.textContent = `Local model: ${data.ollama_model}`;
  })
  .catch(() => {
    modelStatus.textContent = "Local server status unavailable";
  });

pdfInput.addEventListener("change", () => {
  const file = pdfInput.files[0];
  if (file) {
    docMeta.classList.remove("hidden");
    docMeta.textContent = `Selected: ${file.name}`;
  }
});

uploadButton.addEventListener("click", async () => {
  const file = pdfInput.files[0];
  if (!file) {
    showMetaError("Choose a PDF before analyzing.");
    return;
  }

  const formData = new FormData();
  formData.append("pdf", file);
  uploadButton.disabled = true;
  uploadButton.textContent = "Analyzing...";

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Upload failed.");

    state.docId = data.doc_id;
    state.filename = data.filename;
    docMeta.classList.remove("hidden");
    docMeta.innerHTML = `
      <strong>${escapeHtml(data.filename)}</strong><br>
      ${data.page_count} pages • ${data.chunk_count} searchable chunks<br>
      <span>${escapeHtml(data.preview)}...</span>
    `;
    addMessage("assistant", "Document indexed. Ask a question and I will answer using the PDF context.");
  } catch (error) {
    showMetaError(error.message);
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Analyze Document";
  }
});

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();

  if (!state.docId) {
    addMessage("assistant error", "Upload and analyze a PDF first.");
    return;
  }
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  setAskState(true);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: state.docId, question }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not answer the question.");

    const sourceSummary = data.sources
      .slice(0, 3)
      .map((source) => `Source ${source.rank}: score ${source.score}`)
      .join(" • ");
    addMessage("assistant", `${data.answer}\n\n${sourceSummary}`, data.mode);
  } catch (error) {
    addMessage("assistant error", error.message);
  } finally {
    setAskState(false);
  }
});

function addMessage(role, text, mode = "") {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.innerHTML = `
    <span>${escapeHtml(text)}</span>
    ${mode ? `<div class="sources">Mode: ${escapeHtml(mode)}</div>` : ""}
  `;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function setAskState(isLoading) {
  const button = questionForm.querySelector("button");
  button.disabled = isLoading;
  button.textContent = isLoading ? "..." : "Ask";
}

function showMetaError(message) {
  docMeta.classList.remove("hidden");
  docMeta.innerHTML = `<span class="error">${escapeHtml(message)}</span>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
