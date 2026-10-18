const imageInput = document.getElementById("image-input");
const dropzoneLabel = document.getElementById("dropzone-label");
const preview = document.getElementById("preview");
const identifyBtn = document.getElementById("identify-btn");
const statusEl = document.getElementById("status");
const resultPanel = document.getElementById("result-panel");
const speciesValue = document.getElementById("species-value");
const individualValue = document.getElementById("individual-value");
const similarityValue = document.getElementById("similarity-value");
const matchesList = document.getElementById("matches-list");
const flagUnknown = document.getElementById("flag-unknown");
const thresholdInput = document.getElementById("threshold");
const flagUnrecognisedSpecies = document.getElementById("flag-unrecognised-species");
const speciesThresholdInput = document.getElementById("species-threshold");

let selectedFile = null;

flagUnknown.addEventListener("change", () => {
  thresholdInput.disabled = !flagUnknown.checked;
});

flagUnrecognisedSpecies.addEventListener("change", () => {
  speciesThresholdInput.disabled = !flagUnrecognisedSpecies.checked;
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropzoneLabel.hidden = true;
  identifyBtn.disabled = false;
  setStatus("");
});

identifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  identifyBtn.disabled = true;
  setStatus("Identifying… (first request loads the models, can take a moment)");
  resultPanel.hidden = true;

  const formData = new FormData();
  formData.append("image", selectedFile);
  if (flagUnknown.checked) {
    formData.append("threshold", thresholdInput.value);
  }
  if (flagUnrecognisedSpecies.checked) {
    formData.append("species_threshold", speciesThresholdInput.value);
  }

  try {
    const response = await fetch("/api/identify", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Identification failed");
    }

    renderResult(data);
    setStatus("Done.");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    identifyBtn.disabled = false;
  }
});

function renderResult(data) {
  if (data.individual === null) {
    // Species was rejected, so its confidence number refers to a guess we're
    // discarding - showing it next to "not recognised" reads backwards.
    speciesValue.textContent = data.species;
  } else {
    const confidencePercent = (data.species_confidence * 100).toFixed(1);
    speciesValue.textContent = `${data.species} (${confidencePercent}% confidence)`;
  }
  individualValue.textContent = data.individual ?? "— (species not recognised, no specialist to check)";
  similarityValue.textContent = data.similarity ?? "—";

  matchesList.innerHTML = "";
  for (const match of data.top_matches) {
    const item = document.createElement("li");
    item.textContent = `${match.individual} — ${match.similarity}`;
    matchesList.append(item);
  }

  resultPanel.hidden = false;
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}
