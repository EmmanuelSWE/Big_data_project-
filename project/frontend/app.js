const imageInput = document.getElementById("image-input");
const dropzoneLabel = document.getElementById("dropzone-label");
const preview = document.getElementById("preview");
const identifyBtn = document.getElementById("identify-btn");
const statusEl = document.getElementById("status");
const resultPanel = document.getElementById("result-panel");
const speciesValue = document.getElementById("species-value");
const individualValue = document.getElementById("individual-value");
const animalsList = document.getElementById("animals-list");
const animalsEmpty = document.getElementById("animals-empty");

let selectedFile = null;

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
  setStatus("Identifying…");
  resultPanel.hidden = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/api/identify", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Identification failed");
    }

    renderResult(data);
    setStatus("Done.");
    await refreshAnimals();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    identifyBtn.disabled = false;
  }
});

function renderResult(data) {
  const { species, individual } = data;
  speciesValue.textContent = `${species.label} (${(species.confidence * 100).toFixed(1)}%)`;

  if (individual.isNew) {
    individualValue.textContent = `New individual — ${individual.id}`;
  } else {
    individualValue.textContent = `Matched ${individual.id} (${(individual.confidence * 100).toFixed(1)}%)`;
  }

  resultPanel.hidden = false;
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

async function refreshAnimals() {
  const response = await fetch("/api/animals");
  const animals = await response.json();

  animalsList.innerHTML = "";
  animalsEmpty.hidden = animals.length > 0;

  for (const animal of animals) {
    const item = document.createElement("li");

    const meta = document.createElement("div");
    meta.className = "animal-meta";

    const idEl = document.createElement("span");
    idEl.className = "animal-id";
    idEl.textContent = animal.id;

    const speciesEl = document.createElement("span");
    speciesEl.className = "animal-species";
    speciesEl.textContent = animal.species;

    meta.append(idEl, speciesEl);

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => removeAnimal(animal.id));

    item.append(meta, removeBtn);
    animalsList.append(item);
  }
}

async function removeAnimal(id) {
  await fetch(`/api/animals/${id}`, { method: "DELETE" });
  await refreshAnimals();
}

refreshAnimals();
