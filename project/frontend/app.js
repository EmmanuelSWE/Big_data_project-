const imageInput      = document.getElementById('image-input');
const dropzone        = document.querySelector('.dropzone');
const dropzoneLabel   = document.getElementById('dropzone-label');
const preview         = document.getElementById('preview');
const identifyBtn     = document.getElementById('identify-btn');
const status          = document.getElementById('status');
const resultPanel     = document.getElementById('result-panel');
const speciesValue    = document.getElementById('species-value');
const individualValue = document.getElementById('individual-value');
const animalsList     = document.getElementById('animals-list');
const animalsEmpty    = document.getElementById('animals-empty');

let selectedFile = null;

dropzone.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  selectedFile = file;
  dropzoneLabel.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  identifyBtn.disabled = false;
  status.textContent = '';
});

identifyBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  identifyBtn.disabled = true;
  resultPanel.hidden = true;
  status.textContent = 'Running Model A → Model B …';

  const form = new FormData();
  form.append('image', selectedFile);

  try {
    const response = await fetch('/api/identify', { method: 'POST', body: form });
    const data     = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);

    const s = data.species;
    speciesValue.textContent = `${s.label} (${(s.confidence * 100).toFixed(1)}%)`;

    const ind = data.individual;
    if (!ind) {
      individualValue.textContent = '—';
    } else if (ind.is_new) {
      individualValue.textContent =
        `new individual (best similarity ${ind.similarity})`;
    } else {
      individualValue.textContent =
        `${ind.individual} (similarity ${ind.similarity})`;
    }

    resultPanel.hidden = false;
    status.textContent = 'Done.';
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  } finally {
    identifyBtn.disabled = false;
  }
});

async function refreshAnimals() {
  try {
    const response = await fetch('/api/animals');
    const payload  = await response.json();
    if (payload.errors) console.warn('Model B errors:', payload.errors);

    const animals = payload.animals || [];
    animalsList.innerHTML = '';

    if (!animals.length) {
      animalsEmpty.hidden = false;
      return;
    }
    animalsEmpty.hidden = true;

    for (const a of animals) {
      const li = document.createElement('li');
      li.textContent = `${a.species} — ${a.individual}`;
      animalsList.appendChild(li);
    }
  } catch (err) {
    console.error(err);
  }
}

refreshAnimals();