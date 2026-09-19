/**
 * app.js — SNR-Aware Automatic Modulation Classification System
 * ─────────────────────────────────────────────────────────────────────────────
 * Connects the cyberpunk HUD frontend to the FastAPI deep learning backend.
 *
 * Core Capabilities:
 *   1. Client-side binary .npy validation and header parsing
 *   2. Instant I/Q, Magnitude, and Phase telemetry oscillogram rendering
 *   3. Multipart/Form-Data signal transmission to the unified /predict endpoint
 *   4. Dynamic SNR region badge & probability distribution rendering
 *   5. Modulation classification with expert-model availability fallback
 *   6. Toast notification system & SDR hardware modal management
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   1. CONFIGURATION
   ═══════════════════════════════════════════════════════════════════════════ */
const API_BASE_URL = 'http://127.0.0.1:8000'; // Change to '' if served directly by FastAPI
const SIGNAL_LENGTH = 128;                   // Expected samples per channel (2 × 128)

/* ═══════════════════════════════════════════════════════════════════════════
   2. APPLICATION STATE
   ═══════════════════════════════════════════════════════════════════════════ */
const state = {
  file: null,              // Selected .npy File object
  iChannel: null,          // Float32Array of In-phase samples (128)
  qChannel: null,          // Float32Array of Quadrature samples (128)
  magnitude: null,         // Float32Array of sqrt(I^2 + Q^2)
  phase: null,             // Float32Array of atan2(Q, I) in degrees
  activeView: 'iq',        // Active oscillogram tab: 'iq' | 'magnitude' | 'phase'
  chartInstance: null,     // Chart.js instance
  isPredicting: false,     // Prediction request in-flight flag
};

/* ═══════════════════════════════════════════════════════════════════════════
   3. DOM REFERENCES
   ═══════════════════════════════════════════════════════════════════════════ */
const dom = {
  // Upload Zone
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  fileInfo: document.getElementById('fileInfo'),
  fileName: document.getElementById('fileName'),
  fileMeta: document.getElementById('fileMeta'),
  removeFileBtn: document.getElementById('removeFileBtn'),

  // Action Buttons
  predictBtn: document.getElementById('predictBtn'),
  predictLabel: document.getElementById('predictLabel'),
  sdrBtn: document.getElementById('sdrBtn'),

  // Oscilloscope
  waveformSection: document.getElementById('waveformSection'),
  waveformMeta: document.getElementById('waveformMeta'),
  signalChart: document.getElementById('signalChart'),
  waveformTabs: document.querySelectorAll('.waveform-tab'),

  // Results — Stage 1 (SNR)
  resultsSection: document.getElementById('resultsSection'),
  rSnrClass: document.getElementById('rSnrClass'),
  rSnrConf: document.getElementById('rSnrConf'),
  snrProbs: document.getElementById('snrProbs'),

  // Results — Stage 2 (Modulation)
  modOnlineContent: document.getElementById('modOnlineContent'),
  modNotAvailable: document.getElementById('modNotAvailable'),
  modNotAvailableText: document.getElementById('modNotAvailableText'),
  rModulation: document.getElementById('rModulation'),
  rConf: document.getElementById('rConf'),
  rConfBar: document.getElementById('rConfBar'),
  probaBars: document.getElementById('probaBars'),

  // Modal & Status
  sdrModal: document.getElementById('sdrModal'),
  closeSdrModalBtn: document.getElementById('closeSdrModalBtn'),
  statusDot: document.getElementById('statusDot'),
  statusBadge: document.getElementById('statusBadge'),
  backendStatus: document.getElementById('backendStatus'),
  toast: document.getElementById('toast'),
};

/* ═══════════════════════════════════════════════════════════════════════════
   4. TOAST NOTIFICATION SYSTEM
   ═══════════════════════════════════════════════════════════════════════════ */
let toastTimeout = null;

function showToast(message, type = 'error') {
  if (!dom.toast) return;
  clearTimeout(toastTimeout);

  dom.toast.textContent = message;
  dom.toast.className = `toast visible ${type === 'success' ? 'toast--success' : 'toast--error'}`;

  toastTimeout = setTimeout(() => {
    dom.toast.classList.remove('visible');
  }, 4000);
}

/* ═══════════════════════════════════════════════════════════════════════════
   5. CLIENT-SIDE NUMPY (.NPY) BINARY PARSER & VALIDATION
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Parses raw .npy ArrayBuffer according to the NumPy binary format specification.
 * Extracts array shape, dtype, and typed data buffer.
 */
function parseNpyBuffer(buffer) {
  const bytes = new Uint8Array(buffer);

  // Validate magic prefix: \x93NUMPY
  const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) {
      throw new Error('Magic header mismatch. File is not a valid .npy binary.');
    }
  }

  const majorVersion = bytes[6];
  let headerLen = 0;
  let dataOffset = 0;

  if (majorVersion === 1) {
    const view = new DataView(buffer);
    headerLen = view.getUint16(8, true);
    dataOffset = 10 + headerLen;
  } else if (majorVersion === 2) {
    const view = new DataView(buffer);
    headerLen = view.getUint32(8, true);
    dataOffset = 12 + headerLen;
  } else {
    throw new Error(`Unsupported .npy version ${majorVersion}`);
  }

  const decoder = new TextDecoder('utf-8');
  const headerBytes = bytes.slice(majorVersion === 1 ? 10 : 12, (majorVersion === 1 ? 10 : 12) + headerLen);
  const headerStr = decoder.decode(headerBytes).trim();

  // Extract dtype
  const dtypeMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  if (!dtypeMatch) throw new Error('Cannot locate dtype descriptor in .npy header.');
  const rawDtype = dtypeMatch[1];
  const typeChar = rawDtype[1];
  const byteSize = parseInt(rawDtype.slice(2), 10);
  const isLittleEndian = (rawDtype[0] === '<' || rawDtype[0] === '=' || rawDtype[0] === '|');

  if (typeChar !== 'f' || (byteSize !== 4 && byteSize !== 8)) {
    throw new Error(`Unsupported dtype '${rawDtype}'. Only float32 or float64 arrays are accepted.`);
  }

  // Extract shape
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error('Cannot locate shape in .npy header.');
  const shapeStr = shapeMatch[1].trim();
  const shape = shapeStr
    ? shapeStr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n))
    : [];

  const totalElements = shape.length > 0
    ? shape.reduce((a, b) => a * b, 1)
    : (buffer.byteLength - dataOffset) / byteSize;

  const dataBuffer = buffer.slice(dataOffset, dataOffset + totalElements * byteSize);

  let data;
  if (byteSize === 4) {
    if (isLittleEndian) {
      data = new Float32Array(dataBuffer);
    } else {
      const view = new DataView(dataBuffer);
      data = new Float32Array(totalElements);
      for (let i = 0; i < totalElements; i++) {
        data[i] = view.getFloat32(i * 4, false);
      }
    }
  } else {
    // Float64 → convert to Float32
    const view = new DataView(dataBuffer);
    data = new Float32Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
      data[i] = isLittleEndian ? view.getFloat64(i * 8, true) : view.getFloat64(i * 8, false);
    }
  }

  return { data, shape, dtype: rawDtype };
}

/**
 * Extracts canonical I and Q channels (each 128 samples).
 */
function extractChannels(data, shape) {
  const SL = SIGNAL_LENGTH; // 128

  // (2, 128)
  if (shape.length === 2 && shape[0] === 2 && shape[1] === SL) {
    return {
      iChannel: data.slice(0, SL),
      qChannel: data.slice(SL, 2 * SL),
    };
  }

  // (1, 2, 128)
  if (shape.length === 3 && shape[0] === 1 && shape[1] === 2 && shape[2] === SL) {
    return {
      iChannel: data.slice(0, SL),
      qChannel: data.slice(SL, 2 * SL),
    };
  }

  // (256,)
  if (shape.length === 1 && shape[0] === 2 * SL) {
    return {
      iChannel: data.slice(0, SL),
      qChannel: data.slice(SL, 2 * SL),
    };
  }

  // (128, 2)
  if (shape.length === 2 && shape[0] === SL && shape[1] === 2) {
    const iCh = new Float32Array(SL);
    const qCh = new Float32Array(SL);
    for (let i = 0; i < SL; i++) {
      iCh[i] = data[i * 2];
      qCh[i] = data[i * 2 + 1];
    }
    return { iChannel: iCh, qChannel: qCh };
  }

  throw new Error(`Unexpected tensor shape (${shape.join(', ')}). Model requires 2 × 128 I/Q floats.`);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

/* ═══════════════════════════════════════════════════════════════════════════
   6. FILE SELECTION & INGESTION
   ═══════════════════════════════════════════════════════════════════════════ */

async function handleFileSelection(file) {
  if (!file) return;

  // Strict .npy extension validation
  if (!file.name.toLowerCase().endsWith('.npy')) {
    showToast('INVALID FILE TYPE. ONLY .NPY SIGNAL FILES ARE ACCEPTED.', 'error');
    return;
  }

  try {
    const buffer = await file.arrayBuffer();
    const { data, shape, dtype } = parseNpyBuffer(buffer);
    const { iChannel, qChannel } = extractChannels(data, shape);

    // Compute Derived Telemetry Signals: Magnitude & Phase
    const magnitude = new Float32Array(SIGNAL_LENGTH);
    const phase = new Float32Array(SIGNAL_LENGTH);

    for (let k = 0; k < SIGNAL_LENGTH; k++) {
      const i = iChannel[k];
      const q = qChannel[k];
      magnitude[k] = Math.sqrt(i * i + q * q);
      phase[k] = Math.atan2(q, i) * (180 / Math.PI); // Phase in degrees [-180, +180]
    }

    // Save to State
    state.file = file;
    state.iChannel = iChannel;
    state.qChannel = qChannel;
    state.magnitude = magnitude;
    state.phase = phase;

    // Update File Preview HUD
    dom.fileName.textContent = file.name;
    dom.fileMeta.textContent = `SHAPE: (${shape.join(', ')}) | DTYPE: ${dtype} | SIZE: ${formatBytes(file.size)}`;
    dom.fileInfo.classList.add('visible');

    // Enable Execute Button
    dom.predictBtn.disabled = false;
    dom.predictBtn.setAttribute('aria-disabled', 'false');

    // Render Waveform Oscillogram
    renderWaveform();
    dom.waveformSection.classList.add('visible');

    // Reset previous prediction results until user executes
    dom.resultsSection.classList.remove('visible');

    showToast(`SIGNAL TELEMETRY LOADED: ${file.name}`, 'success');

  } catch (err) {
    showToast(`INVALID NPY SIGNAL DATA — ${err.message}`, 'error');
    clearFile();
  }
}

function clearFile() {
  state.file = null;
  state.iChannel = null;
  state.qChannel = null;
  state.magnitude = null;
  state.phase = null;

  dom.fileInput.value = '';
  dom.fileInfo.classList.remove('visible');
  dom.predictBtn.disabled = true;
  dom.predictBtn.setAttribute('aria-disabled', 'true');
  dom.waveformSection.classList.remove('visible');
  dom.resultsSection.classList.remove('visible');

  if (state.chartInstance) {
    state.chartInstance.destroy();
    state.chartInstance = null;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   7. WAVEFORM OSCILLOGRAM (Chart.js)
   ═══════════════════════════════════════════════════════════════════════════ */

function renderWaveform() {
  if (!state.iChannel || !state.qChannel || !dom.signalChart) return;

  const labels = Array.from({ length: SIGNAL_LENGTH }, (_, i) => i);
  let datasets = [];
  let yAxisLabel = 'NORM. AMPLITUDE';
  let yMin = undefined;
  let yMax = undefined;

  if (state.activeView === 'iq') {
    dom.waveformMeta.textContent = `${SIGNAL_LENGTH} SAMPLES // DUAL I/Q CHANNEL`;
    datasets = [
      {
        label: 'I (In-Phase)',
        data: Array.from(state.iChannel),
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.05)',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.2,
        fill: true,
      },
      {
        label: 'Q (Quadrature)',
        data: Array.from(state.qChannel),
        borderColor: '#ff00ff',
        backgroundColor: 'rgba(255, 0, 255, 0.05)',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.2,
        fill: true,
      },
    ];
  } else if (state.activeView === 'magnitude') {
    dom.waveformMeta.textContent = `${SIGNAL_LENGTH} SAMPLES // ENVELOPE MAGNITUDE √(I² + Q²)`;
    yAxisLabel = 'MAGNITUDE |r|';
    datasets = [
      {
        label: 'Magnitude |r|',
        data: Array.from(state.magnitude),
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0, 212, 255, 0.08)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.2,
        fill: true,
      },
    ];
  } else if (state.activeView === 'phase') {
    dom.waveformMeta.textContent = `${SIGNAL_LENGTH} SAMPLES // INSTANTANEOUS PHASE ∠θ (DEGREES)`;
    yAxisLabel = 'PHASE (DEGREES)';
    yMin = -190;
    yMax = 190;
    datasets = [
      {
        label: 'Phase θ (°)',
        data: Array.from(state.phase),
        borderColor: '#ffaa00',
        backgroundColor: 'rgba(255, 170, 0, 0.08)',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.1,
        fill: true,
      },
    ];
  }

  if (state.chartInstance) {
    state.chartInstance.destroy();
  }

  const ctx = dom.signalChart.getContext('2d');
  state.chartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: '#e0e0e0',
            font: { family: "'Share Tech Mono', monospace", size: 12 },
            usePointStyle: true,
            boxWidth: 8,
          },
        },
        tooltip: {
          backgroundColor: '#12121a',
          borderColor: 'rgba(0, 255, 136, 0.4)',
          borderWidth: 1,
          titleColor: '#00d4ff',
          bodyColor: '#e0e0e0',
          titleFont: { family: "'Share Tech Mono', monospace" },
          bodyFont: { family: "'JetBrains Mono', monospace" },
          callbacks: {
            title: items => `SAMPLE: ${items[0].label}`,
            label: item => ` ${item.dataset.label}: ${item.raw.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'TIME INDEX (k)',
            color: '#6b7280',
            font: { family: "'Share Tech Mono', monospace", size: 11 },
          },
          ticks: {
            color: '#6b7280',
            font: { family: "'JetBrains Mono', monospace", size: 9 },
            maxTicksLimit: 16,
          },
          grid: { color: 'rgba(42, 42, 58, 0.4)' },
        },
        y: {
          min: yMin,
          max: yMax,
          title: {
            display: true,
            text: yAxisLabel,
            color: '#6b7280',
            font: { family: "'Share Tech Mono', monospace", size: 11 },
          },
          ticks: {
            color: '#6b7280',
            font: { family: "'JetBrains Mono', monospace", size: 9 },
          },
          grid: { color: 'rgba(42, 42, 58, 0.4)' },
        },
      },
    },
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   8. BACKEND API PREDICTION PIPELINE
   ═══════════════════════════════════════════════════════════════════════════ */

async function runPrediction() {
  if (state.isPredicting || !state.file) return;

  state.isPredicting = true;
  dom.predictBtn.classList.add('loading');
  dom.predictBtn.disabled = true;
  dom.predictLabel.textContent = 'ANALYZING SIGNAL...';

  // Build FormData with the actual .npy file
  const formData = new FormData();
  formData.append('file', state.file);

  try {
    const endpoint = `${API_BASE_URL}/predict`;
    const res = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errDetail = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errDetail;
      } catch (_) {}
      throw new Error(errDetail);
    }

    const data = await res.json();
    renderResults(data);
    showToast('PREDICTION EXECUTED SUCCESSFULLY', 'success');

  } catch (err) {
    const msg = err.message || 'Unknown network error';
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
      showToast('[ERROR] BACKEND CONNECTION FAILED — Is the server running at ' + API_BASE_URL + '?', 'error');
    } else {
      showToast(`[ERROR] ${msg.toUpperCase()}`, 'error');
    }
  } finally {
    state.isPredicting = false;
    dom.predictBtn.classList.remove('loading');
    dom.predictBtn.disabled = false;
    dom.predictLabel.textContent = '> EXECUTE PREDICTION';
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   9. RESULTS RENDERING
   ═══════════════════════════════════════════════════════════════════════════ */

function renderResults(data) {
  // ── Stage 1: SNR Region ──
  const snrCategory = (data.snr_category || 'UNKNOWN').toUpperCase();
  const snrConfidence = data.snr_confidence !== undefined ? data.snr_confidence : 0;
  const snrPct = (snrConfidence * 100).toFixed(1);

  dom.rSnrClass.textContent = `${snrCategory} SNR`;
  dom.rSnrConf.textContent = `CONFIDENCE: ${snrPct}%`;

  // Badge class adjustment
  dom.rSnrClass.className = 'snr-badge';
  if (snrCategory.includes('HIGH')) {
    dom.rSnrClass.classList.add('snr-badge--high');
  } else if (snrCategory.includes('MED')) {
    dom.rSnrClass.classList.add('snr-badge--medium');
  } else {
    dom.rSnrClass.classList.add('snr-badge--low');
  }

  // Render SNR Probability Distribution
  renderSnrProbabilities(data.snr_probabilities || {});

  // ── Stage 2: Modulation Type ──
  const isAvailable = data.modulation_available === true;

  if (isAvailable && data.modulation_class) {
    dom.modOnlineContent.style.display = 'block';
    dom.modNotAvailable.style.display = 'none';

    const modClass = data.modulation_class.toUpperCase();
    const modConf = data.modulation_confidence !== undefined ? data.modulation_confidence : 0;
    const modPct = (modConf * 100).toFixed(1);

    dom.rModulation.textContent = modClass;
    dom.rConf.textContent = `CONFIDENCE: ${modPct}%`;

    // Reset and animate confidence bar
    dom.rConfBar.style.width = '0%';
    setTimeout(() => {
      dom.rConfBar.style.width = `${Math.min(modConf * 100, 100)}%`;
    }, 100);

    // Render all class probabilities
    renderAllClassProbabilities(data.all_probabilities || {}, modClass);

  } else {
    // Model not available for this SNR region (e.g. Low SNR offline)
    dom.modOnlineContent.style.display = 'none';
    dom.modNotAvailable.style.display = 'block';
    dom.modNotAvailableText.textContent =
      data.message || `No AMC expert model is currently available for this SNR region (${snrCategory} SNR).`;
  }

  // Reveal results section and scroll into view smoothly
  dom.resultsSection.classList.add('visible');
  dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderSnrProbabilities(probs) {
  dom.snrProbs.innerHTML = '';

  const regions = [
    { key: 'HIGH',   label: 'HIGH',   val: probs.HIGH   !== undefined ? probs.HIGH   : 0, cls: 'high' },
    { key: 'MEDIUM', label: 'MEDIUM', val: probs.MEDIUM !== undefined ? probs.MEDIUM : 0, cls: 'medium' },
    { key: 'LOW',    label: 'LOW',    val: probs.LOW    !== undefined ? probs.LOW    : 0, cls: 'low' },
  ];

  regions.forEach(region => {
    const pct = (region.val * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'snr-prob-bar';
    row.innerHTML = `
      <span class="snr-prob-bar__label">${region.label}</span>
      <div class="snr-prob-bar__track">
        <div class="snr-prob-bar__fill snr-prob-bar__fill--${region.cls}" style="width:0%" data-target="${pct}"></div>
      </div>
      <span class="snr-prob-bar__value">${pct}%</span>
    `;
    dom.snrProbs.appendChild(row);
  });

  // Animate width transition
  requestAnimationFrame(() => {
    setTimeout(() => {
      dom.snrProbs.querySelectorAll('.snr-prob-bar__fill').forEach(bar => {
        bar.style.width = `${bar.dataset.target}%`;
      });
    }, 100);
  });
}

function renderAllClassProbabilities(allProbs, winnerClass) {
  dom.probaBars.innerHTML = '';
  const sorted = Object.entries(allProbs).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([clsName, prob]) => {
    const pct = (prob * 100).toFixed(1);
    const isWinner = clsName.toUpperCase() === winnerClass.toUpperCase();

    const row = document.createElement('div');
    row.className = 'snr-prob-bar';
    row.innerHTML = `
      <span class="snr-prob-bar__label" style="color: ${isWinner ? 'var(--accent-secondary)' : 'var(--muted-foreground)'}; font-weight: ${isWinner ? '700' : '400'}">${clsName}</span>
      <div class="snr-prob-bar__track">
        <div class="snr-prob-bar__fill" style="width:0%; background: ${isWinner ? 'var(--accent-secondary)' : 'var(--accent-tertiary)'}; box-shadow: ${isWinner ? '0 0 10px var(--accent-secondary)' : 'none'}" data-target="${pct}"></div>
      </div>
      <span class="snr-prob-bar__value" style="color: ${isWinner ? 'var(--accent-secondary)' : 'var(--foreground)'}">${pct}%</span>
    `;
    dom.probaBars.appendChild(row);
  });

  // Animate width transition
  requestAnimationFrame(() => {
    setTimeout(() => {
      dom.probaBars.querySelectorAll('.snr-prob-bar__fill').forEach(bar => {
        bar.style.width = `${bar.dataset.target}%`;
      });
    }, 120);
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   10. BACKEND HEALTH MONITOR
   ═══════════════════════════════════════════════════════════════════════════ */

async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      dom.statusBadge.textContent = 'SYS.ONLINE // AI_SIGINT_V1.0';
      dom.backendStatus.textContent = 'SYS_STATUS: ONLINE (PORT 8000)';
      dom.statusDot.style.background = 'var(--accent)';
      dom.statusDot.style.boxShadow = '0 0 8px var(--accent)';
    } else {
      throw new Error();
    }
  } catch (_) {
    dom.statusBadge.textContent = 'SYS.OFFLINE // CONNECT ERROR';
    dom.backendStatus.textContent = 'SYS_STATUS: OFFLINE (PORT 8000)';
    dom.statusDot.style.background = 'var(--destructive)';
    dom.statusDot.style.boxShadow = '0 0 8px var(--destructive)';
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   11. EVENT LISTENERS & BOOTSTRAP
   ═══════════════════════════════════════════════════════════════════════════ */

function initEventListeners() {
  // File Input Changed
  dom.fileInput.addEventListener('change', e => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  });

  // Drag & Drop Listeners
  dom.dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dom.dropZone.classList.add('drag-over');
  });

  dom.dropZone.addEventListener('dragleave', e => {
    e.preventDefault();
    dom.dropZone.classList.remove('drag-over');
  });

  dom.dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dom.dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  // Remove File
  dom.removeFileBtn.addEventListener('click', () => {
    clearFile();
    showToast('SIGNAL BUFFER CLEARED', 'success');
  });

  // Execute Prediction
  dom.predictBtn.addEventListener('click', runPrediction);

  // SDR Hardware Modal
  dom.sdrBtn.addEventListener('click', () => {
    dom.sdrModal.classList.add('visible');
  });

  dom.closeSdrModalBtn.addEventListener('click', () => {
    dom.sdrModal.classList.remove('visible');
  });

  dom.sdrModal.addEventListener('click', e => {
    if (e.target === dom.sdrModal) {
      dom.sdrModal.classList.remove('visible');
    }
  });

  // Waveform View Tabs
  dom.waveformTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      dom.waveformTabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      state.activeView = tab.dataset.view;
      renderWaveform();
    });
  });

  // Initial Health Check + Poll every 15 seconds
  checkBackendHealth();
  setInterval(checkBackendHealth, 15000);
}

document.addEventListener('DOMContentLoaded', initEventListeners);
