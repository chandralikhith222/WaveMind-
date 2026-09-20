'use strict';

const API_BASE_URL = (() => {
  if (window.__API_BASE_URL__ && typeof window.__API_BASE_URL__ === 'string' && window.__API_BASE_URL__.trim()) {
    return window.__API_BASE_URL__.trim().replace(/\/+$/, '');
  }
  return 'https://wavemind.onrender.com';
})();

const SIGNAL_LENGTH = 128;

const state = {
  file: null,
  iChannel: null,
  qChannel: null,
  magnitude: null,
  phase: null,
  activeView: 'iq',
  chartInstance: null,
  isPredicting: false,
};

const dom = {
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  fileInfo: document.getElementById('fileInfo'),
  fileName: document.getElementById('fileName'),
  fileMeta: document.getElementById('fileMeta'),
  removeFileBtn: document.getElementById('removeFileBtn'),
  predictBtn: document.getElementById('predictBtn'),
  predictLabel: document.getElementById('predictLabel'),
  sdrBtn: document.getElementById('sdrBtn'),
  waveformSection: document.getElementById('waveformSection'),
  waveformMeta: document.getElementById('waveformMeta'),
  signalChart: document.getElementById('signalChart'),
  waveformTabs: document.querySelectorAll('.waveform-tab'),
  resultsSection: document.getElementById('resultsSection'),
  rSnrClass: document.getElementById('rSnrClass'),
  rSnrConf: document.getElementById('rSnrConf'),
  snrProbs: document.getElementById('snrProbs'),
  modOnlineContent: document.getElementById('modOnlineContent'),
  modNotAvailable: document.getElementById('modNotAvailable'),
  modNotAvailableText: document.getElementById('modNotAvailableText'),
  rModulation: document.getElementById('rModulation'),
  rConf: document.getElementById('rConf'),
  rConfBar: document.getElementById('rConfBar'),
  probaBars: document.getElementById('probaBars'),
  sdrModal: document.getElementById('sdrModal'),
  closeSdrModalBtn: document.getElementById('closeSdrModalBtn'),
  statusDot: document.getElementById('statusDot'),
  statusBadge: document.getElementById('statusBadge'),
  backendStatus: document.getElementById('backendStatus'),
  toast: document.getElementById('toast'),
};

let toastTimeout = null;

function showToast(message, type = 'error') {
  if (!dom.toast) return;
  clearTimeout(toastTimeout);
  dom.toast.textContent = message;
  dom.toast.className = 'toast visible ' + (type === 'success' ? 'toast--success' : 'toast--error');
  toastTimeout = setTimeout(() => {
    dom.toast.classList.remove('visible');
  }, 5000);
}

function fetchWithTimeout(url, options, timeoutMs) {
  options = options || {};
  timeoutMs = timeoutMs || 90000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, Object.assign({}, options, { signal: controller.signal }))
    .finally(() => clearTimeout(timer));
}

function setStatusWakingUp() {
  dom.statusBadge.textContent = 'SYS.WAKING // RENDER COLD START...';
  dom.backendStatus.textContent = 'SYS_STATUS: WAKING UP - PLEASE WAIT (~30s)';
  dom.statusDot.style.background = '#ffaa00';
  dom.statusDot.style.boxShadow = '0 0 8px #ffaa00';
}

function setStatusOnline() {
  dom.statusBadge.textContent = 'SYS.ONLINE // AI_SIGINT_V1.0';
  dom.backendStatus.textContent = 'SYS_STATUS: ONLINE (' + API_BASE_URL + ')';
  dom.statusDot.style.background = 'var(--accent)';
  dom.statusDot.style.boxShadow = '0 0 8px var(--accent)';
}

function setStatusOffline() {
  dom.statusBadge.textContent = 'SYS.OFFLINE // CONNECT ERROR';
  dom.backendStatus.textContent = 'SYS_STATUS: OFFLINE (' + API_BASE_URL + ')';
  dom.statusDot.style.background = 'var(--destructive)';
  dom.statusDot.style.boxShadow = '0 0 8px var(--destructive)';
}

function parseNpyBuffer(buffer) {
  const bytes = new Uint8Array(buffer);
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
    throw new Error('Unsupported .npy version ' + majorVersion);
  }

  const headerStart = majorVersion === 1 ? 10 : 12;
  const decoder = new TextDecoder('utf-8');
  const headerBytes = bytes.slice(headerStart, headerStart + headerLen);
  const headerStr = decoder.decode(headerBytes).trim();

  const dtypeMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  if (!dtypeMatch) throw new Error('Cannot locate dtype descriptor in .npy header.');
  const rawDtype = dtypeMatch[1];
  const typeChar = rawDtype[1];
  const byteSize = parseInt(rawDtype.slice(2), 10);
  const isLittleEndian = (rawDtype[0] === '<' || rawDtype[0] === '=' || rawDtype[0] === '|');

  if (typeChar !== 'f' || (byteSize !== 4 && byteSize !== 8)) {
    throw new Error('Unsupported dtype ' + rawDtype + '. Only float32 or float64 arrays are accepted.');
  }

  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error('Cannot locate shape in .npy header.');
  const shapeStr = shapeMatch[1].trim();
  const shape = shapeStr
    ? shapeStr.split(',').map(function(s) { return parseInt(s.trim(), 10); }).filter(function(n) { return !isNaN(n); })
    : [];

  const totalElements = shape.length > 0
    ? shape.reduce(function(a, b) { return a * b; }, 1)
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
    const view = new DataView(dataBuffer);
    data = new Float32Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
      data[i] = isLittleEndian ? view.getFloat64(i * 8, true) : view.getFloat64(i * 8, false);
    }
  }

  return { data: data, shape: shape, dtype: rawDtype };
}

function extractChannels(data, shape) {
  const SL = SIGNAL_LENGTH;

  if (shape.length === 2 && shape[0] === 2 && shape[1] === SL) {
    return { iChannel: data.slice(0, SL), qChannel: data.slice(SL, 2 * SL) };
  }
  if (shape.length === 3 && shape[0] === 1 && shape[1] === 2 && shape[2] === SL) {
    return { iChannel: data.slice(0, SL), qChannel: data.slice(SL, 2 * SL) };
  }
  if (shape.length === 1 && shape[0] === 2 * SL) {
    return { iChannel: data.slice(0, SL), qChannel: data.slice(SL, 2 * SL) };
  }
  if (shape.length === 2 && shape[0] === SL && shape[1] === 2) {
    const iCh = new Float32Array(SL);
    const qCh = new Float32Array(SL);
    for (let i = 0; i < SL; i++) {
      iCh[i] = data[i * 2];
      qCh[i] = data[i * 2 + 1];
    }
    return { iChannel: iCh, qChannel: qCh };
  }

  throw new Error('Unexpected tensor shape (' + shape.join(', ') + '). Model requires 2 x 128 I/Q floats.');
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

async function handleFileSelection(file) {
  if (!file) return;

  if (!file.name.toLowerCase().endsWith('.npy')) {
    showToast('INVALID FILE TYPE. ONLY .NPY SIGNAL FILES ARE ACCEPTED.', 'error');
    return;
  }

  try {
    const buffer = await file.arrayBuffer();
    const parsed = parseNpyBuffer(buffer);
    const channels = extractChannels(parsed.data, parsed.shape);
    const iChannel = channels.iChannel;
    const qChannel = channels.qChannel;

    const magnitude = new Float32Array(SIGNAL_LENGTH);
    const phase = new Float32Array(SIGNAL_LENGTH);

    for (let k = 0; k < SIGNAL_LENGTH; k++) {
      const iv = iChannel[k];
      const qv = qChannel[k];
      magnitude[k] = Math.sqrt(iv * iv + qv * qv);
      phase[k] = Math.atan2(qv, iv) * (180 / Math.PI);
    }

    state.file = file;
    state.iChannel = iChannel;
    state.qChannel = qChannel;
    state.magnitude = magnitude;
    state.phase = phase;

    dom.fileName.textContent = file.name;
    dom.fileMeta.textContent = 'SHAPE: (' + parsed.shape.join(', ') + ') | DTYPE: ' + parsed.dtype + ' | SIZE: ' + formatBytes(file.size);
    dom.fileInfo.classList.add('visible');

    dom.predictBtn.disabled = false;
    dom.predictBtn.setAttribute('aria-disabled', 'false');

    renderWaveform();
    dom.waveformSection.classList.add('visible');
    dom.resultsSection.classList.remove('visible');

    showToast('SIGNAL TELEMETRY LOADED: ' + file.name, 'success');

  } catch (err) {
    showToast('INVALID NPY SIGNAL DATA - ' + err.message, 'error');
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

function renderWaveform() {
  if (!state.iChannel || !state.qChannel || !dom.signalChart) return;

  const labels = Array.from({ length: SIGNAL_LENGTH }, function(_, i) { return i; });
  let datasets = [];
  let yAxisLabel = 'NORM. AMPLITUDE';
  let yMin = undefined;
  let yMax = undefined;

  if (state.activeView === 'iq') {
    dom.waveformMeta.textContent = SIGNAL_LENGTH + ' SAMPLES // DUAL I/Q CHANNEL';
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
    dom.waveformMeta.textContent = SIGNAL_LENGTH + ' SAMPLES // ENVELOPE MAGNITUDE';
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
    dom.waveformMeta.textContent = SIGNAL_LENGTH + ' SAMPLES // INSTANTANEOUS PHASE (DEGREES)';
    yAxisLabel = 'PHASE (DEGREES)';
    yMin = -190;
    yMax = 190;
    datasets = [
      {
        label: 'Phase (deg)',
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
    data: { labels: labels, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: { mode: 'index', intersect: false },
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
            title: function(items) { return 'SAMPLE: ' + items[0].label; },
            label: function(item) { return ' ' + item.dataset.label + ': ' + item.raw.toFixed(4); },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'TIME INDEX (k)', color: '#6b7280', font: { family: "'Share Tech Mono', monospace", size: 11 } },
          ticks: { color: '#6b7280', font: { family: "'JetBrains Mono', monospace", size: 9 }, maxTicksLimit: 16 },
          grid: { color: 'rgba(42, 42, 58, 0.4)' },
        },
        y: {
          min: yMin,
          max: yMax,
          title: { display: true, text: yAxisLabel, color: '#6b7280', font: { family: "'Share Tech Mono', monospace", size: 11 } },
          ticks: { color: '#6b7280', font: { family: "'JetBrains Mono', monospace", size: 9 } },
          grid: { color: 'rgba(42, 42, 58, 0.4)' },
        },
      },
    },
  });
}

async function runPrediction() {
  if (state.isPredicting || !state.file) return;

  state.isPredicting = true;
  dom.predictBtn.classList.add('loading');
  dom.predictBtn.disabled = true;
  dom.predictLabel.textContent = 'ANALYZING SIGNAL...';

  setStatusWakingUp();
  showToast('CONNECTING TO BACKEND - MAY TAKE UP TO 30s ON COLD START...', 'error');

  const formData = new FormData();
  formData.append('file', state.file);

  try {
    const endpoint = API_BASE_URL.replace(/\/+$/, '') + '/predict';
    const res = await fetchWithTimeout(endpoint, { method: 'POST', body: formData }, 90000);

    if (!res.ok) {
      let errDetail = 'HTTP ' + res.status;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errDetail;
      } catch (_) {}
      throw new Error(errDetail);
    }

    const data = await res.json();
    setStatusOnline();
    renderResults(data);
    showToast('PREDICTION EXECUTED SUCCESSFULLY', 'success');

  } catch (err) {
    setStatusOffline();
    if (err.name === 'AbortError') {
      showToast('[ERROR] REQUEST TIMED OUT - BACKEND DID NOT RESPOND IN 90s. TRY AGAIN.', 'error');
    } else if (
      err.message.indexOf('Failed to fetch') !== -1 ||
      err.message.indexOf('NetworkError') !== -1 ||
      err.message.indexOf('Load failed') !== -1
    ) {
      showToast('[ERROR] BACKEND UNREACHABLE - CHECK https://wavemind.onrender.com IS DEPLOYED.', 'error');
    } else {
      showToast('[ERROR] ' + err.message.toUpperCase(), 'error');
    }
  } finally {
    state.isPredicting = false;
    dom.predictBtn.classList.remove('loading');
    dom.predictBtn.disabled = false;
    dom.predictLabel.textContent = '> EXECUTE PREDICTION';
  }
}

function renderResults(data) {
  const snrCategory = (data.snr_category || 'UNKNOWN').toUpperCase();
  const snrConfidence = data.snr_confidence !== undefined ? data.snr_confidence : 0;
  const snrPct = (snrConfidence * 100).toFixed(1);

  dom.rSnrClass.textContent = snrCategory + ' SNR';
  dom.rSnrConf.textContent = 'CONFIDENCE: ' + snrPct + '%';

  dom.rSnrClass.className = 'snr-badge';
  if (snrCategory.indexOf('HIGH') !== -1) {
    dom.rSnrClass.classList.add('snr-badge--high');
  } else if (snrCategory.indexOf('MED') !== -1) {
    dom.rSnrClass.classList.add('snr-badge--medium');
  } else {
    dom.rSnrClass.classList.add('snr-badge--low');
  }

  renderSnrProbabilities(data.snr_probabilities || {});

  const isAvailable = data.modulation_available === true;

  if (isAvailable && data.modulation_class) {
    dom.modOnlineContent.style.display = 'block';
    dom.modNotAvailable.style.display = 'none';

    const modClass = data.modulation_class.toUpperCase();
    const modConf = data.modulation_confidence !== undefined ? data.modulation_confidence : 0;
    const modPct = (modConf * 100).toFixed(1);

    dom.rModulation.textContent = modClass;
    dom.rConf.textContent = 'CONFIDENCE: ' + modPct + '%';

    dom.rConfBar.style.width = '0%';
    setTimeout(function() {
      dom.rConfBar.style.width = Math.min(modConf * 100, 100) + '%';
    }, 100);

    renderAllClassProbabilities(data.all_probabilities || {}, modClass);

  } else {
    dom.modOnlineContent.style.display = 'none';
    dom.modNotAvailable.style.display = 'block';
    dom.modNotAvailableText.textContent =
      data.message || 'No AMC expert model is currently available for this SNR region (' + snrCategory + ' SNR).';
  }

  dom.resultsSection.classList.add('visible');
  dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderSnrProbabilities(probs) {
  dom.snrProbs.innerHTML = '';

  const regions = [
    { label: 'HIGH',   val: probs.HIGH   !== undefined ? probs.HIGH   : 0, cls: 'high' },
    { label: 'MEDIUM', val: probs.MEDIUM !== undefined ? probs.MEDIUM : 0, cls: 'medium' },
    { label: 'LOW',    val: probs.LOW    !== undefined ? probs.LOW    : 0, cls: 'low' },
  ];

  regions.forEach(function(region) {
    const pct = (region.val * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'snr-prob-bar';
    row.innerHTML =
      '<span class="snr-prob-bar__label">' + region.label + '</span>' +
      '<div class="snr-prob-bar__track">' +
        '<div class="snr-prob-bar__fill snr-prob-bar__fill--' + region.cls + '" style="width:0%" data-target="' + pct + '"></div>' +
      '</div>' +
      '<span class="snr-prob-bar__value">' + pct + '%</span>';
    dom.snrProbs.appendChild(row);
  });

  requestAnimationFrame(function() {
    setTimeout(function() {
      dom.snrProbs.querySelectorAll('.snr-prob-bar__fill').forEach(function(bar) {
        bar.style.width = bar.dataset.target + '%';
      });
    }, 100);
  });
}

function renderAllClassProbabilities(allProbs, winnerClass) {
  dom.probaBars.innerHTML = '';
  const sorted = Object.entries(allProbs).sort(function(a, b) { return b[1] - a[1]; });

  sorted.forEach(function(entry) {
    const clsName = entry[0];
    const prob = entry[1];
    const pct = (prob * 100).toFixed(1);
    const isWinner = clsName.toUpperCase() === winnerClass.toUpperCase();

    const row = document.createElement('div');
    row.className = 'snr-prob-bar';
    row.innerHTML =
      '<span class="snr-prob-bar__label" style="color:' + (isWinner ? 'var(--accent-secondary)' : 'var(--muted-foreground)') + ';font-weight:' + (isWinner ? '700' : '400') + '">' + clsName + '</span>' +
      '<div class="snr-prob-bar__track">' +
        '<div class="snr-prob-bar__fill" style="width:0%;background:' + (isWinner ? 'var(--accent-secondary)' : 'var(--accent-tertiary)') + ';box-shadow:' + (isWinner ? '0 0 10px var(--accent-secondary)' : 'none') + '" data-target="' + pct + '"></div>' +
      '</div>' +
      '<span class="snr-prob-bar__value" style="color:' + (isWinner ? 'var(--accent-secondary)' : 'var(--foreground)') + '">' + pct + '%</span>';
    dom.probaBars.appendChild(row);
  });

  requestAnimationFrame(function() {
    setTimeout(function() {
      dom.probaBars.querySelectorAll('.snr-prob-bar__fill').forEach(function(bar) {
        bar.style.width = bar.dataset.target + '%';
      });
    }, 120);
  });
}

async function checkBackendHealth() {
  try {
    const healthUrl = API_BASE_URL.replace(/\/+$/, '') + '/health';
    const res = await fetchWithTimeout(healthUrl, {}, 15000);
    if (res.ok) {
      setStatusOnline();
    } else {
      setStatusOffline();
    }
  } catch (_) {
    setStatusOffline();
  }
}

function initEventListeners() {
  dom.fileInput.addEventListener('change', function(e) {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  });

  dom.dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    dom.dropZone.classList.add('drag-over');
  });

  dom.dropZone.addEventListener('dragleave', function(e) {
    e.preventDefault();
    dom.dropZone.classList.remove('drag-over');
  });

  dom.dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dom.dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  dom.removeFileBtn.addEventListener('click', function() {
    clearFile();
    showToast('SIGNAL BUFFER CLEARED', 'success');
  });

  dom.predictBtn.addEventListener('click', runPrediction);

  dom.sdrBtn.addEventListener('click', function() {
    dom.sdrModal.classList.add('visible');
  });

  dom.closeSdrModalBtn.addEventListener('click', function() {
    dom.sdrModal.classList.remove('visible');
  });

  dom.sdrModal.addEventListener('click', function(e) {
    if (e.target === dom.sdrModal) {
      dom.sdrModal.classList.remove('visible');
    }
  });

  dom.waveformTabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      dom.waveformTabs.forEach(function(t) {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      state.activeView = tab.dataset.view;
      renderWaveform();
    });
  });

  checkBackendHealth();
  setInterval(checkBackendHealth, 30000);
}

document.addEventListener('DOMContentLoaded', initEventListeners);