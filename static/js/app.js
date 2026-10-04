document.addEventListener('DOMContentLoaded', () => {
    // --- State Management ---
    let debounceTimer = null;
    let currentPresets = [];
    let batchData = null;
    let doughnutChartInstance = null;
    let barChartInstance = null;

    // --- DOM Elements ---
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Live Tab Elements
    const domainSelect = document.getElementById('domain-select');
    const presetChipsContainer = document.getElementById('preset-chips');
    const liveInput = document.getElementById('live-text-input');
    const charCounter = document.getElementById('char-counter');
    const realtimeToggle = document.getElementById('realtime-toggle');
    const clearBtn = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const copyJsonBtn = document.getElementById('copy-json-btn');

    const resultDisplay = document.getElementById('result-display');
    const sentimentEmoji = document.getElementById('sentiment-emoji');
    const sentimentLabel = document.getElementById('sentiment-label');
    const confidenceScore = document.getElementById('confidence-score');
    
    const probPosVal = document.getElementById('prob-pos-val');
    const probNeuVal = document.getElementById('prob-neu-val');
    const probNegVal = document.getElementById('prob-neg-val');
    const probPosBar = document.getElementById('prob-pos-bar');
    const probNeuBar = document.getElementById('prob-neu-bar');
    const probNegBar = document.getElementById('prob-neg-bar');

    const polarityVal = document.getElementById('polarity-value');
    const polarityDot = document.getElementById('polarity-dot');
    const subjectivityVal = document.getElementById('subjectivity-value');
    const subjectivityFill = document.getElementById('subjectivity-fill');
    const highlightedTextBox = document.getElementById('highlighted-text');

    // Batch Tab Elements
    const dropZone = document.getElementById('drop-zone');
    const csvFileInput = document.getElementById('csv-file-input');
    const loadSampleCsvBtn = document.getElementById('load-sample-csv-btn');
    const batchResultsWrapper = document.getElementById('batch-results-wrapper');
    const tableSearch = document.getElementById('table-search');
    const tableFilter = document.getElementById('table-filter');
    const batchTableBody = document.getElementById('batch-table-body');
    const tableShowingCount = document.getElementById('table-showing-count');

    // Model Analytics Elements
    const retrainBtn = document.getElementById('retrain-btn');

    // API Playground Elements
    const sendApiBtn = document.getElementById('send-api-btn');
    const apiRequestBody = document.getElementById('api-request-body');
    const apiResponseCode = document.getElementById('api-response-code');
    const codeTabs = document.querySelectorAll('.code-tab');
    const codeSnippetBox = document.getElementById('code-snippet-box');

    // --- Tab Switching ---
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.getAttribute('data-tab');
            navTabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(target).classList.add('active');

            if (target === 'tab-model') {
                loadModelStats();
            }
        });
    });

    // --- Fetch Presets ---
    async function initPresets() {
        try {
            const res = await fetch('/api/presets');
            if (res.ok) {
                currentPresets = await res.json();
                renderPresetChips();
            }
        } catch (err) {
            console.error('Failed to load presets:', err);
        }
    }

    function renderPresetChips() {
        if (!presetChipsContainer) return;
        presetChipsContainer.innerHTML = '';
        
        const selectedDomain = domainSelect.value;
        let categoryIndex = 0;

        if (selectedDomain === 'product') categoryIndex = 0;
        else if (selectedDomain === 'social') categoryIndex = 1;
        else if (selectedDomain === 'support') categoryIndex = 2;

        const category = currentPresets[categoryIndex] || currentPresets[0];
        if (!category) return;

        category.samples.forEach(sample => {
            const chip = document.createElement('button');
            chip.className = 'chip';
            chip.innerHTML = `<span>${getEmojiForSentiment(sample.expected)}</span> ${sample.title}`;
            chip.addEventListener('click', () => {
                liveInput.value = sample.text;
                updateCharCounter();
                performLiveAnalysis();
            });
            presetChipsContainer.appendChild(chip);
        });
    }

    function getEmojiForSentiment(sentiment) {
        if (sentiment === 'positive') return '😍';
        if (sentiment === 'negative') return '😡';
        return '😐';
    }

    domainSelect.addEventListener('change', renderPresetChips);

    // --- Text Input Listeners ---
    liveInput.addEventListener('input', () => {
        updateCharCounter();
        if (realtimeToggle.checked) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(performLiveAnalysis, 300);
        }
    });

    clearBtn.addEventListener('click', () => {
        liveInput.value = '';
        updateCharCounter();
        resetDiagnosisDisplay();
    });

    analyzeBtn.addEventListener('click', performLiveAnalysis);

    function updateCharCounter() {
        const count = liveInput.value.length;
        charCounter.textContent = `${count} chars`;
    }

    // --- Perform Live Analysis ---
    async function performLiveAnalysis() {
        const text = liveInput.value.trim();
        if (!text) {
            resetDiagnosisDisplay();
            return;
        }

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    domain: domainSelect.value
                })
            });

            if (response.ok) {
                const data = await response.json();
                renderDiagnosis(data);
            }
        } catch (err) {
            console.error('Analysis request error:', err);
        }
    }

    function renderDiagnosis(data) {
        // Update display card background state
        resultDisplay.className = `result-display ${data.sentiment}`;

        // Emoji & Badge
        sentimentEmoji.textContent = getEmojiForSentiment(data.sentiment);
        sentimentLabel.textContent = data.sentiment.toUpperCase();
        confidenceScore.textContent = `${data.confidence}% Confidence`;

        // Progress bars
        probPosVal.textContent = `${data.probabilities.positive}%`;
        probNeuVal.textContent = `${data.probabilities.neutral}%`;
        probNegVal.textContent = `${data.probabilities.negative}%`;

        probPosBar.style.width = `${data.probabilities.positive}%`;
        probNeuBar.style.width = `${data.probabilities.neutral}%`;
        probNegBar.style.width = `${data.probabilities.negative}%`;

        // Polarity Score (-1 to 1 mapped to 0% to 100%)
        polarityVal.textContent = data.polarity > 0 ? `+${data.polarity}` : data.polarity;
        const dotPos = ((data.polarity + 1) / 2) * 100;
        polarityDot.style.left = `${Math.min(Math.max(dotPos, 5), 95)}%`;

        // Subjectivity (0 to 1 mapped to 0% to 100%)
        subjectivityVal.textContent = data.subjectivity;
        subjectivityFill.style.width = `${data.subjectivity * 100}%`;

        // Token Highlighter
        renderTokenHighlights(data.tokens);

        // Save last result payload for JSON copy
        resultDisplay.dataset.payload = JSON.stringify(data, null, 2);
    }

    function renderTokenHighlights(tokens) {
        if (!tokens || tokens.length === 0) {
            highlightedTextBox.innerHTML = '<span class="placeholder-text">Enter text to highlight sentiment drivers...</span>';
            return;
        }

        let html = '';
        tokens.forEach(t => {
            if (t.sentiment === 'positive') {
                html += `<span class="token-pill positive" title="Positive driver (score: +${t.score})">${t.word}</span> `;
            } else if (t.sentiment === 'negative') {
                html += `<span class="token-pill negative" title="Negative driver (score: ${t.score})">${t.word}</span> `;
            } else {
                html += `<span class="token-pill neutral">${t.word}</span> `;
            }
        });

        highlightedTextBox.innerHTML = html;
    }

    function resetDiagnosisDisplay() {
        resultDisplay.className = 'result-display neutral';
        sentimentEmoji.textContent = '😐';
        sentimentLabel.textContent = 'NEUTRAL';
        confidenceScore.textContent = '0.0% Confidence';

        probPosVal.textContent = '0.0%';
        probNeuVal.textContent = '0.0%';
        probNegVal.textContent = '0.0%';
        probPosBar.style.width = '0%';
        probNeuBar.style.width = '0%';
        probNegBar.style.width = '0%';

        polarityVal.textContent = '0.00';
        polarityDot.style.left = '50%';
        subjectivityVal.textContent = '0.00';
        subjectivityFill.style.width = '0%';

        highlightedTextBox.innerHTML = '<span class="placeholder-text">Enter text to highlight positive and negative sentiment drivers...</span>';
    }

    copyJsonBtn.addEventListener('click', () => {
        const payload = resultDisplay.dataset.payload;
        if (payload) {
            navigator.clipboard.writeText(payload);
            copyJsonBtn.innerHTML = `✓ Copied!`;
            setTimeout(() => {
                copyJsonBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Export JSON`;
            }, 2000);
        }
    });

    // --- BATCH CSV ANALYSIS ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    csvFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    loadSampleCsvBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/sample-csv');
            const blob = await res.blob();
            const file = new File([blob], 'sample_reviews.csv', { type: 'text/csv' });
            handleFileUpload(file);
        } catch (err) {
            console.error('Failed to load sample dataset:', err);
        }
    });

    async function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            dropZone.querySelector('h3').textContent = `Processing ${file.name}...`;
            const res = await fetch('/api/batch-upload', {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                batchData = await res.json();
                renderBatchDashboard(batchData);
            } else {
                alert('Failed to process uploaded file.');
            }
        } catch (err) {
            console.error('File upload error:', err);
        } finally {
            dropZone.querySelector('h3').textContent = 'Drag & Drop CSV / JSON File Here';
        }
    }

    function renderBatchDashboard(data) {
        batchResultsWrapper.classList.remove('hidden');

        // Summary cards
        document.getElementById('batch-total-count').textContent = data.total_items;
        document.getElementById('batch-pos-count').textContent = `${data.summary.counts.positive} (${data.summary.percentages.positive}%)`;
        document.getElementById('batch-neu-count').textContent = `${data.summary.counts.neutral} (${data.summary.percentages.neutral}%)`;
        document.getElementById('batch-neg-count').textContent = `${data.summary.counts.negative} (${data.summary.percentages.negative}%)`;
        document.getElementById('batch-avg-conf').textContent = `${data.summary.average_confidence}%`;

        // Render Charts
        renderBatchDoughnutChart(data.summary.counts);
        renderBatchBarChart(data.items);

        // Render Table
        renderBatchTable(data.items);
    }

    function renderBatchDoughnutChart(counts) {
        const ctx = document.getElementById('batchDoughnutChart').getContext('2d');
        if (doughnutChartInstance) doughnutChartInstance.destroy();

        doughnutChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [counts.positive, counts.neutral, counts.negative],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 0,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#9ca3af', font: { family: 'Outfit' } }
                    }
                }
            }
        });
    }

    function renderBatchBarChart(items) {
        const ctx = document.getElementById('batchBarChart').getContext('2d');
        if (barChartInstance) barChartInstance.destroy();

        // Bin confidence scores into buckets
        const buckets = { '50-60%': 0, '60-70%': 0, '70-80%': 0, '80-90%': 0, '90-100%': 0 };
        items.forEach(item => {
            const conf = item.confidence;
            if (conf >= 90) buckets['90-100%']++;
            else if (conf >= 80) buckets['80-90%']++;
            else if (conf >= 70) buckets['70-80%']++;
            else if (conf >= 60) buckets['60-70%']++;
            else buckets['50-60%']++;
        });

        barChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(buckets),
                datasets: [{
                    label: 'Items in Confidence Range',
                    data: Object.values(buckets),
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#9ca3af' }, grid: { display: false } },
                    y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    function renderBatchTable(items) {
        batchTableBody.innerHTML = '';
        const search = tableSearch.value.toLowerCase();
        const filter = tableFilter.value;

        let visibleCount = 0;

        items.forEach((item, index) => {
            const textMatch = item.text.toLowerCase().includes(search);
            const filterMatch = (filter === 'all') || (item.sentiment === filter);

            if (textMatch && filterMatch) {
                visibleCount++;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${item.id || index + 1}</td>
                    <td>${escapeHtml(item.text)}</td>
                    <td><span class="sentiment-tag ${item.sentiment}">${item.sentiment}</span></td>
                    <td><strong>${item.confidence}%</strong></td>
                    <td>${item.polarity}</td>
                    <td>${item.subjectivity}</td>
                `;
                batchTableBody.appendChild(tr);
            }
        });

        tableShowingCount.textContent = `Showing ${visibleCount} of ${items.length} items`;
    }

    tableSearch.addEventListener('input', () => {
        if (batchData) renderBatchTable(batchData.items);
    });

    tableFilter.addEventListener('change', () => {
        if (batchData) renderBatchTable(batchData.items);
    });

    // --- MODEL ANALYTICS & STATS ---
    async function loadModelStats() {
        try {
            const res = await fetch('/api/model/stats');
            if (res.ok) {
                const stats = await res.json();
                renderModelMetrics(stats);
            }
        } catch (err) {
            console.error('Failed to load model stats:', err);
        }
    }

    function renderModelMetrics(stats) {
        document.getElementById('m-acc').textContent = `${stats.accuracy}%`;
        document.getElementById('m-prec').textContent = `${stats.precision}%`;
        document.getElementById('m-rec').textContent = `${stats.recall}%`;
        document.getElementById('m-f1').textContent = `${stats.f1_score}%`;
        document.getElementById('m-samples').textContent = stats.sample_count;

        // Render 3x3 Confusion Matrix
        const cmGrid = document.getElementById('cm-grid');
        cmGrid.innerHTML = '';

        if (stats.confusion_matrix) {
            stats.confusion_matrix.forEach((row, rIdx) => {
                row.forEach((val, cIdx) => {
                    const cell = document.createElement('div');
                    cell.className = `cm-cell ${rIdx === cIdx ? 'diagonal' : ''}`;
                    cell.innerHTML = `
                        <span class="cm-val">${val}</span>
                        <span style="font-size: 0.7rem; color: #9ca3af;">(${rIdx === cIdx ? 'Correct' : 'Misclassified'})</span>
                    `;
                    cmGrid.appendChild(cell);
                });
            });
        }

        // Top Positive & Negative Feature Chips
        const posChipsBox = document.getElementById('top-pos-features');
        const negChipsBox = document.getElementById('top-neg-features');
        posChipsBox.innerHTML = '';
        negChipsBox.innerHTML = '';

        (stats.top_positive_features || []).forEach(f => {
            const chip = document.createElement('span');
            chip.className = 'f-chip pos';
            chip.textContent = `${f.word} (+${f.weight})`;
            posChipsBox.appendChild(chip);
        });

        (stats.top_negative_features || []).forEach(f => {
            const chip = document.createElement('span');
            chip.className = 'f-chip neg';
            chip.textContent = `${f.word} (${f.weight})`;
            negChipsBox.appendChild(chip);
        });
    }

    retrainBtn.addEventListener('click', async () => {
        retrainBtn.textContent = 'Retraining...';
        try {
            const res = await fetch('/api/model/train', { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                renderModelMetrics(data.metrics);
                alert('Model successfully retrained with latest features!');
            }
        } catch (err) {
            console.error('Retrain failed:', err);
        } finally {
            retrainBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg> Retrain Model`;
        }
    });

    // --- API PLAYGROUND ---
    sendApiBtn.addEventListener('click', async () => {
        try {
            const bodyJson = JSON.parse(apiRequestBody.value);
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(bodyJson)
            });
            const data = await res.json();
            apiResponseCode.textContent = JSON.stringify(data, null, 2);
        } catch (err) {
            apiResponseCode.textContent = `Error: ${err.message}`;
        }
    });

    codeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            codeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const lang = tab.getAttribute('data-lang');
            if (lang === 'python') {
                codeSnippetBox.textContent = `import requests\n\nurl = "http://localhost:8000/api/analyze"\npayload = {\n    "text": "Great product, fast shipping!",\n    "domain": "product"\n}\nresponse = requests.post(url, json=payload)\nprint(response.json())`;
            } else if (lang === 'javascript') {
                codeSnippetBox.textContent = `fetch("http://localhost:8000/api/analyze", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify({\n    text: "Great product, fast shipping!",\n    domain: "product"\n  })\n})\n.then(res => res.json())\n.then(data => console.log(data));`;
            } else if (lang === 'curl') {
                codeSnippetBox.textContent = `curl -X POST "http://localhost:8000/api/analyze" \\\n  -H "Content-Type: application/json" \\\n  -d '{"text": "Great product, fast shipping!", "domain": "product"}'`;
            }
        });
    });

    // --- Helper ---
    function escapeHtml(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Initialize App
    initPresets();
    updateCharCounter();
    performLiveAnalysis();
});
