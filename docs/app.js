(function initializeApp() {
    // Available models configuration
    const MODELS = {
        'gpt-5-nano': {
            name: 'GPT-5 Nano',
            file: 'data/responses/gpt-5-nano_responses.json',
            provider: 'OpenAI',
            isDefault: true
        },
        'grok-4-fast': {
            name: 'Grok 4 Fast',
            file: 'data/responses/grok-4-fast_responses.json',
            provider: 'xAI'
        },
        'gemma-3-27b': {
            name: 'Gemma 3 27B',
            file: 'data/responses/gemma-3-27b_responses.json',
            provider: 'Google'
        },
        'nemotron-nano-9b': {
            name: 'Nemotron Nano 9B',
            file: 'data/responses/nemotron-nano-9b_responses.json',
            provider: 'NVIDIA'
        },
        'deepseek-chat-v3.1': {
            name: 'DeepSeek Chat v3.1',
            file: 'data/responses/deepseek-chat-v3.1_responses.json',
            provider: 'DeepSeek'
        },
        'kimi-k2': {
            name: 'Kimi K2',
            file: 'data/responses/kimi-k2_responses.json',
            provider: 'Moonshot AI'
        }
    };

    /** DOM elements */
    const modelSelect = document.getElementById('modelSelect');
    const authorFilterSelect = document.getElementById('authorFilter');
    const sourceSearchInput = document.getElementById('sourceSearch');
    const resetButton = document.getElementById('resetBtn');
    const compareButton = document.getElementById('compareBtn');
    const downloadButton = document.getElementById('downloadBtn');
    const resultCountSpan = document.getElementById('resultCount');
    const itemsListDiv = document.getElementById('itemsList');
    const detailPanel = document.getElementById('detailPanel');
    const listContainer = document.getElementById('listContainer');
    const toggleListBtn = document.getElementById('toggleListBtn');

    // Compare modal elements
    const compareModal = document.getElementById('compareModal');
    const closeModal = document.getElementById('closeModal');
    const compareSelect = document.getElementById('compareSelect');
    const compareContent = document.getElementById('compareContent');

    /** State */
    let allModelData = {}; // Cache for all loaded model data
    let currentModel = '';
    let currentData = [];
    let currentFilterAuthor = '';
    let currentSearchSource = '';
    let lastRenderedItems = [];
    let isListCollapsed = false;
    let currentDetailItem = null;
    // Similarity/compare state
    const SIM_MODELS = Object.keys(MODELS);
    let perDilemmaSim = null;
    let currentCompareIndex = null;
    let activeGraphStop = null; // cancel current graph animation if re-rendering
    const simState = {
        kind: 'body',
        threshold: 0.75,
        heatmap: false,
        weights: { body: 0.6, in_favor: 0.2, against: 0.2 },
        _autoAppliedForIndex: null
    };

    /** Utilities */
    function normalizeString(value) {
        return (value || '').toString().trim().toLowerCase();
    }

    function uniqueSorted(values) {
        return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b));
    }

    function setBusy(isBusy) {
        itemsListDiv.setAttribute('aria-busy', isBusy ? 'true' : 'false');
    }

    /** Model management */
    function buildModelList() {
        const frag = document.createDocumentFragment();
        let defaultModel = null;

        for (const [key, model] of Object.entries(MODELS)) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = `${model.name} (${model.provider})`;
            frag.appendChild(option);

            if (model.isDefault) {
                defaultModel = key;
            }
        }

        modelSelect.appendChild(frag);

        // Set default model
        if (defaultModel) {
            modelSelect.value = defaultModel;
            setTimeout(() => switchModel(defaultModel), 100);
        }
    }

    /** List collapse functionality */
    function toggleList() {
        isListCollapsed = !isListCollapsed;
        listContainer.classList.toggle('collapsed', isListCollapsed);
        const icon = toggleListBtn.querySelector('.toggle-icon');
        icon.textContent = isListCollapsed ? '›' : '‹';
        toggleListBtn.setAttribute('aria-label', isListCollapsed ? 'Expand dilemma list' : 'Collapse dilemma list');
    }

    async function loadModelData(modelKey) {
        if (allModelData[modelKey]) {
            return allModelData[modelKey];
        }

        const model = MODELS[modelKey];
        if (!model) throw new Error(`Unknown model: ${modelKey}`);

        const basePath = model.file;
        const candidatesSet = new Set([
            basePath,
            `./${basePath}`,
            `docs/${basePath}`,
            `../${basePath}`
        ]);
        const candidates = Array.from(candidatesSet);

        let lastError = null;
        for (const candidate of candidates) {
            const url = `${candidate}${candidate.includes('?') ? '&' : '?'}v=${Date.now()}`;
            try {
                const response = await fetch(url);
                if (response.ok) {
                    const data = await response.json();
                    allModelData[modelKey] = Array.isArray(data) ? data : [];
                    return allModelData[modelKey];
                }
                lastError = new Error(`HTTP ${response.status}`);
            } catch (err) {
                lastError = err;
            }
        }

        throw new Error(`Failed to load ${model.name} data${lastError ? `: ${lastError.message}` : ''}`);
    }

    async function switchModel(modelKey) {
        if (!modelKey) {
            currentModel = '';
            currentData = [];
            buildAuthorList([]);
            renderList([]);
            updateResultCount(0, 0);
            clearDetailPanel();
            downloadButton.disabled = true;
            resultCountSpan.textContent = 'Select a model to begin...';
            return;
        }

        try {
            setBusy(true);
            currentData = await loadModelData(modelKey);
            currentModel = modelKey;

            buildAuthorList(currentData);
            buildCompareList(currentData);
            downloadButton.disabled = false;
            updateDownloadButton();
            refresh();
        } catch (err) {
            console.error('Failed to load model data:', err);
            itemsListDiv.innerHTML = `<div class="muted">Failed to load ${MODELS[modelKey]?.name || modelKey} data.</div>`;
            currentData = [];
            updateResultCount(0, 0);
            clearDetailPanel();
        } finally {
            setBusy(false);
        }
    }

    function buildAuthorList(items) {
        // Clear existing options except the first one
        authorFilterSelect.innerHTML = '<option value="">All authors</option>';

        if (items.length === 0) return;

        const authors = uniqueSorted(items.map(item => item.author || 'Unknown'));
        const frag = document.createDocumentFragment();
        for (const author of authors) {
            const option = document.createElement('option');
            option.value = author;
            option.textContent = author;
            frag.appendChild(option);
        }
        authorFilterSelect.appendChild(frag);
    }

    function buildCompareList(items) {
        compareSelect.innerHTML = '<option value="">Choose a dilemma...</option>';

        if (items.length === 0) return;

        const frag = document.createDocumentFragment();
        items.forEach((item, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${item.source || 'Untitled'} (${item.author || 'Unknown'})`;
            frag.appendChild(option);
        });
        compareSelect.appendChild(frag);
    }

    function updateDownloadButton() {
        if (currentModel && MODELS[currentModel]) {
            downloadButton.textContent = `Download ${MODELS[currentModel].name} Data`;
            downloadButton.onclick = () => {
                const dataStr = JSON.stringify(currentData, null, 2);
                const blob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${currentModel}_responses.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            };
        }
    }

    /** Filtering and display */
    function applyFilters() {
        if (!currentData.length) return [];

        const filtered = currentData.filter(item => {
            const authorMatches = currentFilterAuthor ? (item.author || 'Unknown') === currentFilterAuthor : true;
            const sourceMatches = currentSearchSource ? normalizeString(item.source).includes(normalizeString(currentSearchSource)) : true;
            return authorMatches && sourceMatches;
        });
        return filtered;
    }

    function clearDetailPanel() {
        detailPanel.innerHTML = '<div class="empty">Select a model and dilemma to see details.</div>';
    }

    function renderList(items) {
        lastRenderedItems = items;
        itemsListDiv.innerHTML = '';
        const container = document.createElement('div');
        container.className = 'grid';

        if (items.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'muted';
            empty.textContent = currentModel ? 'No results. Try adjusting filters.' : 'Select a model to view dilemmas.';
            itemsListDiv.appendChild(empty);
            return;
        }

        for (const item of items) {
            const card = document.createElement('article');
            card.className = 'card';
            card.tabIndex = 0;
            card.setAttribute('role', 'button');
            card.setAttribute('aria-label', `View details for ${item.source}`);

            const sourceEl = document.createElement('div');
            sourceEl.className = 'source';
            sourceEl.textContent = item.source || 'Untitled source';
            card.appendChild(sourceEl);

            const authorEl = document.createElement('div');
            authorEl.className = 'author';
            authorEl.textContent = item.author || 'Unknown';
            card.appendChild(authorEl);

            // Add model indicator
            const modelEl = document.createElement('div');
            modelEl.className = 'model-indicator';
            modelEl.textContent = MODELS[currentModel]?.name || currentModel;
            card.appendChild(modelEl);

            card.addEventListener('click', () => renderDetail(item));
            card.addEventListener('keypress', (ev) => {
                if (ev.key === 'Enter' || ev.key === ' ') {
                    ev.preventDefault();
                    renderDetail(item);
                }
            });

            container.appendChild(card);
        }

        itemsListDiv.appendChild(container);
    }

    function createList(items) {
        const ul = document.createElement('ul');
        ul.className = 'inline-list';
        for (const text of items) {
            const li = document.createElement('li');
            li.textContent = text;
            ul.appendChild(li);
        }
        return ul;
    }

    function renderDetail(item) {
        currentDetailItem = item;
        detailPanel.innerHTML = '';

        // Header with model info and switcher
        const header = document.createElement('div');
        header.className = 'detail-header';

        const titleRow = document.createElement('div');
        titleRow.className = 'title-row';

        const title = document.createElement('h2');
        title.textContent = item.source || 'Untitled source';
        titleRow.appendChild(title);

        // Model switcher
        const modelSwitcher = document.createElement('div');
        modelSwitcher.className = 'model-switcher';

        const switcherLabel = document.createElement('label');
        switcherLabel.textContent = 'View response from:';
        switcherLabel.className = 'switcher-label';
        modelSwitcher.appendChild(switcherLabel);

        const switcherSelect = document.createElement('select');
        switcherSelect.className = 'switcher-select';
        switcherSelect.id = 'detailModelSelect';

        // Populate switcher with all models
        for (const [key, model] of Object.entries(MODELS)) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = model.name;
            option.selected = key === currentModel;
            switcherSelect.appendChild(option);
        }

        switcherSelect.addEventListener('change', async (e) => {
            const newModel = e.target.value;
            if (newModel !== currentModel) {
                await switchToModelForDetail(newModel, item);
            }
        });

        modelSwitcher.appendChild(switcherSelect);
        titleRow.appendChild(modelSwitcher);
        header.appendChild(titleRow);

        const modelInfo = document.createElement('div');
        modelInfo.className = 'model-info';
        modelInfo.textContent = `${MODELS[currentModel]?.name || currentModel} (${MODELS[currentModel]?.provider || ''})`;
        header.appendChild(modelInfo);

        const author = document.createElement('div');
        author.className = 'muted';
        author.textContent = `Author: ${item.author || 'Unknown'}`;
        header.appendChild(author);

        detailPanel.appendChild(header);

        const qTitle = document.createElement('div');
        qTitle.className = 'section-title';
        qTitle.textContent = 'Question';
        detailPanel.appendChild(qTitle);

        const q = document.createElement('p');
        q.textContent = item.question || '';
        detailPanel.appendChild(q);

        const decisionTitle = document.createElement('div');
        decisionTitle.className = 'section-title';
        decisionTitle.textContent = 'Decision';
        detailPanel.appendChild(decisionTitle);

        const decisionText = document.createElement('p');
        decisionText.textContent = item?.llm_decision?.decision || '—';
        detailPanel.appendChild(decisionText);

        const considerations = item?.llm_decision?.considerations || {};
        const inFavor = Array.isArray(considerations.in_favor) ? considerations.in_favor : [];
        const against = Array.isArray(considerations.against) ? considerations.against : [];

        const consTitle = document.createElement('div');
        consTitle.className = 'section-title';
        consTitle.textContent = 'Considerations';
        detailPanel.appendChild(consTitle);

        if (inFavor.length > 0) {
            const favorLabel = document.createElement('div');
            favorLabel.textContent = 'In favor';
            detailPanel.appendChild(favorLabel);
            detailPanel.appendChild(createList(inFavor));
        }
        if (against.length > 0) {
            const againstLabel = document.createElement('div');
            againstLabel.style.marginTop = '8px';
            againstLabel.textContent = 'Against';
            detailPanel.appendChild(againstLabel);
            detailPanel.appendChild(createList(against));
        }

        const reasoningTitle = document.createElement('div');
        reasoningTitle.className = 'section-title';
        reasoningTitle.textContent = 'Reasoning';
        detailPanel.appendChild(reasoningTitle);

        const reasoningText = document.createElement('p');
        reasoningText.textContent = item?.llm_decision?.reasoning || '—';
        detailPanel.appendChild(reasoningText);
    }

    function updateResultCount(shown, total) {
        if (total === 0) {
            resultCountSpan.textContent = currentModel ? 'No data available' : 'Select a model to begin...';
        } else {
            resultCountSpan.textContent = `Showing ${shown} of ${total}`;
        }
    }

    function refresh() {
        setBusy(true);
        const filtered = applyFilters();
        renderList(filtered);
        updateResultCount(filtered.length, currentData.length);
        setBusy(false);
        if (filtered.length > 0) {
            renderDetail(filtered[0]);
        } else {
            clearDetailPanel();
        }
    }

    /** Model switching for detail view */
    async function switchToModelForDetail(newModelKey, item) {
        try {
            setBusy(true);
            const newData = await loadModelData(newModelKey);

            // Find the same dilemma in the new model's data
            const sameIndex = currentData.findIndex(d =>
                d.source === item.source && d.author === item.author
            );

            if (sameIndex !== -1 && newData[sameIndex]) {
                // Update detail view with new model's response
                renderDetailForModel(newData[sameIndex], newModelKey);
            } else {
                alert('This dilemma is not available in the selected model\'s dataset.');
            }
        } catch (err) {
            alert('Failed to load data for the selected model: ' + err.message);
        } finally {
            setBusy(false);
        }
    }

    function renderDetailForModel(item, modelKey) {
        // Similar to renderDetail but doesn't update currentModel globally
        const tempCurrentModel = currentModel;
        const tempItem = currentDetailItem;

        // Temporarily update for rendering
        currentModel = modelKey;
        currentDetailItem = item;

        renderDetail(item);

        // Restore if this was just for detail view
        if (modelKey !== tempCurrentModel) {
            currentModel = tempCurrentModel;
        }
    }

    /** Compare functionality */
    async function showCompareModal() {
        // Ensure we have data for all models
        const modelKeys = Object.keys(MODELS);
        try {
            setBusy(true);
            await Promise.all(modelKeys.map(key => loadModelData(key)));
            // Lazy-load per-dilemma similarities
            if (!perDilemmaSim) {
                const candidates = [
                    'data/per_dilemma_similarity.json',
                    './data/per_dilemma_similarity.json',
                    'docs/data/per_dilemma_similarity.json',
                    '../data/per_dilemma_similarity.json'
                ];
                let lastErr = null;
                for (const url of candidates) {
                    try {
                        const r = await fetch(`${url}?v=${Date.now()}`);
                        if (r.ok) { perDilemmaSim = await r.json(); break; }
                        lastErr = new Error(`HTTP ${r.status}`);
                    } catch (e) { lastErr = e; }
                }
                if (!perDilemmaSim) {
                    console.warn('Per-dilemma similarity JSON not found', lastErr);
                }
            }
            compareModal.style.display = 'flex';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
            // Auto-select the first dilemma by default
            if (compareSelect.options.length > 1) {
                compareSelect.selectedIndex = 1; // first real dilemma option
                renderComparison(compareSelect.value);
            }
        } catch (err) {
            alert('Failed to load comparison data: ' + err.message);
        } finally {
            setBusy(false);
        }
    }

    function hideCompareModal() {
        compareModal.style.display = 'none';
        document.body.style.overflow = ''; // Restore scrolling
        compareContent.innerHTML = '<div class="empty">Select a dilemma to compare responses across models.</div>';
        compareSelect.value = '';
    }

    function renderComparison(dilemmaIndex) {
        const index = parseInt(dilemmaIndex);
        if (isNaN(index)) return;

        currentCompareIndex = index;
        compareContent.innerHTML = '';

        // Get dilemma data from all models
        const modelKeys = Object.keys(MODELS);
        const responses = {};
        let dilemma = null;

        // Collect responses from all models
        for (const modelKey of modelKeys) {
            const modelData = allModelData[modelKey];
            if (modelData && modelData[index]) {
                responses[modelKey] = modelData[index];
                if (!dilemma) {
                    dilemma = modelData[index];
                }
            }
        }

        if (!dilemma) {
            compareContent.innerHTML = '<div class="muted">Dilemma not found.</div>';
            return;
        }

        // Dilemma header
        const header = document.createElement('div');
        header.className = 'compare-header';

        const title = document.createElement('h3');
        title.textContent = dilemma.source || 'Untitled';
        header.appendChild(title);

        const author = document.createElement('div');
        author.className = 'muted';
        author.textContent = `Author: ${dilemma.author || 'Unknown'}`;
        header.appendChild(author);

        // Dual questions: simplified (GPT-5 decisions) and enhanced (from enhanced dataset)
        const questionWrap = document.createElement('div');
        questionWrap.className = 'question-wrap';

        const simplifiedQ = document.createElement('div');
        simplifiedQ.className = 'question simplified';
        simplifiedQ.innerHTML = '<strong>Dilemma:</strong> ' + (dilemma.question || '');
        questionWrap.appendChild(simplifiedQ);

        // Try to pull enhanced version from enhanced_dilemmas.json if loaded/cached
        // Fallback: show only simplified
        const enhancedModelKey = 'gpt-5-nano'; // any full set model (687) shares same questions content; but enhanced text is in enhanced/enhanced_dilemmas.json
        const enhancedText = (function () {
            try {
                // We don't have enhanced_dilemmas loaded in the app; infer from another model's question string if longer
                const candidates = Object.keys(MODELS)
                    .filter(k => k !== 'gpt-5-decisions')
                    .map(k => (allModelData[k] && allModelData[k][index]) ? allModelData[k][index].question : null)
                    .filter(Boolean);
                if (candidates.length === 0) return null;
                // Choose the longest question as the enhanced one
                return candidates.sort((a, b) => b.length - a.length)[0];
            } catch (_) { return null; }
        })();

        if (enhancedText && enhancedText !== dilemma.question) {
            const enhancedQ = document.createElement('div');
            enhancedQ.className = 'question enhanced';
            enhancedQ.innerHTML = '<strong>Enhanced:</strong> ' + enhancedText;
            questionWrap.appendChild(enhancedQ);
        }

        header.appendChild(questionWrap);

        compareContent.appendChild(header);

        // Similarity controls and heatmap (optional)
        let orderedModelKeys = Object.keys(MODELS);
        if (perDilemmaSim && perDilemmaSim.items && perDilemmaSim.items[String(index)]) {
            const controls = document.createElement('div');
            controls.className = 'compare-sim-controls';
            controls.style.display = 'flex';
            controls.style.gap = '12px';
            controls.style.alignItems = 'center';
            controls.style.margin = '8px 0 12px 0';

            const kindLabel = document.createElement('label');
            kindLabel.textContent = 'Similarity:';
            const kindSelect = document.createElement('select');
            kindSelect.id = 'simKind';
            ;['body', 'in_favor', 'against', 'overall'].forEach(k => {
                const opt = document.createElement('option');
                opt.value = k; opt.textContent = (k === 'body' ? 'decision' : k.replace('_', ' '));
                if (k === simState.kind) opt.selected = true;
                kindSelect.appendChild(opt);
            });
            kindSelect.addEventListener('change', () => { simState.kind = kindSelect.value; renderComparison(String(currentCompareIndex)); });

            const thWrap = document.createElement('div'); thWrap.style.display = 'inline-flex'; thWrap.style.gap = '6px';
            const thLabel = document.createElement('label'); thLabel.textContent = 'Threshold';
            const thInput = document.createElement('input'); thInput.type = 'range'; thInput.min = '0.00'; thInput.max = '1.00'; thInput.step = '0.01'; thInput.value = String(simState.threshold);
            const thVal = document.createElement('span'); thVal.textContent = Number(simState.threshold).toFixed(2);
            thInput.addEventListener('input', () => { simState.threshold = parseFloat(thInput.value); thVal.textContent = simState.threshold.toFixed(2); });
            thInput.addEventListener('change', () => { renderComparison(String(currentCompareIndex)); });
            const autoBtn = document.createElement('button'); autoBtn.textContent = 'Auto'; autoBtn.className = 'secondary'; autoBtn.style.marginLeft = '6px';
            thWrap.appendChild(thLabel); thWrap.appendChild(thInput); thWrap.appendChild(thVal); thWrap.appendChild(autoBtn);

            controls.appendChild(kindLabel);
            controls.appendChild(kindSelect);
            controls.appendChild(thWrap);
            const stats = document.createElement('span'); stats.className = 'muted'; stats.style.marginLeft = '8px'; stats.id = 'clusterStats'; controls.appendChild(stats);
            compareContent.appendChild(controls);

            // Reconstruct similarity matrix and cluster ordering
            const itemObj = perDilemmaSim.items[String(index)];
            const uiModels = perDilemmaSim.models || orderedModelKeys;
            const N = uiModels.length;
            function triToSquare(tri) { const M = Array.from({ length: N }, () => new Array(N).fill(0)); let p = 0; for (let i = 0; i < N - 1; i++) { for (let j = i + 1; j < N; j++) { const v = tri[p++]; M[i][j] = v; M[j][i] = v; } } for (let i = 0; i < N; i++) M[i][i] = 1.0; return M; }
            function triMaskToSquare(tri) { const M = Array.from({ length: N }, () => new Array(N).fill(false)); let p = 0; for (let i = 0; i < N - 1; i++) { for (let j = i + 1; j < N; j++) { const v = !!tri[p++]; M[i][j] = v; M[j][i] = v; } } for (let i = 0; i < N; i++) M[i][i] = true; return M; }
            function combineOverall() {
                const W = simState.weights; const kinds = ['body', 'in_favor', 'against'];
                const sims = kinds.map(k => triToSquare(itemObj[k]?.tri || new Array((N * (N - 1)) / 2).fill(0)));
                const masks = kinds.map(k => triMaskToSquare(itemObj[k]?.mask || new Array((N * (N - 1)) / 2).fill(false)));
                const out = Array.from({ length: N }, () => new Array(N).fill(0));
                for (let i = 0; i < N; i++) { out[i][i] = 1.0; for (let j = i + 1; j < N; j++) { let num = 0, den = 0; if (masks[0][i][j]) { num += W.body * sims[0][i][j]; den += W.body; } if (masks[1][i][j]) { num += W.in_favor * sims[1][i][j]; den += W.in_favor; } if (masks[2][i][j]) { num += W.against * sims[2][i][j]; den += W.against; } const v = den > 0 ? (num / den) : sims[0][i][j]; out[i][j] = v; out[j][i] = v; } }
                return out;
            }
            let simMatrix = null; const kindKey = simState.kind;
            if (kindKey === 'overall') simMatrix = combineOverall();
            else if (itemObj[kindKey] && itemObj[kindKey].tri) simMatrix = triToSquare(itemObj[kindKey].tri);
            if (simMatrix) {
                let maskSq;
                if (kindKey === 'overall') {
                    const mk = ['body', 'in_favor', 'against'].map(k => triMaskToSquare(itemObj[k]?.mask || new Array((N * (N - 1)) / 2).fill(false)));
                    maskSq = Array.from({ length: N }, () => new Array(N).fill(false));
                    for (let i = 0; i < N; i++) { maskSq[i][i] = true; for (let j = i + 1; j < N; j++) { const m = mk[0][i][j] || mk[1][i][j] || mk[2][i][j]; maskSq[i][j] = m; maskSq[j][i] = m; } }
                } else { maskSq = triMaskToSquare(itemObj[kindKey].mask); }
                const th = simState.threshold;
                // Visual normalization domain from in-mask off-diagonal values (p10..p90)
                const vals = [];
                for (let i = 0; i < N; i++) { for (let j = i + 1; j < N; j++) { if (maskSq[i][j]) vals.push(simMatrix[i][j]); } }
                const sorted = vals.slice().sort((a, b) => a - b);
                const pct = (arr, p) => { if (arr.length === 0) return NaN; const k = Math.floor((arr.length - 1) * p); return arr[k]; };
                const vLo = pct(sorted, 0.10); const vHi = pct(sorted, 0.90);
                const mapVal = (v) => {
                    if (!(vHi > vLo + 1e-6)) return Math.max(0, Math.min(1, (v - 0.6) / 0.4));
                    return Math.max(0, Math.min(1, (v - vLo) / (vHi - vLo)));
                };
                const thNorm = mapVal(th);
                // Build mutual k-NN graph at current threshold to avoid a single giant component
                const MNN_K = 2;
                const top = Array.from({ length: N }, () => new Set());
                for (let i = 0; i < N; i++) {
                    const cand = [];
                    for (let j = 0; j < N; j++) {
                        if (i === j) continue;
                        if (!maskSq[i][j]) continue;
                        const v = simMatrix[i][j];
                        if (v >= th) cand.push([j, v]);
                    }
                    cand.sort((a, b) => b[1] - a[1]);
                    for (let t = 0; t < Math.min(MNN_K, cand.length); t++) top[i].add(cand[t][0]);
                }
                const parent = new Array(N).fill(0).map((_, i) => i);
                const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
                const unite = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
                const links = [];
                for (let i = 0; i < N; i++) {
                    for (let j = i + 1; j < N; j++) {
                        if (!maskSq[i][j]) continue;
                        const v = simMatrix[i][j];
                        if (v >= th && top[i].has(j) && top[j].has(i)) { unite(i, j); links.push({ i, j, w: v }); }
                    }
                }
                const groups = new Map(); for (let i = 0; i < N; i++) { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); }
                const clusters = Array.from(groups.values()).sort((a, b) => b.length - a.length);
                function sortWithinCluster(arr) { return arr.slice().sort((i, j) => { const si = arr.reduce((acc, k) => acc + (k === i ? 0 : simMatrix[i][k]), 0); const sj = arr.reduce((acc, k) => acc + (k === j ? 0 : simMatrix[j][k]), 0); return sj - si; }); }
                const orderIdx = []; clusters.forEach(c => orderIdx.push(...sortWithinCluster(c)));
                orderedModelKeys = orderIdx.map(i => uiModels[i]);
                const statsEl = document.getElementById('clusterStats'); if (statsEl) statsEl.textContent = `Clusters: ${clusters.length}`;

                // Auto-threshold finder: highest t with at least 2 clusters of size >= 2
                function findAutoThreshold() {
                    const vals = [];
                    for (let i = 0; i < N; i++) { for (let j = i + 1; j < N; j++) { if (maskSq[i][j]) vals.push(simMatrix[i][j]); } }
                    const uniq = Array.from(new Set(vals)).sort((a, b) => b - a);
                    const MNN_K = 2;
                    // helper to cluster at t
                    function clustersAt(t) {
                        const top = Array.from({ length: N }, () => new Set());
                        for (let i = 0; i < N; i++) {
                            const cand = [];
                            for (let j = 0; j < N; j++) { if (i === j) continue; if (!maskSq[i][j]) continue; const v = simMatrix[i][j]; if (v >= t) cand.push([j, v]); }
                            cand.sort((a, b) => b[1] - a[1]);
                            for (let m = 0; m < Math.min(MNN_K, cand.length); m++) top[i].add(cand[m][0]);
                        }
                        const parent = new Array(N).fill(0).map((_, i) => i);
                        const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
                        const unite = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
                        for (let i = 0; i < N; i++) { for (let j = i + 1; j < N; j++) { if (maskSq[i][j] && simMatrix[i][j] >= t && top[i].has(j) && top[j].has(i)) unite(i, j); } }
                        const groups = new Map(); for (let i = 0; i < N; i++) { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); }
                        return Array.from(groups.values());
                    }
                    for (const t of uniq) {
                        const cls = clustersAt(t);
                        let good = 0; for (const c of cls) if (c.length >= 2) good++;
                        if (good >= 2) return t;
                    }
                    // fallback: median
                    if (uniq.length) return uniq[Math.floor(uniq.length / 2)];
                    return simState.threshold;
                }
                autoBtn.onclick = () => { simState.threshold = findAutoThreshold(); renderComparison(String(currentCompareIndex)); };

                // Auto-apply when opening a dilemma (run once per index)
                if (simState._autoAppliedForIndex !== index) {
                    const autoT = findAutoThreshold();
                    // apply and re-render only if threshold meaningfully changes (> 0.005)
                    if (Math.abs((autoT ?? simState.threshold) - simState.threshold) > 0.005) {
                        simState._autoAppliedForIndex = index;
                        simState.threshold = autoT;
                        renderComparison(String(index));
                        return; // avoid duplicate rendering below
                    } else {
                        simState._autoAppliedForIndex = index;
                    }
                }
                // Visuals side-by-side: graph (left) and heatmap (right)
                const visWrap = document.createElement('div');
                visWrap.style.display = 'grid';
                visWrap.style.gap = '12px';
                visWrap.style.alignItems = 'start';
                visWrap.style.margin = '6px 0 10px 0';
                visWrap.style.gridTemplateColumns = 'minmax(220px, 520px) minmax(360px, 660px)';
                visWrap.style.justifyContent = 'center';
                visWrap.style.justifyItems = 'center';
                compareContent.appendChild(visWrap);

                // Graph
                if (activeGraphStop) { try { activeGraphStop(); } catch (_) { } activeGraphStop = null; }
                const graphWrap = document.createElement('div'); graphWrap.style.position = 'relative'; graphWrap.style.justifySelf = 'center';
                const canvas = document.createElement('canvas'); canvas.style.width = '100%'; canvas.style.display = 'block'; canvas.style.border = '1px solid #f0f0f0'; canvas.style.background = '#fff';
                graphWrap.appendChild(canvas); visWrap.appendChild(graphWrap);
                // Size canvas to container to avoid large empty padding
                const rectInit = graphWrap.getBoundingClientRect();
                const side = Math.max(220, Math.min(360, Math.floor((rectInit.width || 420) * 0.7)));
                canvas.width = side;
                canvas.height = side; // make graph square for denser layout
                const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height;
                // Nodes & links
                const nodes = [];
                const cx = W * 0.5, cy = H * 0.52; const R = Math.min(W, H) * 0.42;
                for (let i = 0; i < N; i++) { const ang = (i / N) * Math.PI * 2 - Math.PI / 2; nodes.push({ x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang), vx: 0, vy: 0 }); }

                // Tooltip
                const tip = document.createElement('div');
                tip.style.position = 'absolute'; tip.style.display = 'none'; tip.style.pointerEvents = 'none';
                tip.style.background = '#000'; tip.style.color = '#fff';
                tip.style.border = '1px solid #333'; tip.style.boxShadow = '0 2px 10px rgba(0,0,0,0.6)';
                tip.style.padding = '10px'; tip.style.fontSize = '12px'; tip.style.lineHeight = '1.4'; tip.style.maxWidth = '520px'; tip.style.zIndex = '5';
                tip.style.borderRadius = '4px'; tip.style.whiteSpace = 'pre-wrap';
                graphWrap.appendChild(tip);

                function draw() {
                    ctx.clearRect(0, 0, W, H);
                    // edges
                    for (const { i, j, w } of links) {
                        const a = nodes[i], b = nodes[j];
                        const wN = mapVal(w);
                        const alpha = Math.min(1, Math.max(0.15, (wN - thNorm) / Math.max(0.001, 1.0 - thNorm)));
                        ctx.strokeStyle = `rgba(60, 120, 200, ${alpha.toFixed(3)})`;
                        ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
                    }
                    // nodes
                    for (let i = 0; i < N; i++) {
                        const p = nodes[i];
                        ctx.fillStyle = '#3a7bd5'; // same color for all
                        ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI * 2); ctx.fill();
                        ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI * 2); ctx.stroke();
                        ctx.fillStyle = '#222'; ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                        ctx.fillText(uiModels[i], p.x, p.y + 10);
                    }
                }

                // Simple force simulation
                let running = true; let rafId = 0;
                function step() {
                    if (!running) return;
                    // forces
                    const kRep = 1200, kSpring = 0.02, kCenter = 0.008, damping = 0.9;
                    for (let i = 0; i < N; i++) {
                        let ax = 0, ay = 0;
                        // repulsion
                        for (let j = 0; j < N; j++) { if (i === j) continue; const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y; const d2 = dx * dx + dy * dy + 1e-3; const inv = 1 / Math.sqrt(d2); const f = kRep / d2; ax += (dx * inv) * f; ay += (dy * inv) * f; }
                        // springs
                        for (const { i: a, j: b, w } of links) {
                            const other = (a === i) ? b : (b === i) ? a : -1; if (other === -1) continue;
                            const dx = nodes[other].x - nodes[i].x, dy = nodes[other].y - nodes[i].y; const dist = Math.sqrt(dx * dx + dy * dy) + 1e-6; const nx = dx / dist, ny = dy / dist;
                            const wN = mapVal(w);
                            const L0 = 140 * (1 - 0.5 * (wN - thNorm) / Math.max(0.001, 1.0 - thNorm));
                            const f = kSpring * (dist - L0);
                            ax += nx * f; ay += ny * f;
                        }
                        // centering
                        ax += (cx - nodes[i].x) * kCenter; ay += (cy - nodes[i].y) * kCenter;
                        // integrate
                        nodes[i].vx = (nodes[i].vx + ax) * damping; nodes[i].vy = (nodes[i].vy + ay) * damping;
                    }
                    for (let i = 0; i < N; i++) { nodes[i].x += nodes[i].vx; nodes[i].y += nodes[i].vy; }
                    draw();
                    rafId = requestAnimationFrame(step);
                }
                draw(); rafId = requestAnimationFrame(step);
                activeGraphStop = () => { running = false; if (rafId) cancelAnimationFrame(rafId); };

                // Hover tooltip on nodes
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width; const scaleY = canvas.height / rect.height;
                    const mx = (e.clientX - rect.left) * scaleX; const my = (e.clientY - rect.top) * scaleY;
                    let best = -1, bestd2 = 1000;
                    for (let i = 0; i < N; i++) { const dx = nodes[i].x - mx, dy = nodes[i].y - my; const d2 = dx * dx + dy * dy; if (d2 < bestd2 && d2 <= (10 * 10)) { bestd2 = d2; best = i; } }
                    if (best >= 0) {
                        const key = uiModels[best]; const resp = responses[key]?.llm_decision || {};
                        const decision = (resp.decision || '—').toString();
                        tip.textContent = decision;
                        tip.style.left = `${(mx / scaleX) + 12}px`; tip.style.top = `${(my / scaleY) + 12}px`; tip.style.display = 'block';
                    } else { tip.style.display = 'none'; }
                });
                canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

                // Heatmap (right)
                if (true) {
                    const heatWrap = document.createElement('div'); heatWrap.style.width = '100%'; heatWrap.style.maxWidth = '660px'; heatWrap.style.overflow = 'hidden'; heatWrap.style.justifySelf = 'center';
                    visWrap.appendChild(heatWrap);
                    const rectH = heatWrap.getBoundingClientRect();
                    const labelW = 90; // row label column
                    const cellSize = Math.max(14, Math.min(32, Math.floor((rectH.width - labelW) / N)));
                    const table = document.createElement('table'); table.style.borderCollapse = 'collapse'; table.style.fontSize = '10px'; table.style.width = '100%';
                    const thead = document.createElement('thead'); const htr = document.createElement('tr');
                    const corner = document.createElement('th'); corner.textContent = ''; corner.style.width = labelW + 'px'; corner.style.padding = '0 4px'; htr.appendChild(corner);
                    for (let j = 0; j < N; j++) { const thd = document.createElement('th'); thd.style.padding = '0'; thd.style.width = cellSize + 'px'; thd.style.height = (cellSize + 12) + 'px'; thd.style.whiteSpace = 'nowrap'; thd.style.overflow = 'hidden'; thd.style.textOverflow = 'ellipsis'; thd.style.textAlign = 'center'; thd.textContent = uiModels[j]; thd.title = uiModels[j]; htr.appendChild(thd); }
                    thead.appendChild(htr);
                    const tbody = document.createElement('tbody');
                    for (let i = 0; i < N; i++) {
                        const tr = document.createElement('tr');
                        const rowLabel = document.createElement('th'); rowLabel.style.textAlign = 'right'; rowLabel.style.padding = '0 6px'; rowLabel.style.width = labelW + 'px'; rowLabel.textContent = uiModels[i]; tr.appendChild(rowLabel);
                        for (let j = 0; j < N; j++) {
                            const td = document.createElement('td'); td.style.width = cellSize + 'px'; td.style.height = cellSize + 'px'; td.style.padding = '0'; td.style.border = '1px solid #eee';
                            const v = simMatrix[i][j]; const c = Math.max(0, Math.min(255, Math.round(mapVal(v) * 255)));
                            td.style.backgroundColor = `rgb(${255 - c},${255 - Math.floor(c * 0.5)},255)`; td.title = `${uiModels[i]} vs ${uiModels[j]}: ${v.toFixed(2)}`;
                            tr.appendChild(td);
                        }
                        tbody.appendChild(tr);
                    }
                    table.appendChild(thead); table.appendChild(tbody); heatWrap.appendChild(table);
                }
            }
        }

        // Create collapsible sections
        const sections = ['decisions', 'in_favor', 'against', 'reasoning'];
        const sectionTitles = {
            decisions: 'Decisions',
            in_favor: 'Arguments In Favor',
            against: 'Arguments Against',
            reasoning: 'Reasoning'
        };

        for (const section of sections) {
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'compare-section';

            const sectionHeader = document.createElement('div');
            sectionHeader.className = 'compare-section-header';
            sectionHeader.innerHTML = `
                <h4>${sectionTitles[section]}</h4>
                <button class="section-toggle" aria-label="Toggle section">▼</button>
            `;

            const sectionContent = document.createElement('div');
            sectionContent.className = 'compare-section-content';

            // Add responses for this section, ordered if available
            const iterate = (orderedModelKeys && orderedModelKeys.length === Object.keys(MODELS).length)
                ? orderedModelKeys
                : Object.keys(MODELS);
            for (const modelKey of iterate) {
                const modelConfig = MODELS[modelKey];
                const response = responses[modelKey];
                const modelRow = document.createElement('div');
                modelRow.className = 'compare-model-row';

                const modelLabel = document.createElement('div');
                modelLabel.className = 'compare-model-label';
                modelLabel.textContent = `${modelConfig.name} (${modelConfig.provider})`;
                modelRow.appendChild(modelLabel);

                const modelContent = document.createElement('div');
                modelContent.className = 'compare-model-content';

                if (response && response.llm_decision) {
                    const decision = response.llm_decision;

                    if (section === 'decisions') {
                        modelContent.textContent = decision.decision || '—';
                    } else if (section === 'reasoning') {
                        modelContent.textContent = decision.reasoning || '—';
                    } else if (section === 'in_favor') {
                        const inFavor = decision.considerations?.in_favor || [];
                        if (inFavor.length > 0) {
                            modelContent.appendChild(createList(inFavor));
                        } else {
                            modelContent.textContent = 'No arguments listed';
                        }
                    } else if (section === 'against') {
                        const against = decision.considerations?.against || [];
                        if (against.length > 0) {
                            modelContent.appendChild(createList(against));
                        } else {
                            modelContent.textContent = 'No arguments listed';
                        }
                    }
                } else {
                    modelContent.textContent = 'No response available';
                    modelContent.className += ' muted';
                }

                modelRow.appendChild(modelContent);
                sectionContent.appendChild(modelRow);
            }

            // Add click handler for toggle
            sectionHeader.addEventListener('click', () => {
                sectionContent.classList.toggle('collapsed');
                const toggle = sectionHeader.querySelector('.section-toggle');
                toggle.textContent = sectionContent.classList.contains('collapsed') ? '▶' : '▼';
            });

            sectionDiv.appendChild(sectionHeader);
            sectionDiv.appendChild(sectionContent);
            compareContent.appendChild(sectionDiv);
        }
    }

    /** Event bindings */
    modelSelect.addEventListener('change', () => {
        switchModel(modelSelect.value);
    });

    authorFilterSelect.addEventListener('change', () => {
        currentFilterAuthor = authorFilterSelect.value;
        refresh();
    });

    sourceSearchInput.addEventListener('input', () => {
        currentSearchSource = sourceSearchInput.value;
        refresh();
    });

    resetButton.addEventListener('click', () => {
        currentFilterAuthor = '';
        currentSearchSource = '';
        authorFilterSelect.value = '';
        sourceSearchInput.value = '';
        refresh();
    });

    toggleListBtn.addEventListener('click', toggleList);

    compareButton.addEventListener('click', showCompareModal);
    closeModal.addEventListener('click', hideCompareModal);

    compareModal.addEventListener('click', (e) => {
        if (e.target === compareModal) {
            hideCompareModal();
        }
    });

    compareSelect.addEventListener('change', () => {
        if (compareSelect.value) {
            renderComparison(compareSelect.value);
        } else {
            compareContent.innerHTML = '<div class="empty">Select a dilemma to compare responses across models.</div>';
        }
    });

    /** Init */
    (async function init() {
        try {
            buildModelList();
            itemsListDiv.innerHTML = '<div class="muted">Select a model to view ethical dilemmas and responses.</div>';
        } catch (err) {
            console.error('Initialization error:', err);
            itemsListDiv.innerHTML = '<div class="muted">Failed to initialize application.</div>';
        }
    })();
})();