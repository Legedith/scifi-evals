(function initializeApp() {
    // Available models configuration
    const MODELS = {
        'gpt-4-decisions': {
            name: 'GPT-4 Decisions (Short)',
            file: '../data/enhanced/dilemmas_with_gpt4_decisions.json',
            provider: 'OpenAI',
            isDefault: true
        },
        'gpt-5-nano': {
            name: 'GPT-5 Nano',
            file: '../data/responses/gpt-5-nano_responses.json',
            provider: 'OpenAI'
        },
        'grok-4-fast': {
            name: 'Grok 4 Fast',
            file: '../data/responses/grok-4-fast_responses.json',
            provider: 'xAI'
        },
        'gemma-3-27b': {
            name: 'Gemma 3 27B',
            file: '../data/responses/gemma-3-27b_responses.json',
            provider: 'Google'
        },
        'nemotron-nano-9b': {
            name: 'Nemotron Nano 9B',
            file: '../data/responses/nemotron-nano-9b_responses.json',
            provider: 'NVIDIA'
        },
        'deepseek-chat-v3.1': {
            name: 'DeepSeek Chat v3.1',
            file: '../data/responses/deepseek-chat-v3.1_responses.json',
            provider: 'DeepSeek'
        },
        'kimi-k2': {
            name: 'Kimi K2',
            file: '../data/responses/kimi-k2_responses.json',
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

        const response = await fetch(model.file);
        if (!response.ok) throw new Error(`Failed to load ${model.name} data`);
        
        const data = await response.json();
        allModelData[modelKey] = Array.isArray(data) ? data : [];
        return allModelData[modelKey];
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
            compareModal.style.display = 'flex';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
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
        
        const question = document.createElement('div');
        question.className = 'question';
        question.textContent = dilemma.question || '';
        header.appendChild(question);
        
        compareContent.appendChild(header);

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
            
            // Add responses for this section
            for (const [modelKey, modelConfig] of Object.entries(MODELS)) {
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