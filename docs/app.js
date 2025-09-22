(function initializeApp() {
    const decisionsUrl = 'data/scifi-ethical-dilemmas-with-decisions.json';
    const questionsUrl = 'data/scifi-ethical-dilemmas.json';

    /** DOM elements */
    const authorFilterSelect = document.getElementById('authorFilter');
    const sourceSearchInput = document.getElementById('sourceSearch');
    const resetButton = document.getElementById('resetBtn');
    const resultCountSpan = document.getElementById('resultCount');
    const itemsListDiv = document.getElementById('itemsList');
    const detailPanel = document.getElementById('detailPanel');

    /** State */
    let allDecisions = [];
    let allQuestions = [];
    let currentFilterAuthor = '';
    let currentSearchSource = '';
    let lastRenderedItems = [];

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

    /** Data loading */
    async function loadData() {
        const [decisionsRes, questionsRes] = await Promise.all([
            fetch(decisionsUrl),
            fetch(questionsUrl)
        ]);

        if (!decisionsRes.ok) throw new Error('Failed to load decisions JSON');
        // Questions JSON is optional for this UI; if missing, continue gracefully
        let qJson = [];
        try {
            if (questionsRes.ok) {
                qJson = await questionsRes.json();
            }
        } catch (_) { /* ignore */ }

        const decisionsJson = await decisionsRes.json();
        return { decisionsJson, qJson };
    }

    function buildAuthorList(items) {
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

    function applyFilters() {
        const filtered = allDecisions.filter(item => {
            const authorMatches = currentFilterAuthor ? (item.author || 'Unknown') === currentFilterAuthor : true;
            const sourceMatches = currentSearchSource ? normalizeString(item.source).includes(normalizeString(currentSearchSource)) : true;
            return authorMatches && sourceMatches;
        });
        return filtered;
    }

    function clearDetailPanel() {
        detailPanel.innerHTML = '<div class="empty">Select a dilemma to see details.</div>';
    }

    function renderList(items) {
        lastRenderedItems = items;
        itemsListDiv.innerHTML = '';
        const container = document.createElement('div');
        container.className = 'grid';

        if (items.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'muted';
            empty.textContent = 'No results. Try adjusting filters.';
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
        detailPanel.innerHTML = '';

        const title = document.createElement('h2');
        title.textContent = item.source || 'Untitled source';
        detailPanel.appendChild(title);

        const author = document.createElement('div');
        author.className = 'muted';
        author.textContent = `Author: ${item.author || 'Unknown'}`;
        detailPanel.appendChild(author);

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
        resultCountSpan.textContent = `Showing ${shown} of ${total}`;
    }

    function refresh() {
        setBusy(true);
        const filtered = applyFilters();
        renderList(filtered);
        updateResultCount(filtered.length, allDecisions.length);
        setBusy(false);
        if (filtered.length > 0) {
            renderDetail(filtered[0]);
        } else {
            clearDetailPanel();
        }
    }

    /** Event bindings */
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

    /** Init */
    (async function init() {
        try {
            setBusy(true);
            const { decisionsJson, qJson } = await loadData();
            allDecisions = Array.isArray(decisionsJson) ? decisionsJson : [];
            allQuestions = Array.isArray(qJson) ? qJson : [];

            buildAuthorList(allDecisions);
            refresh();
        } catch (err) {
            console.error(err);
            itemsListDiv.innerHTML = '<div class="muted">Failed to load data.</div>';
            updateResultCount(0, 0);
            clearDetailPanel();
        } finally {
            setBusy(false);
        }
    })();
})();


