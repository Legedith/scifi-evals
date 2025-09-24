// Fullscreen embedding explorer with pan/zoom, threshold clustering, and k-means
(() => {
  const canvas = document.getElementById('stage');
  const tooltip = document.getElementById('tooltip');
  const thSlider = document.getElementById('threshold');
  const thVal = document.getElementById('thVal');
  const kSlider = document.getElementById('kSlider');
  const kVal = document.getElementById('kVal');
  const reseedBtn = document.getElementById('reseed');
  const statsEl = document.getElementById('stats');
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const thControls = document.getElementById('th-controls');
  const kmControls = document.getElementById('km-controls');

  const ctx = canvas.getContext('2d');
  let points = [];
  let vectors = null; // Float32Array flat, shape [N, D]
  let W = 0, H = 0;
  let radius = 2.0;
  let hoverIdx = -1;
  let transform = { x: 0, y: 0, s: 1.0 };
  let isPanning = false, lastX = 0, lastY = 0;
  let nnGraph = []; // neighbor list by idx
  let clusters = [];
  let clusterAssign = null; // Int32Array of cluster ids

  function resize() {
    W = canvas.clientWidth = window.innerWidth;
    H = canvas.clientHeight = window.innerHeight;
    canvas.width = W * devicePixelRatio;
    canvas.height = H * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    draw();
  }
  window.addEventListener('resize', resize);

  function screenToWorld(x, y) {
    return { x: (x - transform.x) / transform.s, y: (y - transform.y) / transform.s };
  }
  function worldToScreen(x, y) {
    return { x: x * transform.s + transform.x, y: y * transform.s + transform.y };
  }

  function setMode(mode) {
    if (mode === 'kmeans') {
      thControls.style.display = 'none';
      kmControls.style.display = 'inline-flex';
      runKMeans();
    } else {
      kmControls.style.display = 'none';
      thControls.style.display = 'inline-flex';
      clusters = []; clusterAssign = null; // will compute per draw using threshold
      draw();
    }
  }

  async function load() {
    const res = await fetch('data/embedding_points.json');
    const data = await res.json();
    points = data.points || [];
    // normalize input coords to unit square centered
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const p of points) { if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; }
    const sx = 1 / Math.max(1e-6, maxx - minx);
    const sy = 1 / Math.max(1e-6, maxy - miny);
    for (const p of points) { p.x = (p.x - minx) * sx; p.y = (p.y - miny) * sy; }
    for (const p of points) { p.x = (p.x - 0.5) * 2; p.y = (p.y - 0.5) * 2; }
    // neighbor graph
    nnGraph = points.map(p => p.sim || []);
    statsEl.textContent = `${points.length} points`;
    // initial view
    transform.s = Math.min(W, H) * 0.45; transform.x = W * 0.5; transform.y = H * 0.5;
    draw();
  }

  function computeThresholdClusters(th) {
    const n = points.length;
    const parent = new Int32Array(n); for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    const unite = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
    for (let i = 0; i < n; i++) {
      const nb = nnGraph[i];
      for (let j = 0; j < nb.length; j++) { const [idx, sim] = nb[j]; if (sim < th) break; unite(i, idx); }
    }
    const map = new Map();
    for (let i = 0; i < n; i++) { const r = find(i); if (!map.has(r)) map.set(r, []); map.get(r).push(i); }
    clusters = Array.from(map.values());
  }

  function seedCenters(k) {
    const n = points.length;
    const centers = new Float32Array(k * 2);
    for (let i = 0; i < k; i++) {
      const p = points[Math.floor(Math.random() * n)];
      centers[i * 2 + 0] = p.x; centers[i * 2 + 1] = p.y;
    }
    return centers;
  }

  function runKMeans() {
    const k = parseInt(kSlider.value, 10);
    kVal.textContent = String(k);
    const n = points.length;
    let centers = seedCenters(k);
    const assign = new Int32Array(n);
    const MAX_IT = 30;
    for (let it = 0; it < MAX_IT; it++) {
      // assign
      for (let i = 0; i < n; i++) {
        let best = -1, bestd = 1e9; const x = points[i].x, y = points[i].y;
        for (let c = 0; c < k; c++) {
          const cx = centers[c * 2 + 0], cy = centers[c * 2 + 1];
          const dx = x - cx, dy = y - cy; const d2 = dx * dx + dy * dy;
          if (d2 < bestd) { bestd = d2; best = c; }
        }
        assign[i] = best;
      }
      // update
      const sum = new Float32Array(k * 2); const cnt = new Int32Array(k);
      for (let i = 0; i < n; i++) { const c = assign[i]; sum[c * 2 + 0] += points[i].x; sum[c * 2 + 1] += points[i].y; cnt[c]++; }
      let moved = 0;
      for (let c = 0; c < k; c++) {
        if (cnt[c] > 0) {
          const nx = sum[c * 2 + 0] / cnt[c]; const ny = sum[c * 2 + 1] / cnt[c];
          if (Math.abs(nx - centers[c * 2 + 0]) + Math.abs(ny - centers[c * 2 + 1]) > 1e-3) moved++;
          centers[c * 2 + 0] = nx; centers[c * 2 + 1] = ny;
        } else {
          // reseed empty center
          const p = points[Math.floor(Math.random() * n)]; centers[c * 2 + 0] = p.x; centers[c * 2 + 1] = p.y; moved++;
        }
      }
      if (moved === 0) break;
    }
    clusterAssign = assign;
    statsEl.textContent = `${points.length} points • k=${k}`;
    draw();
  }

  function colorFor(i) {
    let hue;
    if (clusterAssign) { hue = (clusterAssign[i] * 137) % 360; }
    else if (clusters.length) {
      // derive cluster id by membership
      let cid = -1;
      for (let c = 0; c < clusters.length; c++) { if (clusters[c].includes(i)) { cid = c; break; } }
      hue = (cid * 137) % 360;
    } else { hue = (i * 137) % 360; }
    return `hsl(${hue} 70% 55%)`;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.s, transform.s);
    // edges (threshold mode only)
    const mode = document.querySelector('input[name="mode"]:checked').value;
    if (mode === 'threshold') {
      const th = parseFloat(thSlider.value); thVal.textContent = th.toFixed(2);
      computeThresholdClusters(th);
      ctx.lineWidth = 0.5 / transform.s; ctx.strokeStyle = '#eee';
      for (let i = 0; i < points.length; i++) {
        const nb = nnGraph[i];
        for (let j = 0; j < nb.length; j++) { const [idx, sim] = nb[j]; if (sim < th) break; const a = points[i], b = points[idx]; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      }
    }
    // points
    for (let i = 0; i < points.length; i++) {
      ctx.fillStyle = colorFor(i);
      const p = points[i]; ctx.beginPath(); ctx.arc(p.x, p.y, radius / transform.s, 0, Math.PI * 2); ctx.fill();
    }
    // hover ring
    if (hoverIdx >= 0) {
      const p = points[hoverIdx]; ctx.strokeStyle = '#222'; ctx.lineWidth = 2 / transform.s; ctx.beginPath(); ctx.arc(p.x, p.y, (radius + 2) / transform.s, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();
  }

  function hitTest(x, y) {
    const w = screenToWorld(x, y);
    const r2 = (5 / transform.s) ** 2;
    for (let i = 0; i < points.length; i++) { const p = points[i]; const dx = p.x - w.x, dy = p.y - w.y; if (dx * dx + dy * dy <= r2) return i; }
    return -1;
  }

  // interactions
  canvas.addEventListener('mousedown', (e) => { isPanning = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup', () => { isPanning = false; });
  window.addEventListener('mousemove', (e) => {
    if (isPanning) { const dx = e.clientX - lastX, dy = e.clientY - lastY; transform.x += dx; transform.y += dy; lastX = e.clientX; lastY = e.clientY; draw(); }
    const i = hitTest(e.clientX, e.clientY); hoverIdx = i;
    if (i >= 0) {
      const p = points[i];
      tooltip.style.display = 'block';
      tooltip.style.left = `${e.clientX + 12}px`; tooltip.style.top = `${e.clientY + 12}px`;
      tooltip.innerHTML = `<b>${p.model} • ${p.kind}</b> <small>#${p.item_id}</small><br><pre style="white-space:pre-wrap;max-height:240px;overflow:auto;margin:4px 0 0 0">${escapeHtml(p.text)}</pre>`;
    } else { tooltip.style.display = 'none'; }
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); const factor = Math.exp(-e.deltaY * 0.001); const wx = (e.clientX - transform.x) / transform.s; const wy = (e.clientY - transform.y) / transform.s; transform.s *= factor; transform.x = e.clientX - wx * transform.s; transform.y = e.clientY - wy * transform.s; draw(); }, { passive: false });

  function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

  // UI events
  thSlider.addEventListener('input', draw);
  kSlider.addEventListener('input', runKMeans);
  reseedBtn.addEventListener('click', runKMeans);
  modeRadios.forEach(r => r.addEventListener('change', (e) => setMode(e.target.value)));

  resize();
  load();
})();


