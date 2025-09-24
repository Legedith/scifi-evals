(() => {
  const canvas = document.getElementById('plot');
  const tooltip = document.getElementById('tooltip');
  const thresholdEl = document.getElementById('threshold');
  const thVal = document.getElementById('thVal');
  if (!canvas || !tooltip || !thresholdEl || !thVal) return;

  const ctx = canvas.getContext('2d');
  let points = [];
  let scale = { x: 1, y: 1, ox: 0, oy: 0 };
  let radius = 3;
  let hoverIdx = -1;
  let clusters = [];

  async function load() {
    const res = await fetch('data/embedding_points.json');
    const data = await res.json();
    points = data.points || [];
    fit();
    draw();
  }

  function fit() {
    // Compute bounds
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const p of points) {
      if (p.x < minx) minx = p.x;
      if (p.y < miny) miny = p.y;
      if (p.x > maxx) maxx = p.x;
      if (p.y > maxy) maxy = p.y;
    }
    const w = canvas.width - 40;
    const h = canvas.height - 40;
    const sx = w / Math.max(1e-6, (maxx - minx));
    const sy = h / Math.max(1e-6, (maxy - miny));
    const s = Math.min(sx, sy);
    scale = { x: s, y: -s, ox: 20 - minx * s, oy: canvas.height - 20 + miny * s };
  }

  function toScreen(p) {
    return { X: p.x * scale.x + scale.ox, Y: p.y * scale.y + scale.oy };
  }

  function computeClusters(th) {
    // Simple union-find based on neighbor sim >= th
    const n = points.length;
    const parent = new Array(n).fill(0).map((_, i) => i);
    const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    const unite = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
    for (let i = 0; i < n; i++) {
      const nb = points[i].sim || [];
      for (let j = 0; j < nb.length; j++) {
        const [idx, sim] = nb[j];
        if (sim >= th) unite(i, idx);
      }
    }
    const groups = new Map();
    for (let i = 0; i < n; i++) {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(i);
    }
    clusters = Array.from(groups.values());
  }

  function colorForCluster(ci) {
    // deterministic pastel colors
    const hue = (ci * 137) % 360;
    return `hsl(${hue} 70% 60%)`;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const th = parseFloat(thresholdEl.value);
    thVal.textContent = th.toFixed(2);
    computeClusters(th);
    // draw edges for neighbors above threshold (light)
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = '#eee';
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const a = toScreen(p);
      const nb = p.sim || [];
      for (let k = 0; k < nb.length; k++) {
        const [j, sim] = nb[k];
        if (sim < th) break; // sorted desc
        const q = toScreen(points[j]);
        ctx.beginPath();
        ctx.moveTo(a.X, a.Y);
        ctx.lineTo(q.X, q.Y);
        ctx.stroke();
      }
    }
    // draw points colored by cluster
    for (let ci = 0; ci < clusters.length; ci++) {
      const group = clusters[ci];
      ctx.fillStyle = colorForCluster(ci);
      for (const i of group) {
        const p = toScreen(points[i]);
        ctx.beginPath();
        ctx.arc(p.X, p.Y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (hoverIdx >= 0) {
      const p = toScreen(points[hoverIdx]);
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.X, p.Y, radius + 2, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  function hitTest(x, y) {
    for (let i = 0; i < points.length; i++) {
      const p = toScreen(points[i]);
      const dx = p.X - x, dy = p.Y - y;
      if (dx * dx + dy * dy <= (radius + 2) * (radius + 2)) return i;
    }
    return -1;
  }

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const i = hitTest(x, y);
    hoverIdx = i;
    if (i >= 0) {
      const p = points[i];
      tooltip.style.display = 'block';
      tooltip.style.left = `${e.clientX + 12}px`;
      tooltip.style.top = `${e.clientY + 12}px`;
      tooltip.innerHTML = `<b>${p.model} • ${p.kind}</b><br/><small>ID ${p.item_id}</small><br/><pre style="white-space:pre-wrap;max-height:220px;overflow:auto;margin:0">${escapeHtml(p.text)}</pre>`;
    } else {
      tooltip.style.display = 'none';
    }
    draw();
  });

  canvas.addEventListener('click', (e) => {
    // lock tooltip position on click
    if (hoverIdx >= 0) {
      // no-op; hovering keeps it visible
    }
  });

  thresholdEl.addEventListener('input', draw);

  function escapeHtml(s) {
    return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  }

  load();
})();


