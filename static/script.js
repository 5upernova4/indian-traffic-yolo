document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const loadingState = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    const mediaContainer = document.getElementById('mediaContainer');
    const reuploadBtn = document.getElementById('reuploadBtn');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');

    // Progress bar elements
    const loadingText = document.getElementById('loadingText');
    const loadingSubtext = document.getElementById('loadingSubtext');
    const progressWrapper = document.getElementById('progressWrapper');
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressFrames = document.getElementById('progressFrames');

    // Results panel + counting-line settings
    const settingsRow = document.getElementById('settingsRow');
    const lineOrientation = document.getElementById('lineOrientation');
    const linePosition = document.getElementById('linePosition');
    const linePositionValue = document.getElementById('linePositionValue');
    const linePreviewLine = document.getElementById('linePreviewLine');
    const statsGrid = document.getElementById('statsGrid');
    const mediaTabs = document.getElementById('mediaTabs');
    const classBars = document.getElementById('classBars');
    const classCardTitle = document.getElementById('classCardTitle');
    const classCardSub = document.getElementById('classCardSub');
    const timelineCard = document.getElementById('timelineCard');
    const timelineCanvas = document.getElementById('timelineChart');
    const timelineLegend = document.getElementById('timelineLegend');
    const extraDownloads = document.getElementById('extraDownloads');
    let lastResult = null;

    function updateLinePreview() {
        const pct = linePosition.value;
        linePositionValue.textContent = `${pct}%`;
        const vertical = lineOrientation.value === 'vertical';
        linePreviewLine.classList.toggle('vertical', vertical);
        linePreviewLine.style.top = vertical ? '0' : `${pct}%`;
        linePreviewLine.style.left = vertical ? `${pct}%` : '0';
    }
    linePosition.addEventListener('input', updateLinePreview);
    lineOrientation.addEventListener('change', updateLinePreview);
    updateLinePreview();

    // Trigger file selection window
    uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and Drop Visual Feedback Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    // Capture Dropped Files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileProcess(files[0]);
        }
    });

    // Capture Standard Browsed Files
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileProcess(e.target.files[0]);
        }
    });

    // Reset layout for processing new uploads
    reuploadBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        dropzone.classList.remove('hidden');
        settingsRow.classList.remove('hidden');
        mediaContainer.innerHTML = '';
        fileInput.value = '';
        lastResult = null;
        resetProgress();
    });

    /**
     * Reset the progress bar to its initial state.
     */
    function resetProgress() {
        if (progressFill) progressFill.style.width = '0%';
        if (progressPercent) progressPercent.textContent = '0%';
        if (progressFrames) progressFrames.textContent = '';
        if (progressWrapper) progressWrapper.classList.add('hidden');
        if (loadingText) loadingText.textContent = 'Analyzing footage…';
        if (loadingSubtext) loadingSubtext.textContent = 'Our neural network is working magic.';
    }

    /**
     * Poll /progress/<run_id> until status is 'done' or 'error'.
     * Updates the progress bar and status text in real time.
     */
    function pollProgress(runId) {
        return new Promise((resolve, reject) => {
            const POLL_INTERVAL_MS = 400;

            const tick = async () => {
                try {
                    const res = await fetch(`/progress/${runId}`);
                    const data = await res.json();

                    if (data.status === 'done') {
                        // Fill bar to 100% before resolving
                        setProgress(1, 1, 'Complete!');
                        setTimeout(() => resolve(data), 300);
                        return;
                    }

                    if (data.status === 'error') {
                        reject(new Error(data.error || 'Processing failed.'));
                        return;
                    }

                    // Still processing — update UI
                    if (data.total && data.total > 1) {
                        // Video: show frame-level progress
                        progressWrapper.classList.remove('hidden');
                        setProgress(data.progress, data.total, data.message);
                    } else {
                        // Image or early init
                        if (data.message && loadingSubtext) {
                            loadingSubtext.textContent = data.message;
                        }
                    }

                    setTimeout(tick, POLL_INTERVAL_MS);
                } catch (err) {
                    // Network hiccup — retry silently
                    setTimeout(tick, POLL_INTERVAL_MS * 2);
                }
            };

            tick();
        });
    }

    /**
     * Update the progress bar fill, percentage label, and frame counter.
     */
    function setProgress(current, total, message) {
        const pct = Math.min(Math.round((current / total) * 100), 100);
        if (progressFill) progressFill.style.width = `${pct}%`;
        if (progressPercent) progressPercent.textContent = `${pct}%`;
        if (progressFrames) progressFrames.textContent = `${current} / ${total} frames`;
        if (message && loadingSubtext) loadingSubtext.textContent = message;
    }

    /**
     * Pipeline — upload file, then poll for progress until results are ready.
     */
    async function handleFileProcess(file) {
        // 1. Swap UI into Processing Mode
        dropzone.classList.add('hidden');
        settingsRow.classList.add('hidden');
        resultSection.classList.add('hidden');
        loadingState.classList.remove('hidden');
        mediaContainer.innerHTML = '';
        resetProgress();

        const isVideo = file.type.startsWith('video/');
        if (isVideo && loadingText) {
            loadingText.textContent = 'Processing video…';
        }

        // 2. POST to Flask /upload — returns immediately with a run_id
        const formData = new FormData();
        formData.append('file', file);
        formData.append('line_position', (linePosition.value / 100).toFixed(2));
        formData.append('line_orientation', lineOrientation.value);

        try {
            const uploadRes = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            const uploadData = await uploadRes.json();

            if (!uploadRes.ok) {
                throw new Error(uploadData.error || 'Upload failed.');
            }

            const runId = uploadData.run_id;

            // 3. Poll /progress until done
            const result = await pollProgress(runId);

            // 4. Hide loading, display results
            loadingState.classList.add('hidden');
            resultSection.classList.remove('hidden');

            // 5. Render the annotated media + analytics returned by the backend
            lastResult = result;
            renderMedia(result, 'annotated');
            renderStats(result);
            renderClassBars(result);
            renderTimeline(result);
            renderDownloads(result);

            // 6. Wire up CSV download button
            if (downloadCsvBtn && result.csv_url) {
                downloadCsvBtn.href = result.csv_url;
                downloadCsvBtn.download = 'detections.csv';
            }
        } catch (err) {
            loadingState.classList.add('hidden');
            dropzone.classList.remove('hidden');
            settingsRow.classList.remove('hidden');
            alert('Error: ' + err.message);
            console.error('Processing failed:', err);
        }
    }

    // ── Results rendering ───────────────────────────────────────────────────
    mediaTabs.addEventListener('click', (e) => {
        const tab = e.target.closest('.tab');
        if (!tab || !lastResult) return;
        mediaTabs.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
        renderMedia(lastResult, tab.dataset.view);
    });

    function renderMedia(result, view) {
        mediaContainer.innerHTML = '';
        mediaTabs.classList.toggle('hidden', result.type !== 'video');
        if (result.type === 'video' && view === 'heatmap') {
            const imgEl = document.createElement('img');
            imgEl.src = result.heatmap_url;
            imgEl.alt = 'Where vehicles were detected most often';
            mediaContainer.appendChild(imgEl);
            return;
        }
        if (result.type === 'video') {
            const videoEl = document.createElement('video');
            videoEl.controls = true;
            videoEl.autoplay = true;
            videoEl.muted = true;
            videoEl.playsInline = true;

            // Use <source> with MIME type for proper codec negotiation
            const sourceEl = document.createElement('source');
            sourceEl.src = result.url;
            sourceEl.type = 'video/mp4';
            videoEl.appendChild(sourceEl);

            // Fallback if browser can't decode the codec
            sourceEl.addEventListener('error', () => {
                mediaContainer.innerHTML = '';
                const fallback = document.createElement('div');
                fallback.className = 'video-fallback';
                fallback.innerHTML = `
                    <p class="loading-text">Video preview unavailable</p>
                    <p class="loading-subtext">Your browser cannot play this codec. Download the file instead.</p>
                    <a href="${result.url}" download class="btn btn-primary" style="margin-top:1rem;">Download Video</a>
                `;
                mediaContainer.appendChild(fallback);
            });

            mediaContainer.appendChild(videoEl);
        } else {
            const imgEl = document.createElement('img');
            imgEl.src = result.url;
            imgEl.alt = 'YOLO Detection Result';
            mediaContainer.appendChild(imgEl);
        }
    }

    function statCard(label, value, hint) {
        return `<div class="stat-card glass-panel-inner">
            <span class="stat-value">${value}</span>
            <span class="stat-label">${label}</span>
            ${hint ? `<span class="stat-hint">${hint}</span>` : ''}
        </div>`;
    }

    function sumValues(obj) {
        return Object.values(obj || {}).reduce((a, b) => a + b, 0);
    }

    function lineTotals(summary) {
        const tot = { in: 0, out: 0, perClass: {} };
        Object.values(summary.line_counts || {}).forEach(line => {
            ['in', 'out'].forEach(dir => {
                Object.entries(line[dir] || {}).forEach(([cls, n]) => {
                    tot[dir] += n;
                    tot.perClass[cls] = tot.perClass[cls] || { in: 0, out: 0 };
                    tot.perClass[cls][dir] += n;
                });
            });
        });
        return tot;
    }

    function renderStats(result) {
        const s = result.summary;
        if (result.type === 'video') {
            const t = lineTotals(s);
            statsGrid.innerHTML = [
                statCard('Vehicles counted', s.total_counted, 'crossed the counting line'),
                statCard('Direction in / out', `${t.in} / ${t.out}`, 'in = down / right across the line'),
                statCard('Vehicles tracked', sumValues(s.unique_vehicles_per_class), 'unique track IDs'),
                statCard('Processing speed', `${s.processing_fps} fps`, `${s.frames_processed} frames · ${s.duration_s}s`),
            ].join('');
        } else {
            const counts = s.detections_per_class;
            const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
            const label = result.classes.find(c => c.id === top[0]);
            statsGrid.innerHTML = [
                statCard('Vehicles detected', s.total_vehicles),
                statCard('Most common', s.total_vehicles ? label.label : '–', s.total_vehicles ? `${top[1]} in frame` : ''),
                statCard('Inference time', `${s.inference_ms} ms`, `${s.model} @ ${s.imgsz}px`),
            ].join('');
        }
    }

    function renderClassBars(result) {
        const s = result.summary;
        let rows;
        if (result.type === 'video') {
            const t = lineTotals(s);
            classCardTitle.textContent = 'Vehicles counted by class';
            classCardSub.innerHTML = '<span class="dir-chip in"></span>in <span class="dir-chip out"></span>out' +
                ' &middot; tracked vehicles in brackets';
            rows = result.classes.map(c => ({
                ...c,
                in: (t.perClass[c.id] || {}).in || 0,
                out: (t.perClass[c.id] || {}).out || 0,
                tracked: (s.unique_vehicles_per_class || {})[c.id] || 0,
            }));
        } else {
            classCardTitle.textContent = 'Vehicles by class';
            classCardSub.textContent = 'Detections in this image (confidence ≥ 0.25)';
            rows = result.classes.map(c => ({ ...c, in: s.detections_per_class[c.id] || 0, out: 0 }));
        }
        const max = Math.max(1, ...rows.map(r => r.in + r.out));
        classBars.innerHTML = rows.map(r => `
            <div class="bar-row">
                <span class="bar-label"><span class="dot" style="background:${r.color}"></span>${r.label}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${(r.in / max) * 100}%;background:${r.color}"></div>
                    ${result.type === 'video' ? `<div class="bar-fill out" style="width:${(r.out / max) * 100}%;background:${r.color}"></div>` : ''}
                </div>
                <span class="bar-value">${r.in + r.out}${result.type === 'video' ? ` <small>(${r.tracked})</small>` : ''}</span>
            </div>`).join('');
    }

    function renderTimeline(result) {
        const timeline = (result.summary.timeline || []);
        if (result.type !== 'video' || timeline.length < 2) {
            timelineCard.classList.add('hidden');
            return;
        }
        timelineCard.classList.remove('hidden');
        requestAnimationFrame(() => drawTimeline(timelineCanvas, timeline, result.classes));
        timelineLegend.innerHTML = result.classes.map(c =>
            `<span><span class="dot" style="background:${c.color}"></span>${c.label}</span>`).join('');
    }

    function drawTimeline(canvas, timeline, classes) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth, h = canvas.clientHeight || 220;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);
        const pad = { l: 32, r: 8, t: 8, b: 24 };
        const maxY = Math.max(1, ...timeline.flatMap(p => classes.map(c => p[c.id] || 0)));
        const tMax = timeline[timeline.length - 1].t_start_s || 1;
        const x = t => pad.l + (t / tMax) * (w - pad.l - pad.r);
        const y = v => h - pad.b - (v / maxY) * (h - pad.t - pad.b);

        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.fillStyle = '#9ca3af';
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.lineWidth = 1;
        const ticks = 4;
        for (let i = 0; i <= ticks; i++) {
            const v = (maxY / ticks) * i;
            ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(w - pad.r, y(v)); ctx.stroke();
            ctx.textAlign = 'right';
            ctx.fillText(v.toFixed(maxY < 4 ? 1 : 0), pad.l - 6, y(v) + 4);
        }
        ctx.textAlign = 'center';
        for (let i = 0; i <= 4; i++) {
            const t = (tMax / 4) * i;
            ctx.fillText(`${t.toFixed(0)}s`, x(t), h - 6);
        }
        classes.forEach(c => {
            if (!timeline.some(p => (p[c.id] || 0) > 0)) return;
            ctx.strokeStyle = c.color;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            timeline.forEach((p, i) => {
                const px = x(p.t_start_s), py = y(p[c.id] || 0);
                i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
            });
            ctx.stroke();
        });
    }

    function renderDownloads(result) {
        const links = [['Detections CSV', result.csv_url, 'detections.csv'], ['Summary JSON', result.json_url, 'summary.json']];
        if (result.type === 'video') {
            links.splice(1, 0, ['Line crossings CSV', result.crossings_csv_url, 'crossings.csv']);
            links.push(['Annotated video', result.url, 'annotated.mp4']);
        }
        extraDownloads.innerHTML = '<span>Downloads:</span>' + links.map(([label, href, name]) =>
            `<a href="${href}" download="${name}">${label}</a>`).join('');
    }

    window.addEventListener('resize', () => {
        if (lastResult && lastResult.type === 'video' && (lastResult.summary.timeline || []).length > 1) {
            drawTimeline(timelineCanvas, lastResult.summary.timeline, lastResult.classes);
        }
    });
});