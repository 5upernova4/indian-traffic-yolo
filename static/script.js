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
        mediaContainer.innerHTML = '';
        fileInput.value = '';
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

            // 5. Render the annotated media returned by the backend
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
                videoEl.addEventListener('error', () => {
                    mediaContainer.innerHTML = '';
                    const fallback = document.createElement('div');
                    fallback.className = 'video-fallback';
                    fallback.innerHTML = `
                        <p class="loading-text">Video preview unavailable</p>
                        <p class="loading-subtext">Your browser cannot play this codec. Download the file instead.</p>
                        <a href="${result.url}" download class="btn btn-primary" style="margin-top:1rem;">⬇ Download Video</a>
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

            // 6. Wire up CSV download button
            if (downloadCsvBtn && result.csv_url) {
                downloadCsvBtn.href = result.csv_url;
                downloadCsvBtn.download = 'detections.csv';
            }
        } catch (err) {
            loadingState.classList.add('hidden');
            dropzone.classList.remove('hidden');
            alert('Error: ' + err.message);
            console.error('Processing failed:', err);
        }
    }
});