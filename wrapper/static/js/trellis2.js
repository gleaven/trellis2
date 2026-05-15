/**
 * TRELLIS.2 — Wrapper Controller
 * Polls model status, updates header indicators, manages iframe visibility,
 * handles GLB detection (server-side), STL conversion, and persistent gallery.
 */
(function() {
    'use strict';

    // ── DOM Refs ──────────────────────────────────────────────────

    const statusDot = document.getElementById('status-dot');
    const statusBadge = document.getElementById('status-badge');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const gradioFrame = document.getElementById('gradio-frame');
    const gpuFill = document.getElementById('gpu-fill');
    const gpuValue = document.getElementById('gpu-value');
    const stlBtn = document.getElementById('stl-btn');
    const stlPanel = document.getElementById('stl-panel');
    const stlPanelClose = document.getElementById('stl-panel-close');
    const stlStats = document.getElementById('stl-stats');
    const stlDownload = document.getElementById('stl-download');

    // Gallery
    const galleryStrip = document.getElementById('gallery-strip');
    const galleryBar = document.getElementById('gallery-bar');
    const galleryCards = document.getElementById('gallery-cards');
    const galleryCount = document.getElementById('gallery-count');
    const galleryEmpty = document.getElementById('gallery-empty');

    // Viewer overlay
    const viewerOverlay = document.getElementById('viewer-overlay');
    const viewerClose = document.getElementById('viewer-close');
    const viewerName = document.getElementById('viewer-name');
    const viewerStlBtn = document.getElementById('viewer-stl-btn');
    const viewerDownloadBtn = document.getElementById('viewer-download-btn');
    const galleryViewer = document.getElementById('gallery-viewer');

    let gradioReady = false;
    let currentGlbPath = null;
    let lastGlbTimestamp = 0;
    let currentViewerItemId = null;

    // ── Status Polling ──────────────────────────────────────────

    async function pollStatus() {
        try {
            const resp = await fetch('api/status');
            const data = await resp.json();

            if (data.gradio_ready && !gradioReady) {
                gradioReady = true;
                statusDot.classList.add('active');
                statusBadge.textContent = 'MODEL READY';
                statusBadge.className = 'status-badge active';
                loadingOverlay.classList.add('hidden');
                gradioFrame.style.display = 'block';
            } else if (!data.gradio_ready) {
                statusBadge.textContent = 'LOADING MODEL';
                statusBadge.className = 'status-badge loading';
                loadingText.textContent = 'Loading TRELLIS.2 4B pipeline...';
            }
        } catch (e) {
            // Server not yet up
        }
    }

    // ── GPU Polling ─────────────────────────────────────────────

    async function pollGPU() {
        try {
            const resp = await fetch('api/system/stats');
            const data = await resp.json();
            if (data.gpu_percent != null) {
                const pct = Math.round(data.gpu_percent);
                gpuFill.style.width = pct + '%';
                gpuValue.textContent = pct + '%';
            }
        } catch (e) { /* ignore */ }
    }

    // ── Server-Side GLB Detection ───────────────────────────────
    // Replaces the broken postMessage bridge. The reverse proxy tracks
    // .glb files as they pass through to Gradio and exposes them here.

    async function pollLatestGlb() {
        try {
            const resp = await fetch('api/latest-glb');
            const data = await resp.json();
            if (data.path && data.timestamp > lastGlbTimestamp) {
                lastGlbTimestamp = data.timestamp;
                currentGlbPath = data.path;
                stlBtn.classList.add('visible');
                // Auto-save to gallery
                saveToGallery(data.path, data.image_path);
            }
        } catch (e) { /* ignore */ }
    }

    // ── STL Export (Header Button) ──────────────────────────────

    if (stlBtn) {
        stlBtn.addEventListener('click', function() {
            if (!currentGlbPath) return;
            convertAndShowSTL({ glb_path: currentGlbPath }, stlBtn);
        });
    }

    function convertAndShowSTL(body, btn) {
        if (!btn) return;
        btn.classList.add('converting');
        var origText = btn.textContent;
        btn.textContent = 'CONVERTING...';

        fetch('api/convert-stl', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            if (data.error) {
                btn.textContent = 'EXPORT FAILED';
                btn.classList.remove('converting');
                setTimeout(function() {
                    btn.textContent = origText;
                }, 3000);
                return;
            }
            stlStats.innerHTML = buildStatsHTML(data);
            stlDownload.href = data.download_url;
            stlDownload.download = data.filename;
            stlPanel.classList.add('open');
            btn.textContent = origText;
            btn.classList.remove('converting');
        })
        .catch(function() {
            btn.textContent = origText;
            btn.classList.remove('converting');
        });
    }

    function buildStatsHTML(data) {
        var rows = [
            ['Vertices', data.vertices.toLocaleString()],
            ['Faces', data.faces.toLocaleString()],
            ['Watertight', data.is_watertight ? 'Yes' : 'No'],
            ['GLB Size', data.glb_size_mb + ' MB'],
            ['STL Size', data.stl_size_mb + ' MB'],
        ];
        if (data.bounding_box_mm) {
            var bb = data.bounding_box_mm.map(function(v) { return v.toFixed(3); });
            rows.push(['Bounds', bb.join(' x ')]);
        }
        return rows.map(function(r) {
            return '<div class="stl-stat-row">' +
                '<span class="stl-stat-label">' + r[0] + '</span>' +
                '<span class="stl-stat-value">' + r[1] + '</span>' +
                '</div>';
        }).join('');
    }

    if (stlPanelClose) {
        stlPanelClose.addEventListener('click', function() {
            stlPanel.classList.remove('open');
        });
    }

    // ── Gallery ─────────────────────────────────────────────────

    function toggleGallery() {
        galleryStrip.classList.toggle('expanded');
    }

    if (galleryBar) {
        galleryBar.addEventListener('click', toggleGallery);
    }

    async function loadGallery() {
        try {
            const resp = await fetch('api/gallery');
            const data = await resp.json();
            renderGallery(data.items || []);
        } catch (e) {
            renderGallery([]);
        }
    }

    function renderGallery(items) {
        galleryCount.textContent = items.length;
        galleryCards.innerHTML = '';

        if (items.length === 0) {
            galleryCards.appendChild(galleryEmpty);
            return;
        }

        items.forEach(function(item) {
            var card = document.createElement('div');
            card.className = 'gallery-card';
            card.addEventListener('click', function(e) {
                if (e.target.closest('.gallery-card-delete')) return;
                openGalleryViewer(item);
            });

            var thumbSrc = item.has_thumbnail
                ? 'api/gallery/' + item.id + '/thumbnail.jpg'
                : 'data:image/svg+xml,' + encodeURIComponent(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="140" height="85" fill="%2315151f">' +
                    '<rect width="140" height="85"/>' +
                    '<text x="70" y="48" text-anchor="middle" fill="%23555" font-size="11" font-family="monospace">3D</text>' +
                    '</svg>'
                );

            card.innerHTML =
                '<button class="gallery-card-delete" title="Delete">&times;</button>' +
                '<img class="gallery-card-thumb" src="' + thumbSrc + '" alt="' + item.name + '">' +
                '<div class="gallery-card-info">' +
                    '<span class="gallery-card-name">' + item.name + '</span>' +
                    '<span class="gallery-card-size">' + item.glb_size_mb + ' MB</span>' +
                '</div>';

            card.querySelector('.gallery-card-delete').addEventListener('click', function(e) {
                e.stopPropagation();
                deleteGalleryItem(item.id, item.name);
            });

            galleryCards.appendChild(card);
        });
    }

    async function saveToGallery(glbPath, imagePath) {
        try {
            const resp = await fetch('api/gallery', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ glb_path: glbPath, image_path: imagePath || '' }),
            });
            const data = await resp.json();
            if (!data.duplicate) {
                // Expand gallery and reload
                galleryStrip.classList.add('expanded');
                loadGallery();
            }
        } catch (e) {
            console.warn('Gallery save failed:', e);
        }
    }

    async function deleteGalleryItem(id, name) {
        if (!confirm('Delete "' + name + '" from gallery?')) return;
        try {
            await fetch('api/gallery/' + id, { method: 'DELETE' });
            loadGallery();
        } catch (e) {
            console.warn('Gallery delete failed:', e);
        }
    }

    // ── Gallery Viewer ──────────────────────────────────────────

    function openGalleryViewer(item) {
        currentViewerItemId = item.id;
        viewerName.textContent = item.name;
        galleryViewer.setAttribute('src', 'api/gallery/' + item.id + '/model.glb');
        viewerDownloadBtn.href = 'api/gallery/' + item.id + '/model.glb';
        viewerDownloadBtn.download = item.name + '.glb';
        viewerOverlay.classList.add('open');
    }

    function closeGalleryViewer() {
        viewerOverlay.classList.remove('open');
        galleryViewer.removeAttribute('src');
        currentViewerItemId = null;
    }

    if (viewerClose) {
        viewerClose.addEventListener('click', closeGalleryViewer);
    }

    if (viewerStlBtn) {
        viewerStlBtn.addEventListener('click', function() {
            if (!currentViewerItemId) return;
            convertAndShowSTL({ gallery_id: currentViewerItemId }, viewerStlBtn);
        });
    }

    // ESC key closes viewer
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            if (viewerOverlay.classList.contains('open')) {
                closeGalleryViewer();
            }
        }
    });

    // ── Start Polling ───────────────────────────────────────────

    setInterval(pollStatus, 3000);
    setInterval(pollGPU, 5000);
    setInterval(pollLatestGlb, 3000);
    pollStatus();
    pollGPU();
    loadGallery();
})();
