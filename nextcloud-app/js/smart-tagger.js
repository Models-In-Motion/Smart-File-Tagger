/**
 * smart-tagger.js — Nextcloud 27 compatible
 * Shows suggestion banner and manual tag panel as fixed overlays.
 */
(function() {
    'use strict';

    const FIXED_LABELS = ['Lecture Notes', 'Problem Set', 'Exam', 'Reading', 'Other'];

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // ── Get current file ID from Nextcloud API ───────────────────────────────

    function getCurrentFileId() {
        try {
            if (typeof OCA !== 'undefined' &&
                OCA.Files &&
                OCA.Files.Sidebar &&
                OCA.Files.Sidebar.file) {
                const filePath = OCA.Files.Sidebar.file;
                const fileName = filePath.split('/').pop();
                if (OCA.Files.App && OCA.Files.App.fileList) {
                    const model = OCA.Files.App.fileList.getModelForFile(fileName);
                    if (model && model.get('id')) {
                        return String(model.get('id'));
                    }
                }
            }
        } catch(e) {}
        return null;
    }

    // ── Suggestion banner ────────────────────────────────────────────────────

    function loadSuggestion(fileId) {
        fetch(OC.generateUrl('/apps/smartfiletagger/suggestion/' + fileId), {
            headers: { 'requesttoken': OC.requestToken }
        })
        .then(r => r.json())
        .then(data => {
            if (data && data.tag) {
                renderBanner(fileId, data);
            }
        })
        .catch(() => {});
    }

    function renderBanner(fileId, data) {
        removeBanners();
        const confidence = Math.round(data.confidence * 100);
        const banner = document.createElement('div');
        banner.id = 'sft-banner-' + fileId;
        banner.className = 'sft-suggestion-banner';
        banner.innerHTML = `
            <div class="sft-banner-header">
                <span class="sft-icon">🏷️</span>
                <strong>Suggested tag:</strong>
                <span class="sft-tag-pill">${escapeHtml(data.tag)}</span>
                <span class="sft-confidence">${confidence}% confidence</span>
            </div>
            <div class="sft-actions">
                <button class="sft-btn sft-btn-confirm" id="sft-confirm-${fileId}">✓ Apply tag</button>
                <button class="sft-btn sft-btn-reject" id="sft-reject-${fileId}">✗ Not this tag</button>
            </div>
        `;
        document.body.appendChild(banner);

        document.getElementById('sft-confirm-' + fileId).addEventListener('click', () => handleConfirm(fileId, data.tag));
        document.getElementById('sft-reject-' + fileId).addEventListener('click', () => handleReject(fileId, data.tag));
    }

    function handleConfirm(fileId, tag) {
        fetch(OC.generateUrl('/apps/smartfiletagger/confirm'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'requesttoken': OC.requestToken
            },
            body: 'fileId=' + encodeURIComponent(fileId) + '&tag=' + encodeURIComponent(tag)
        })
        .then(r => r.json())
        .then(() => {
            removeBanners();
            showToast('✅ Tag applied: ' + tag);
        })
        .catch(() => {});
    }

    function handleReject(fileId, tag) {
        fetch(OC.generateUrl('/apps/smartfiletagger/reject'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'requesttoken': OC.requestToken
            },
            body: 'fileId=' + encodeURIComponent(fileId) + '&tag=' + encodeURIComponent(tag)
        })
        .then(() => {
            removeBanners();
            showToast('Feedback saved.');
        })
        .catch(() => {});
    }

    // ── Manual tag panel ─────────────────────────────────────────────────────

    function renderManualTagPanel(fileId) {
        removeManualPanels();
        const panel = document.createElement('div');
        panel.id = 'sft-manual-panel-' + fileId;
        panel.className = 'sft-manual-tag-panel';
        panel.innerHTML = `
            <div class="sft-manual-header">
                <span class="sft-icon">🏷️</span>
                <strong>Assign tag manually</strong>
                <button class="sft-close" id="sft-close-${fileId}">✕</button>
            </div>
            <div class="sft-manual-row">
                <select id="sft-label-select-${fileId}" class="sft-label-select">
                    <option value="">— Select a label —</option>
                    ${FIXED_LABELS.map(l => `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`).join('')}
                </select>
                <button class="sft-btn sft-btn-confirm" id="sft-manual-apply-${fileId}">Apply</button>
            </div>
            <div id="sft-manual-result-${fileId}" class="sft-manual-result"></div>
        `;
        document.body.appendChild(panel);

        document.getElementById('sft-close-' + fileId).addEventListener('click', removeManualPanels);
        document.getElementById('sft-manual-apply-' + fileId).addEventListener('click', () => {
            const tag = document.getElementById('sft-label-select-' + fileId).value;
            if (!tag) { alert('Please select a label first.'); return; }
            applyManualTag(fileId, tag);
        });
    }

    function applyManualTag(fileId, tag) {
        fetch(OC.generateUrl('/apps/smartfiletagger/manual-tag'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'requesttoken': OC.requestToken
            },
            body: 'fileId=' + encodeURIComponent(fileId) + '&tag=' + encodeURIComponent(tag)
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                removeManualPanels();
                showToast('✅ Tag applied: ' + tag);
            } else {
                const result = document.getElementById('sft-manual-result-' + fileId);
                if (result) result.innerHTML = '<span style="color:red">Failed: ' + escapeHtml(data.error || 'unknown') + '</span>';
            }
        })
        .catch(() => {});
    }

    // ── Toast notification ───────────────────────────────────────────────────

    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'sft-toast';
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }

    // ── Cleanup helpers ──────────────────────────────────────────────────────

    function removeBanners() {
        document.querySelectorAll('[id^="sft-banner-"]').forEach(el => el.remove());
    }

    function removeManualPanels() {
        document.querySelectorAll('[id^="sft-manual-panel-"]').forEach(el => el.remove());
    }

    // ── Main polling loop ────────────────────────────────────────────────────
    // Poll every second to detect file selection changes

    let lastFileId = null;

    function poll() {
        const fileId = getCurrentFileId();

        if (!fileId) {
            if (lastFileId !== null) {
                removeBanners();
                removeManualPanels();
                lastFileId = null;
            }
            return;
        }

        if (fileId !== lastFileId) {
            lastFileId = fileId;
            removeBanners();
            removeManualPanels();
            loadSuggestion(fileId);
            renderManualTagPanel(fileId);
        }
    }

    // ── Category manager page ────────────────────────────────────────────────

    function initCategoryManager() {
        const form = document.getElementById('sft-create-category-form');
        if (!form) return;

        loadCategoryList();

        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const name = document.getElementById('sft-category-name').value.trim();
            const idsInput = document.getElementById('sft-example-ids').value.trim();
            const fileIds = idsInput.split(',').map(s => s.trim()).filter(s => s !== '');

            if (!name) { alert('Please enter a category name.'); return; }
            if (fileIds.length < 3) { alert('Please enter at least 3 file IDs.'); return; }

            const params = new URLSearchParams();
            params.append('categoryName', name);
            fileIds.forEach(id => params.append('exampleFileIds[]', id));

            fetch(OC.generateUrl('/apps/smartfiletagger/categories/register'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'requesttoken': OC.requestToken
                },
                body: params.toString()
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('sft-category-name').value = '';
                    document.getElementById('sft-example-ids').value = '';
                    loadCategoryList();
                    showToast('Category created successfully.');
                } else {
                    alert('Failed: ' + (data.message || data.error));
                }
            });
        });
    }

    function loadCategoryList() {
        const list = document.getElementById('sft-category-list');
        if (!list) return;
        fetch(OC.generateUrl('/apps/smartfiletagger/categories'), {
            headers: { 'requesttoken': OC.requestToken }
        })
        .then(r => r.json())
        .then(data => {
            list.innerHTML = '';
            if (!data.categories || data.categories.length === 0) {
                list.innerHTML = '<p class="sft-empty">No custom categories yet.</p>';
                return;
            }
            data.categories.forEach(cat => {
                const item = document.createElement('div');
                item.className = 'sft-category-item';
                item.innerHTML = `
                    <span class="sft-tag-pill">${escapeHtml(cat.category_name)}</span>
                    <span class="sft-example-count">${cat.example_count} examples</span>
                    <button class="sft-btn-delete" data-name="${escapeHtml(cat.category_name)}">Remove</button>
                `;
                item.querySelector('.sft-btn-delete').addEventListener('click', function() {
                    deleteCategory(this.dataset.name);
                });
                list.appendChild(item);
            });
        });
    }

    function deleteCategory(name) {
        if (!confirm('Delete category "' + name + '"?')) return;
        fetch(OC.generateUrl('/apps/smartfiletagger/categories/' + encodeURIComponent(name)), {
            method: 'DELETE',
            headers: { 'requesttoken': OC.requestToken }
        })
        .then(() => loadCategoryList());
    }

    // ── Debug helper ─────────────────────────────────────────────────────────

    window.sftDebug = function() {
        console.log('lastFileId:', lastFileId);
        console.log('getCurrentFileId():', getCurrentFileId());
        console.log('OCA.Files.Sidebar.file:', typeof OCA !== 'undefined' && OCA.Files && OCA.Files.Sidebar ? OCA.Files.Sidebar.file : 'NOT FOUND');
    };

    // ── Init ─────────────────────────────────────────────────────────────────

    function startPolling() {
        setInterval(poll, 1000);
    }

    document.addEventListener('DOMContentLoaded', function() {
        // Delay start to let Nextcloud Vue app initialize
        setTimeout(startPolling, 3000);
        initCategoryManager();
    });

    // Also restart if Nextcloud navigates internally
    window.addEventListener('load', function() {
        setTimeout(startPolling, 3000);
    });

})();
