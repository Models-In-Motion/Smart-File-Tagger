/**
 * smart-tagger.js — Nextcloud 27 compatible
 * Uses OCA.Files.Sidebar and fetch() instead of jQuery $.get
 */
(function() {
    'use strict';

    // ── Suggestion banner ────────────────────────────────────────────────────
    // Polls for pending suggestions and injects banner into sidebar

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function loadSuggestion(fileId) {
        if (!fileId) return;
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
        // Remove existing banner
        const existing = document.getElementById('sft-banner-' + fileId);
        if (existing) existing.remove();

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
                <button class="sft-btn sft-btn-confirm" id="sft-confirm-${fileId}">Apply tag</button>
                <button class="sft-btn sft-btn-reject" id="sft-reject-${fileId}">Not this tag</button>
            </div>
        `;

        // Inject into sidebar — try multiple selectors for NC 27 compatibility
        const targets = [
            '#app-sidebar-vue .app-sidebar-header',
            '#app-sidebar .app-sidebar-header',
            '.app-sidebar-header',
            '#app-sidebar',
        ];
        let injected = false;
        for (const sel of targets) {
            const el = document.querySelector(sel);
            if (el) {
                el.parentNode.insertBefore(banner, el.nextSibling);
                injected = true;
                break;
            }
        }
        if (!injected) {
            // Last resort: fixed position overlay
            banner.style.cssText = 'position:fixed;top:70px;right:20px;z-index:9999;background:#fff;border:1px solid #ccc;padding:12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);max-width:300px;';
            document.body.appendChild(banner);
        }

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
            const banner = document.getElementById('sft-banner-' + fileId);
            if (banner) {
                banner.innerHTML = '<div class="sft-confirmed">✅ Tag applied: <strong>' + escapeHtml(tag) + '</strong></div>';
            }
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
            const banner = document.getElementById('sft-banner-' + fileId);
            if (banner) banner.remove();
            // Show Nextcloud notification
            if (window.OC && OC.Notification) {
                OC.Notification.showTemporary('Feedback saved.');
            }
        })
        .catch(() => {});
    }

    // ── Hook into sidebar open ───────────────────────────────────────────────
    // Nextcloud 27 uses a Vue-based sidebar — watch for DOM changes

    function watchSidebar() {
        const observer = new MutationObserver(function(mutations) {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType !== 1) continue;
                    // Look for the file ID in the sidebar
                    const sidebar = document.querySelector('#app-sidebar-vue, #app-sidebar');
                    if (sidebar) {
                        // Try to get file ID from URL hash or data attribute
                        const fileId = getFileIdFromSidebar(sidebar);
                        if (fileId) {
                            loadSuggestion(fileId);
                        }
                    }
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    function getFileIdFromSidebar(sidebar) {
        // Try data attribute
        const el = sidebar.querySelector('[data-file-id], [data-id]');
        if (el) return el.dataset.fileId || el.dataset.id;
        // Try URL hash
        const match = window.location.search.match(/fileid=(\d+)/);
        if (match) return match[1];
        return null;
    }

    // ── Category manager ─────────────────────────────────────────────────────
    // Runs on the /apps/smartfiletagger/categories page

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
            if (fileIds.length < 3) { alert('Please select at least 3 example files.'); return; }

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
                    loadCategoryList();
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

    // ── Init ─────────────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', function() {
        watchSidebar();
        initCategoryManager();
    });

})();