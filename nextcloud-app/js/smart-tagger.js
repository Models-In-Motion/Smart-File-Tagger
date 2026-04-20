/**
 * smart-tagger.js — Nextcloud 27 compatible
 * Uses OCA.Files.Sidebar and fetch() instead of jQuery $.get
 */
(function() {
    'use strict';

    // ── Suggestion banner ────────────────────────────────────────────────────
    // Polls for pending suggestions and injects banner into sidebar
    let _sftReloadTimer = null;

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
                refreshFilesView(200);
            } else {
                removeBanner(fileId);
            }
        })
        .catch(() => {});
    }

    function removeBanner(fileId) {
        const existing = document.getElementById('sft-banner-' + fileId);
        if (existing) existing.remove();
    }

    function refreshFilesView(delayMs = 0) {
        if (_sftReloadTimer) {
            clearTimeout(_sftReloadTimer);
        }
        _sftReloadTimer = setTimeout(function() {
            try {
                if (window.OCA && OCA.Files && OCA.Files.App && OCA.Files.App.fileList) {
                    OCA.Files.App.fileList.reload();
                }
            } catch (_) {}
        }, delayMs);
    }

    function renderBanner(fileId, data) {
        // Remove existing banner
        removeBanner(fileId);

        const confidence = Math.round(data.confidence * 100);
        const isSuggested = (data.action || 'suggest') === 'suggest';
        const banner = document.createElement('div');
        banner.id = 'sft-banner-' + fileId;
        banner.className = 'sft-suggestion-banner' + (isSuggested ? ' sft-banner-suggested' : ' sft-banner-applied');
        banner.innerHTML = isSuggested
            ? `
                <div class="sft-banner-header">
                    <span class="sft-banner-prefix">Suggested:</span>
                    <span class="sft-tag-pill sft-tag-pill-suggested">${escapeHtml(data.tag)}</span>
                    <span class="sft-confidence">${confidence}%</span>
                </div>
                <div class="sft-actions">
                    <button class="sft-btn sft-btn-confirm" id="sft-confirm-${fileId}">Accept</button>
                    <button class="sft-btn sft-btn-reject" id="sft-reject-${fileId}">Reject</button>
                </div>
            `
            : `
                <div class="sft-banner-header">
                    <span class="sft-banner-prefix">Auto-applied:</span>
                    <span class="sft-tag-pill sft-tag-pill-applied">${escapeHtml(data.tag)}</span>
                    <span class="sft-confidence">${confidence}%</span>
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

        if (isSuggested) {
            document.getElementById('sft-confirm-' + fileId).addEventListener('click', () => handleConfirm(fileId, data.tag));
            document.getElementById('sft-reject-' + fileId).addEventListener('click', () => handleReject(fileId, data.tag));
        }

        // Keep the UI clean: auto-dismiss after a few seconds.
        setTimeout(() => removeBanner(fileId), 7000);
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
                banner.className = 'sft-suggestion-banner sft-banner-applied';
                banner.innerHTML = `
                    <div class="sft-banner-header">
                        <span class="sft-banner-prefix">Applied:</span>
                        <span class="sft-tag-pill sft-tag-pill-applied">${escapeHtml(tag)}</span>
                    </div>
                `;
                setTimeout(() => removeBanner(fileId), 5000);
            }
            refreshFilesView(150);
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
            removeBanner(fileId);
            // Show Nextcloud notification
            if (window.OC && OC.Notification) {
                OC.Notification.showTemporary('Feedback saved.');
            }
            refreshFilesView(150);
        })
        .catch(() => {});
    }

    // ── Hook into sidebar open ───────────────────────────────────────────────
    // Nextcloud 27 uses a Vue-based sidebar — watch for DOM changes

    function watchSidebar() {
        function checkUrl() {
            const urlParams = new URLSearchParams(window.location.search);
            const fileId = urlParams.get('openfile') || urlParams.get('fileid');
            if (fileId && fileId !== window._sftLastFileId) {
                window._sftLastFileId = fileId;
                // Small delay to let sidebar DOM render
                setTimeout(function() {
                    loadSuggestion(fileId);
                }, 800);
            }
        }

        // Check on load
        checkUrl();

        // Watch URL changes (Nextcloud uses pushState)
        const originalPushState = history.pushState;
        history.pushState = function() {
            originalPushState.apply(history, arguments);
            setTimeout(checkUrl, 100);
        };

        const originalReplaceState = history.replaceState;
        history.replaceState = function() {
            originalReplaceState.apply(history, arguments);
            setTimeout(checkUrl, 100);
        };

        window.addEventListener('popstate', checkUrl);

        // Also watch DOM for sidebar appearing
        const observer = new MutationObserver(function() {
            const sidebar = document.querySelector('#app-sidebar-vue');
            if (sidebar && sidebar.children.length > 0) {
                checkUrl();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    function watchFileUploads() {
        // Method 1: intercept XMLHttpRequest to detect WebDAV PUT completions
        const originalOpen = XMLHttpRequest.prototype.open;
        const originalSend = XMLHttpRequest.prototype.send;

        XMLHttpRequest.prototype.open = function(method, url) {
            this._method = method;
            this._url = url;
            return originalOpen.apply(this, arguments);
        };

        XMLHttpRequest.prototype.send = function() {
            if (this._method === 'PUT' && this._url &&
                this._url.includes('/remote.php/')) {
                this.addEventListener('load', function() {
                    if (this.status >= 200 && this.status < 300) {
                        // File uploaded successfully — wait for webhook to process
                        // then refresh the file list to show new tags
                        setTimeout(refreshFileList, 3000);
                        setTimeout(refreshFileList, 6000);
                    }
                });
            }
            return originalSend.apply(this, arguments);
        };

        // Method 2: watch for upload completion events from Nextcloud's uploader
        document.addEventListener('uploadDone', function() {
            setTimeout(refreshFileList, 3000);
        });

        // Method 3: OC.Uploader events if available
        if (window.OC && OC.Uploader) {
            document.addEventListener('ajaxComplete', function(e) {
                if (e.detail && e.detail.url &&
                    e.detail.url.includes('remote.php')) {
                    setTimeout(refreshFileList, 3000);
                }
            });
        }
    }

    function refreshFileList() {
        // Method 1: Nextcloud 27 Vue file list refresh
        const event = new CustomEvent('nextcloud:files:reload');
        document.dispatchEvent(event);

        // Method 2: trigger hashchange to force Vue router refresh
        const currentHash = window.location.hash;
        window.dispatchEvent(new Event('hashchange'));

        // Method 3: click the refresh button if visible
        const refreshBtn = document.querySelector(
            '.files-controls .button[data-original-title="Reload"], ' +
            'button[aria-label="Reload"], ' +
            '.app-content-list__refresh'
        );
        if (refreshBtn) refreshBtn.click();

        // Method 4: use OCA.Files if available (NC27)
        if (window.OCA && OCA.Files && OCA.Files.App) {
            const fileList = OCA.Files.App.fileList;
            if (fileList && typeof fileList.reload === 'function') {
                fileList.reload();
            }
        }

        // Method 5: navigate to same URL to force Vue refresh
        if (window.location.pathname.includes('/apps/files')) {
            const url = new URL(window.location.href);
            url.searchParams.set('_t', Date.now());
            window.history.replaceState({}, '', url);
            window.dispatchEvent(new PopStateEvent('popstate'));
        }
    }

    function getFileIdFromSidebar(sidebar) {
        // Try data attribute
        const el = sidebar.querySelector('[data-file-id], [data-id]');
        if (el) return el.dataset.fileId || el.dataset.id;
        // Try URL params
        const urlParams = new URLSearchParams(window.location.search);
        const fileId = urlParams.get('openfile') || urlParams.get('fileid');
        if (fileId) return fileId;
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

        fetch(OC.generateUrl('/apps/smartfiletagger/categories/list'), {
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
    // Use multiple strategies to ensure we init after Nextcloud's Vue app loads

    function init() {
        watchSidebar();
        initCategoryManager();
        watchFileUploads();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            // Wait extra time for Nextcloud Vue app to mount
            setTimeout(init, 1000);
        });
    } else {
        // Already loaded
        setTimeout(init, 500);
    }

    // Also expose globally so we can call manually from console for debugging
    window.sftInit = init;
    window.sftLoadSuggestion = loadSuggestion;
    window.sftRenderBanner = renderBanner;

})();