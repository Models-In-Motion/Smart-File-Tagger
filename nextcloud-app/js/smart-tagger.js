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

        // Inject before the manual tag panel if it's already in the DOM,
        // otherwise fall back to the standard sidebar header targets.
        const manualPanel = document.getElementById('sft-manual-panel-' + fileId);
        if (manualPanel) {
            manualPanel.parentNode.insertBefore(banner, manualPanel);
        } else {
            const targets = [
                '#app-sidebar-vue .app-sidebar-header',
                '#app-sidebar .app-sidebar-header',
                '.app-sidebar-header',
                '#app-sidebar-vue',
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
                banner.style.cssText = 'position:fixed;top:70px;right:20px;z-index:9999;background:#fff;border:1px solid #ccc;padding:12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);max-width:300px;';
                document.body.appendChild(banner);
            }
        }

        if (isSuggested) {
            document.getElementById('sft-confirm-' + fileId).addEventListener('click', () => handleConfirm(fileId, data.tag));
            document.getElementById('sft-reject-' + fileId).addEventListener('click', () => handleReject(fileId, data.tag));
        }

        // Hide manual tag panel while a suggestion is pending — user should Accept/Reject first.
        const existingManualPanel = document.getElementById('sft-manual-panel-' + fileId);
        if (existingManualPanel) {
            existingManualPanel.style.display = 'none';
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
            // Tag accepted — manual panel is no longer needed for this file.
            const manualPanel = document.getElementById('sft-manual-panel-' + fileId);
            if (manualPanel) manualPanel.remove();
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
            // Suggestion rejected — reveal the manual tag panel so the user can pick another label.
            const manualPanel = document.getElementById('sft-manual-panel-' + fileId);
            if (manualPanel) manualPanel.style.display = '';
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
        const observer = new MutationObserver(function() {
            const sidebar = document.querySelector('#app-sidebar-vue, #app-sidebar');
            if (!sidebar) return;

            // Check if sidebar is visible
            const isVisible = getComputedStyle(sidebar).display !== 'none' &&
                             getComputedStyle(sidebar).visibility !== 'hidden';

            if (!isVisible) {
                document.querySelectorAll('[id^="sft-manual-panel-"]').forEach(el => el.remove());
                document.querySelectorAll('[id^="sft-banner-"]').forEach(el => el.remove());
                window._sftLastFileId = null;
                return;
            }

            // Try to get file ID from the sidebar DOM
            const fileId = getFileIdFromDOM();
            if (!fileId) return;

            if (fileId !== window._sftLastFileId) {
                window._sftLastFileId = fileId;
                document.querySelectorAll('[id^="sft-manual-panel-"]').forEach(el => el.remove());
                document.querySelectorAll('[id^="sft-banner-"]').forEach(el => el.remove());
                loadSuggestion(fileId);
                renderManualTagPanel(fileId);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class', 'style', 'data-file-id', 'data-id']
        });

        window.addEventListener('popstate', function() {
            window._sftLastFileId = null;
            document.querySelectorAll('[id^="sft-manual-panel-"]').forEach(el => el.remove());
            document.querySelectorAll('[id^="sft-banner-"]').forEach(el => el.remove());
        });
    }

    function getFileIdFromDOM() {
        const sidebar = document.querySelector('#app-sidebar-vue, #app-sidebar');
        if (!sidebar) return null;

        // Method 1: data attribute directly on sidebar header children
        const header = sidebar.querySelector('[data-file-id], [data-id], [fileid]');
        if (header) {
            return header.dataset.fileId || header.dataset.id || header.getAttribute('fileid') || null;
        }

        // Method 2: active/selected row in the file list
        const activeRow = document.querySelector('.files-list__row--active, tr.selected');
        if (activeRow) {
            return activeRow.dataset.id || activeRow.getAttribute('data-id') || null;
        }

        // Method 3: any element inside sidebar carrying a file id
        const withFileId = sidebar.querySelector('[data-fileid], [data-file-id]');
        if (withFileId) {
            return withFileId.dataset.fileid || withFileId.dataset.fileId || null;
        }

        // Method 4: focused/hovered row in the file list table
        const selectedRow = document.querySelector('tr[data-id].mouseOver, tr[data-id]:focus-within');
        if (selectedRow) {
            return selectedRow.dataset.id || null;
        }

        return null;
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

    // Expose globals for manual debugging from the browser console
    window.sftInit = init;
    window.sftLoadSuggestion = loadSuggestion;
    window.sftRenderBanner = renderBanner;
    window.sftDebug = function() {
        console.log('Last file ID:', window._sftLastFileId);
        console.log('getFileIdFromDOM:', getFileIdFromDOM());
        const sidebar = document.querySelector('#app-sidebar-vue, #app-sidebar');
        console.log('Sidebar visible:', sidebar ? getComputedStyle(sidebar).display : 'NOT FOUND');
        if (sidebar) {
            console.log('Sidebar HTML (first 500 chars):', sidebar.innerHTML.substring(0, 500));
        }
    };

    // ── Manual Tag Panel ─────────────────────────────────────────────────────
    // Shows a dropdown with 5 fixed labels when no tag has been auto-applied.
    // Appears in the Nextcloud sidebar when a file is clicked.

    const FIXED_LABELS = [
        'Lecture Notes',
        'Problem Set',
        'Exam',
        'Reading',
        'Other'
    ];

    function renderManualTagPanel(fileId) {
        const existing = document.getElementById('sft-manual-panel-' + fileId);
        if (existing) existing.remove();

        const panel = document.createElement('div');
        panel.id = 'sft-manual-panel-' + fileId;
        panel.className = 'sft-manual-tag-panel';
        panel.innerHTML = `
            <div class="sft-manual-header">
                <span class="sft-icon">🏷️</span>
                <strong>Assign tag to file #${escapeHtml(fileId)}</strong>
            </div>
            <div class="sft-manual-row">
                <select id="sft-label-select-${fileId}" class="sft-label-select">
                    <option value="">— Select a label —</option>
                    ${FIXED_LABELS.map(l => `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`).join('')}
                </select>
                <button class="sft-btn sft-btn-confirm" id="sft-manual-apply-${fileId}">
                    Apply
                </button>
            </div>
            <div id="sft-manual-result-${fileId}" class="sft-manual-result"></div>
        `;

        const targets = [
            '#app-sidebar-vue .app-sidebar-header',
            '#app-sidebar .app-sidebar-header',
            '.app-sidebar-header',
            '#app-sidebar-vue',
            '#app-sidebar',
        ];
        let injected = false;
        for (const sel of targets) {
            const el = document.querySelector(sel);
            if (el) {
                el.parentNode.insertBefore(panel, el.nextSibling);
                injected = true;
                break;
            }
        }
        if (!injected) {
            panel.style.cssText = 'position:fixed;top:130px;right:20px;z-index:9999;background:#fff;border:1px solid #ccc;padding:12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);max-width:300px;';
            document.body.appendChild(panel);
        }

        document.getElementById('sft-manual-apply-' + fileId).addEventListener('click', () => {
            const select = document.getElementById('sft-label-select-' + fileId);
            const tag = select.value;
            if (!tag) {
                alert('Please select a label first.');
                return;
            }
            applyManualTag(fileId, tag);
        });
    }

    function applyManualTag(fileId, tag) {
        const resultEl = document.getElementById('sft-manual-result-' + fileId);
        if (resultEl) resultEl.innerHTML = '<em>Applying...</em>';

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
                const panel = document.getElementById('sft-manual-panel-' + fileId);
                if (panel) {
                    panel.innerHTML = `<div class="sft-confirmed">✅ Tag applied: <strong>${escapeHtml(tag)}</strong></div>`;
                }
            } else {
                if (resultEl) resultEl.innerHTML = '<span style="color:red">Failed: ' + escapeHtml(data.error || 'unknown error') + '</span>';
            }
        })
        .catch(() => {
            if (resultEl) resultEl.innerHTML = '<span style="color:red">Network error. Try again.</span>';
        });
    }

})();