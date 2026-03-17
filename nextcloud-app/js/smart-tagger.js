/**
 * smart-tagger.js
 *
 * Injected into the Nextcloud Files UI via the App Framework script hook.
 * Responsibilities:
 *   1. Listen for pending suggestion notifications
 *   2. Render an Accept / Reject banner in the file detail sidebar
 *   3. Send confirm/reject back to the PHP controller
 *   4. Render the LLM explanation if present
 */

(function (OCA, OC, $) {
    'use strict';

    // -------------------------------------------------------------------------
    // Suggestion banner — shown in the file detail sidebar (right panel)
    // when the model confidence is between 0.50 and 0.85
    // -------------------------------------------------------------------------

    const SmartTaggerSidebar = {

        /**
         * Called by Nextcloud when the file detail sidebar opens.
         * We check if there is a pending suggestion for this file.
         */
        attach: function (fileInfo) {
            const fileId = fileInfo.get('id');
            SmartTaggerSidebar.loadPendingSuggestion(fileId);
        },

        loadPendingSuggestion: function (fileId) {
            $.get(
                OC.generateUrl('/apps/smartfiletagger/suggestion/' + fileId),
                function (data) {
                    if (data && data.tag) {
                        SmartTaggerSidebar.renderBanner(fileId, data);
                    }
                }
            );
        },

        renderBanner: function (fileId, data) {
            const confidence = Math.round(data.confidence * 100);
            const explanation = data.explanation
                ? `<p class="sft-explanation">${escapeHtml(data.explanation)}</p>`
                : '';

            const banner = $(`
                <div class="sft-suggestion-banner" id="sft-banner-${fileId}">
                    <div class="sft-banner-header">
                        <span class="sft-icon">&#128196;</span>
                        <strong>Suggested tag:</strong>
                        <span class="sft-tag-pill">${escapeHtml(data.tag)}</span>
                        <span class="sft-confidence">${confidence}% confidence</span>
                    </div>
                    ${explanation}
                    <div class="sft-actions">
                        <button class="sft-btn sft-btn-confirm" data-file-id="${fileId}" data-tag="${escapeHtml(data.tag)}">
                            Apply tag
                        </button>
                        <button class="sft-btn sft-btn-reject" data-file-id="${fileId}" data-tag="${escapeHtml(data.tag)}">
                            Not this tag
                        </button>
                    </div>
                </div>
            `);

            // Inject at the top of the sidebar details panel
            $('#app-sidebar .detailFileInfoContainer').prepend(banner);

            // Wire up button actions
            banner.find('.sft-btn-confirm').on('click', SmartTaggerSidebar.handleConfirm);
            banner.find('.sft-btn-reject').on('click', SmartTaggerSidebar.handleReject);
        },

        handleConfirm: function () {
            const fileId = $(this).data('file-id');
            const tag    = $(this).data('tag');

            $.post(OC.generateUrl('/apps/smartfiletagger/confirm'), { fileId, tag })
                .done(function () {
                    $(`#sft-banner-${fileId}`).replaceWith(
                        `<div class="sft-confirmed">Tag applied: <strong>${escapeHtml(tag)}</strong></div>`
                    );
                    // Refresh the sidebar tags list so the new tag appears
                    OCA.Files.Sidebar.reload();
                })
                .fail(function () {
                    OC.Notification.showTemporary(t('smartfiletagger', 'Could not apply tag. Please try again.'));
                });
        },

        handleReject: function () {
            const fileId = $(this).data('file-id');
            const tag    = $(this).data('tag');

            $.post(OC.generateUrl('/apps/smartfiletagger/reject'), { fileId, tag })
                .done(function () {
                    $(`#sft-banner-${fileId}`).remove();
                    OC.Notification.showTemporary(t('smartfiletagger', 'Feedback saved. Tag manually if needed.'));
                });
        },
    };

    // -------------------------------------------------------------------------
    // Register as a Nextcloud Files sidebar plugin
    // Nextcloud calls attach() whenever a file is selected in the Files UI
    // -------------------------------------------------------------------------
    OCA.Files.fileActions.registerAction({
        name: 'SmartTaggerInfo',
        displayName: t('smartfiletagger', 'Smart Tag Info'),
        mime: 'all',
        permissions: OC.PERMISSION_READ,
        icon: OC.imagePath('smartfiletagger', 'tag'),
        actionHandler: function (filename, context) {
            const fileInfo = context.fileList.getModelForFile(filename);
            SmartTaggerSidebar.attach(fileInfo);
        }
    });

    // Also hook into sidebar open event so the banner loads automatically
    $(document).on('click', '.filename', function () {
        const fileId = $(this).closest('tr').data('id');
        if (fileId) {
            SmartTaggerSidebar.loadPendingSuggestion(fileId);
        }
    });

    // -------------------------------------------------------------------------
    // Category manager page — loaded at /apps/smartfiletagger/categories
    // Lets users create custom categories with example files
    // -------------------------------------------------------------------------

    const CategoryManager = {

        init: function () {
            if (!$('#sft-category-manager').length) return;
            CategoryManager.loadCategories();
            $('#sft-create-category-form').on('submit', CategoryManager.handleCreate);
        },

        loadCategories: function () {
            $.get(OC.generateUrl('/apps/smartfiletagger/categories'), function (data) {
                const list = $('#sft-category-list');
                list.empty();
                if (!data.categories || data.categories.length === 0) {
                    list.append('<p class="sft-empty">No categories yet. Create one below.</p>');
                    return;
                }
                data.categories.forEach(function (cat) {
                    list.append(`
                        <div class="sft-category-item">
                            <span class="sft-tag-pill">${escapeHtml(cat.name)}</span>
                            <span class="sft-example-count">${cat.example_count} examples</span>
                            <button class="sft-btn-delete" data-name="${escapeHtml(cat.name)}">Remove</button>
                        </div>
                    `);
                });
                list.find('.sft-btn-delete').on('click', CategoryManager.handleDelete);
            });
        },

        handleCreate: function (e) {
            e.preventDefault();
            const name    = $('#sft-category-name').val().trim();
            const fileIds = CategoryManager.getSelectedFileIds();

            if (!name) {
                OC.Notification.showTemporary(t('smartfiletagger', 'Please enter a category name.'));
                return;
            }
            if (fileIds.length < 3) {
                OC.Notification.showTemporary(t('smartfiletagger', 'Please select at least 3 example files.'));
                return;
            }

            $.post(
                OC.generateUrl('/apps/smartfiletagger/categories/register'),
                { categoryName: name, exampleFileIds: fileIds }
            )
            .done(function () {
                OC.Notification.showTemporary(t('smartfiletagger', 'Category created successfully.'));
                $('#sft-category-name').val('');
                CategoryManager.clearSelectedFiles();
                CategoryManager.loadCategories();
            })
            .fail(function () {
                OC.Notification.showTemporary(t('smartfiletagger', 'Failed to create category. Please try again.'));
            });
        },

        handleDelete: function () {
            const name = $(this).data('name');
            OC.dialogs.confirm(
                t('smartfiletagger', 'Delete category "{name}"? This cannot be undone.'.replace('{name}', name)),
                t('smartfiletagger', 'Delete category'),
                function (confirmed) {
                    if (!confirmed) return;
                    $.ajax({
                        url: OC.generateUrl('/apps/smartfiletagger/categories/' + encodeURIComponent(name)),
                        type: 'DELETE',
                    }).done(function () {
                        CategoryManager.loadCategories();
                    });
                }
            );
        },

        getSelectedFileIds: function () {
            return $('.sft-example-file:checked').map(function () {
                return $(this).val();
            }).get();
        },

        clearSelectedFiles: function () {
            $('.sft-example-file').prop('checked', false);
        },
    };

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // Init on DOM ready
    $(document).ready(function () {
        CategoryManager.init();
    });

})(OCA, OC, jQuery);