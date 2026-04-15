<?php
/** @var \OCP\IL10N $l */
/** @var array $_ */
script('smartfiletagger', 'smart-tagger');
style('smartfiletagger', 'smart-tagger');
?>

<div id="sft-category-manager">

    <h2><?php p($l->t('My document categories')); ?></h2>
    <p class="sft-subtitle">
        <?php p($l->t(
            'Define your own categories with a few example files. ' .
            'Smart File Tagger will learn to recognize similar documents automatically.'
        )); ?>
    </p>

    <!-- Existing categories -->
    <div id="sft-category-list">
        <p class="sft-empty"><?php p($l->t('Loading categories...')); ?></p>
    </div>

    <!-- Create new category -->
    <form id="sft-create-category-form">
        <h3><?php p($l->t('Create a new category')); ?></h3>

        <label for="sft-category-name">
            <?php p($l->t('Category name')); ?>
        </label>
        <input
            type="text"
            id="sft-category-name"
            placeholder="<?php p($l->t('e.g. Client Contracts, Lab Reports, Tax Documents')); ?>"
            maxlength="64"
            required
        />

        <label>
            <?php p($l->t('Example files (select at least 3)')); ?>
        </label>
        <p class="sft-file-picker-hint">
            <?php p($l->t(
                'Choose files from your Nextcloud that are good examples of this category. ' .
                'The more examples you provide, the better the model will recognize future files.'
            )); ?>
        </p>

        <!--
            The file picker is wired up in smart-tagger.js using the
            Nextcloud OC.dialogs.filepicker() API. Selected file IDs
            populate hidden checkboxes here, which the form serializes.
        -->
        <div id="sft-selected-files">
            <p class="sft-empty" id="sft-no-files-hint">
                <?php p($l->t('No files selected yet.')); ?>
            </p>
        </div>

        <button type="button" id="sft-open-file-picker" class="sft-btn">
            <?php p($l->t('Browse files')); ?>
        </button>

        <button type="submit" class="sft-form-submit">
            <?php p($l->t('Create category')); ?>
        </button>
    </form>

</div>