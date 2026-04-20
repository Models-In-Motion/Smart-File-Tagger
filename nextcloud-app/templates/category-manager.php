<?php
/** @var array $_ */
script('smartfiletagger', 'smart-tagger');
style('smartfiletagger', 'smart-tagger');
?>
<div id="app">
    <div id="app-content">
        <div id="sft-category-manager" class="sft-container">
            <h2>Smart File Tagger — Custom Categories</h2>
            <p class="sft-description">
                Create custom categories by selecting 3 or more example files.
                The model will learn from these examples and tag similar files automatically.
            </p>

            <div id="sft-category-list">
                <p class="sft-empty">Loading categories...</p>
            </div>

            <hr/>

            <h3>Create a new category</h3>
            <form id="sft-create-category-form">
                <div class="sft-form-row">
                    <label for="sft-category-name">Category name:</label>
                    <input type="text" id="sft-category-name" placeholder="e.g. Lab Report" required/>
                </div>
                <div class="sft-form-row">
                    <label>Example file IDs (comma separated, min 3):</label>
                    <input type="text" id="sft-example-ids" placeholder="e.g. 42,57,89"/>
                    <small>Find file IDs in the URL when you click a file in Nextcloud (e.g. /f/42)</small>
                </div>
                <button type="submit" class="button primary">Create category</button>
            </form>
        </div>
    </div>
</div>