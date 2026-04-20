<?php

return [
    'routes' => [
        // Called by Nextcloud Flow when a file is uploaded
        // Forwards to sidecar /predict and writes tag back
        ['name' => 'tag#predict',          'url' => '/predict',              'verb' => 'POST'],

        // Called by JS when user clicks Accept on a suggestion
        ['name' => 'tag#confirm',          'url' => '/confirm',              'verb' => 'POST'],

        // Called by JS when user clicks Reject on a suggestion
        ['name' => 'tag#reject',           'url' => '/reject',               'verb' => 'POST'],

        // Called by JS to load pending suggestion for file
        ['name' => 'tag#suggestion',       'url' => '/suggestion/{fileId}',  'verb' => 'GET'],

        // Category manager — renders the settings page
        ['name' => 'category#index',       'url' => '/categories',           'verb' => 'GET'],

        // Called when user submits a new category + examples
        ['name' => 'category#register',    'url' => '/categories/register',  'verb' => 'POST'],

        // Lists all categories defined by the current user
        ['name' => 'category#list',        'url' => '/categories',           'verb' => 'GET'],

        // Deletes a user-defined category
        ['name' => 'category#delete',      'url' => '/categories/{name}',    'verb' => 'DELETE'],
    ]
];