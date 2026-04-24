<?php
return [
    'routes' => [
        // Flow webhook — called by Nextcloud when file is uploaded
        ['name' => 'tag#predict',       'url' => '/predict',                'verb' => 'POST'],
        // Suggestion banner endpoints
        ['name' => 'tag#suggestion',    'url' => '/suggestion/{fileId}',    'verb' => 'GET'],
        ['name' => 'tag#confirm',       'url' => '/confirm',                'verb' => 'POST'],
        ['name' => 'tag#reject',        'url' => '/reject',                 'verb' => 'POST'],
        ['name' => 'tag#manualTag',     'url' => '/manual-tag',             'verb' => 'POST'],
        // Category manager page
        ['name' => 'category#index',    'url' => '/categories',             'verb' => 'GET'],
        // Category list API endpoint
        ['name' => 'category#list',     'url' => '/categories/list',        'verb' => 'GET'],
        // Category API endpoints
        ['name' => 'category#register', 'url' => '/categories/register',    'verb' => 'POST'],
        ['name' => 'category#delete',   'url' => '/categories/{name}',      'verb' => 'DELETE'],
    ]
];