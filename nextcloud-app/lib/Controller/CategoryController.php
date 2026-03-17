<?php

namespace OCA\SmartFileTagger\Controller;

use OCP\AppFramework\Controller;
use OCP\AppFramework\Http\JSONResponse;
use OCP\AppFramework\Http\TemplateResponse;
use OCP\IRequest;
use OCP\IUserSession;

/**
 * CategoryController
 *
 * Handles the "Create your own categories" feature.
 * Proxies to sidecar POST /register-category and GET /categories.
 */
class CategoryController extends Controller {

    private IUserSession $userSession;
    private string $sidecarUrl;

    public function __construct(
        string $appName,
        IRequest $request,
        IUserSession $userSession
    ) {
        parent::__construct($appName, $request);
        $this->userSession = $userSession;
        $this->sidecarUrl  = getenv('SIDECAR_URL') ?: 'http://sidecar:8000';
    }

    /**
     * GET /apps/smartfiletagger/categories
     * Renders the category manager settings page.
     *
     * @NoAdminRequired
     */
    public function index(): TemplateResponse {
        return new TemplateResponse('smartfiletagger', 'category-manager');
    }

    /**
     * POST /apps/smartfiletagger/categories/register
     *
     * Payload: { "categoryName": "Client Contracts", "exampleFileIds": ["1","2","3"] }
     * Reads file content for each example, forwards to sidecar /register-category.
     *
     * @NoAdminRequired
     */
    public function register(): JSONResponse {
        $userId       = $this->userSession->getUser()->getUID();
        $categoryName = $this->request->getParam('categoryName');
        $fileIds      = $this->request->getParam('exampleFileIds', []);

        if (!$categoryName || count($fileIds) < 3) {
            return new JSONResponse([
                'error' => 'Provide a category name and at least 3 example files'
            ], 400);
        }

        // Forward to sidecar — sidecar handles text extraction from file IDs
        $response = $this->callSidecar('/register-category', [
            'user_id'       => $userId,
            'category_name' => $categoryName,
            'file_ids'      => $fileIds,
        ]);

        if ($response === null) {
            return new JSONResponse(['error' => 'sidecar_unavailable'], 503);
        }

        return new JSONResponse($response);
    }

    /**
     * DELETE /apps/smartfiletagger/categories/{name}
     *
     * @NoAdminRequired
     */
    public function delete(string $name): JSONResponse {
        $userId   = $this->userSession->getUser()->getUID();
        $response = $this->callSidecar('/delete-category', [
            'user_id'       => $userId,
            'category_name' => $name,
        ]);

        if ($response === null) {
            return new JSONResponse(['error' => 'sidecar_unavailable'], 503);
        }

        return new JSONResponse($response);
    }

    private function callSidecar(string $endpoint, array $payload): ?array {
        $url = rtrim($this->sidecarUrl, '/') . $endpoint;
        $ch  = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => json_encode($payload),
            CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 10,
        ]);
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($response === false || $httpCode !== 200) {
            return null;
        }

        return json_decode($response, true);
    }
}