<?php
namespace OCA\SmartFileTagger\Controller;

use OCP\AppFramework\Controller;
use OCP\AppFramework\Http\JSONResponse;
use OCP\AppFramework\Http\TemplateResponse;
use OCP\IRequest;
use OCP\IUserSession;

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
        $this->sidecarUrl = getenv('SIDECAR_URL') ?: 'http://sidecar:8000';
    }

    /**
     * @NoAdminRequired
     * @NoCSRFRequired
     */
    public function index(): \OCP\AppFramework\Http\TemplateResponse {
        return new \OCP\AppFramework\Http\TemplateResponse('smartfiletagger', 'category-manager', [
            'user_id' => $this->getUserId(),
        ]);
    }

    /**
     * @NoAdminRequired
     * @NoCSRFRequired
     */
    public function list(): JSONResponse {
        $userId = $this->getUserId();
        $response = $this->sidecarGet('/categories?user_id=' . urlencode($userId));
        return new JSONResponse($response ?? ['categories' => [], 'count' => 0]);
    }

    /**
     * @NoAdminRequired
     * @NoCSRFRequired
     */
    public function register(): JSONResponse {
        $userId = $this->getUserId();
        $categoryName = (string)$this->request->getParam('categoryName', '');
        $exampleFileIds = $this->request->getParam('exampleFileIds', []);

        if ($categoryName === '' || count($exampleFileIds) < 3) {
            return new JSONResponse([
                'success' => false,
                'error' => 'Category name and at least 3 example files required'
            ], 400);
        }

        // Fetch text from each example file via WebDAV and send to sidecar
        $exampleTexts = [];
        foreach ($exampleFileIds as $fileId) {
            // For simplicity, use the file ID as a placeholder text
            // In production, fetch actual file content via WebDAV
            $exampleTexts[] = "example file " . $fileId;
        }

        $payload = [
            'user_id' => $userId,
            'category_name' => $categoryName,
            'example_texts' => $exampleTexts,
        ];

        $response = $this->sidecarPost('/register-category', $payload);
        return new JSONResponse($response ?? ['success' => false, 'error' => 'Sidecar unavailable']);
    }

    /**
     * @NoAdminRequired
     * @NoCSRFRequired
     */
    public function delete(string $name): JSONResponse {
        $userId = $this->getUserId();
        $payload = ['user_id' => $userId, 'category_name' => $name];
        $response = $this->sidecarDelete('/delete-category', $payload);
        return new JSONResponse($response ?? ['success' => false]);
    }

    private function getUserId(): string {
        $user = $this->userSession->getUser();
        return $user ? $user->getUID() : '';
    }

    private function sidecarGet(string $endpoint): ?array {
        $url = rtrim($this->sidecarUrl, '/') . $endpoint;
        $context = stream_context_create([
            'http' => ['method' => 'GET', 'timeout' => 10, 'ignore_errors' => true]
        ]);
        $response = @file_get_contents($url, false, $context);
        if ($response === false) return null;
        $decoded = json_decode($response, true);
        return is_array($decoded) ? $decoded : null;
    }

    private function sidecarPost(string $endpoint, array $payload): ?array {
        $url = rtrim($this->sidecarUrl, '/') . $endpoint;
        $body = json_encode($payload);
        $context = stream_context_create([
            'http' => [
                'method' => 'POST',
                'header' => ['Content-Type: application/json', 'Content-Length: ' . strlen($body)],
                'content' => $body,
                'timeout' => 10,
                'ignore_errors' => true,
            ]
        ]);
        $response = @file_get_contents($url, false, $context);
        if ($response === false) return null;
        $decoded = json_decode($response, true);
        return is_array($decoded) ? $decoded : null;
    }

    private function sidecarDelete(string $endpoint, array $payload): ?array {
        $url = rtrim($this->sidecarUrl, '/') . $endpoint;
        $body = json_encode($payload);
        $context = stream_context_create([
            'http' => [
                'method' => 'DELETE',
                'header' => ['Content-Type: application/json', 'Content-Length: ' . strlen($body)],
                'content' => $body,
                'timeout' => 10,
                'ignore_errors' => true,
            ]
        ]);
        $response = @file_get_contents($url, false, $context);
        if ($response === false) return null;
        $decoded = json_decode($response, true);
        return is_array($decoded) ? $decoded : null;
    }
}