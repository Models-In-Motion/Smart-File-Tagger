<?php
namespace OCA\SmartFileTagger\Controller;

use OCP\AppFramework\Controller;
use OCP\AppFramework\Http\JSONResponse;
use OCP\AppFramework\Http\TemplateResponse;
use OCP\Files\File;
use OCP\Files\IRootFolder;
use OCP\IRequest;
use OCP\IUserSession;

class CategoryController extends Controller {
    private IUserSession $userSession;
    private IRootFolder $rootFolder;
    private string $sidecarUrl;

    public function __construct(
        string $appName,
        IRequest $request,
        IUserSession $userSession,
        IRootFolder $rootFolder
    ) {
        parent::__construct($appName, $request);
        $this->userSession = $userSession;
        $this->rootFolder = $rootFolder;
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
        $exampleFileIds = $this->request->getParam('exampleFileIds', $this->request->getParam('exampleFileIds[]', []));

        if (is_string($exampleFileIds)) {
            $exampleFileIds = array_map('trim', explode(',', $exampleFileIds));
        }
        if (!is_array($exampleFileIds)) {
            $exampleFileIds = [];
        }
        $exampleFileIds = array_values(array_filter(array_map('strval', $exampleFileIds), static fn(string $id): bool => $id !== ''));

        if ($categoryName === '' || count($exampleFileIds) < 3) {
            return new JSONResponse([
                'success' => false,
                'error' => 'Category name and at least 3 example files required'
            ], 400);
        }

        // Read actual file content from Nextcloud by file ID.
        // This is more reliable than resolving file IDs via OCS + WebDAV.
        $exampleTexts = [];
        foreach ($exampleFileIds as $fileId) {
            // First try to get already-extracted text from the predictions DB via FastAPI.
            // This avoids re-reading raw PDF binary which is slow and binary garbage for SBERT.
            $extracted = $this->sidecarGet('/extracted-text/' . urlencode($fileId) . '?user_id=' . urlencode($userId));
            if ($extracted && !empty($extracted['extracted_text']) && strlen($extracted['extracted_text']) > 10) {
                $exampleTexts[] = mb_substr(mb_convert_encoding($extracted['extracted_text'], 'UTF-8', 'UTF-8'), 0, 1000);
            } else {
                // Fall back to reading raw file content if not in predictions DB.
                $text = $this->readTextFromFileId($userId, $fileId);
                if ($text !== null && strlen($text) > 10) {
                    $exampleTexts[] = mb_substr(mb_convert_encoding($text, 'UTF-8', 'UTF-8'), 0, 1000);
                }
            }
        }

        if (count($exampleTexts) < 3) {
            return new JSONResponse([
                'success' => false,
                'error' => 'Could not fetch content for at least 3 files. Got: ' . count($exampleTexts)
            ], 400);
        }

        $payload = [
            'user_id' => $userId,
            'category_name' => $categoryName,
            'example_texts' => $exampleTexts,
        ];

        $response = $this->sidecarPost('/register-category', $payload);
        if (!empty($response['success'])) {
            // Retroactively apply the new category tag to the example files
            foreach ($exampleFileIds as $fileId) {
                $extracted = $this->sidecarGet('/extracted-text/' . urlencode($fileId) . '?user_id=' . urlencode($userId));
                if (empty($extracted['extracted_text'])) continue;

                // Re-run prediction now that category exists
                $predictPayload = [
                    'user_id'  => $userId,
                    'file_id'  => $fileId,
                    'text'     => mb_substr($extracted['extracted_text'], 0, 1000),
                ];
                $prediction = $this->sidecarPost('/predict-text', $predictPayload);
                if (!empty($prediction['predicted_tag']) && $prediction['predicted_tag'] === $categoryName) {
                    // Remove any existing suggested tag and apply the confirmed category tag
                    // We use the tag mapper directly via the file ID
                    try {
                        $tag = $this->tagManager->createTag($categoryName, true, true);
                        $this->tagMapper->assignTags((string)$fileId, 'files', [$tag->getId()]);
                    } catch (\Throwable $e) {
                        // Tag might already exist or assignment might fail — not critical
                    }
                }
            }
        }
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

    private function readTextFromFileId(string $userId, string $fileId): ?string {
        if ($fileId === '' || !ctype_digit($fileId)) {
            return null;
        }

        try {
            $userFolder = $this->rootFolder->getUserFolder($userId);
            $nodes = $userFolder->getById((int)$fileId);
            foreach ($nodes as $node) {
                if (!$node instanceof File) {
                    continue;
                }
                $stream = $node->fopen('r');
                if (!is_resource($stream)) {
                    continue;
                }
                $content = stream_get_contents($stream);
                fclose($stream);
                if (!is_string($content) || $content === '') {
                    continue;
                }
                // Keep payload size bounded for category registration.
                return mb_substr($content, 0, 12000);
            }
        } catch (\Throwable $e) {
            \OC::$server->getLogger()->warning(
                'SmartFileTagger category register: could not read file content',
                [
                    'app' => 'smartfiletagger',
                    'user' => $userId,
                    'file_id' => $fileId,
                    'exception' => $e,
                ]
            );
        }

        return null;
    }
}