<?php

namespace OCA\SmartFileTagger\Controller;

use OCP\AppFramework\Controller;
use OCP\AppFramework\Http\JSONResponse;
use OCP\IRequest;
use OCP\IUserSession;
use OCP\Notification\IManager as INotificationManager;
use OCP\SystemTag\ISystemTagManager;
use OCP\SystemTag\ISystemTagObjectMapper;

class TagController extends Controller {
    private ISystemTagManager $tagManager;
    private ISystemTagObjectMapper $tagMapper;
    private IUserSession $userSession;
    private INotificationManager $notificationManager;
    private string $sidecarUrl;
    private string $nextcloudInternalUrl;
    private string $suggestionStorePath;

    public function __construct(
        string $appName,
        IRequest $request,
        ISystemTagManager $tagManager,
        ISystemTagObjectMapper $tagMapper,
        IUserSession $userSession,
        INotificationManager $notificationManager
    ) {
        parent::__construct($appName, $request);
        $this->tagManager = $tagManager;
        $this->tagMapper = $tagMapper;
        $this->userSession = $userSession;
        $this->notificationManager = $notificationManager;
        $this->sidecarUrl = getenv('SIDECAR_URL') ?: 'http://sidecar:8000';
        $this->nextcloudInternalUrl = getenv('NEXTCLOUD_INTERNAL_URL') ?: 'http://nextcloud';
        $this->suggestionStorePath = sys_get_temp_dir() . '/smartfiletagger_suggestions.json';
    }

    /**
     * @NoAdminRequired
     * @NoCSRFRequired
     * @PublicPage
     */
    public function predict(): JSONResponse {
        try {
            // Try multiple ways to read the request body
            // Nextcloud Webhooks app may send different content types
            $payload = null;

            // Attempt 1: getParams() - works for form-encoded and some JSON
            $params = $this->request->getParams();

            // Attempt 2: raw input stream
            $rawBody = file_get_contents('php://input');

            // Attempt 3: try JSON decode of raw body
            if (!empty($rawBody)) {
                $decoded = json_decode($rawBody, true);
                if (is_array($decoded)) {
                    $payload = $decoded;
                }
            }

            // Fall back to getParams() if raw body didn't work
            if ($payload === null && !empty($params)) {
                $payload = $params;
                // params might have JSON as a string value, try to decode
                foreach ($params as $key => $value) {
                    if (is_string($value)) {
                        $decoded = json_decode($value, true);
                        if (is_array($decoded)) {
                            $payload = $decoded;
                            break;
                        }
                    }
                }
            }

            // Log everything for debugging
            \OC::$server->getLogger()->warning(
                'SmartFileTagger debug - rawBody: ' . substr($rawBody, 0, 500) .
                ' | params: ' . json_encode($params) .
                ' | payload: ' . json_encode($payload),
                ['app' => 'smartfiletagger']
            );

            if ($payload === null || !is_array($payload)) {
                return new JSONResponse([
                    'success' => false,
                    'error' => 'Could not parse payload. rawBody: ' . substr($rawBody, 0, 200) . ' params: ' . json_encode($params),
                ], 400);
            }

            // Support both payload formats:
            // Format 1 (Nextcloud Webhooks app): flat with "node" at top level
            // Format 2 (manual/legacy): nested under "event.node" with "user.uid"
            if (isset($payload['node'])) {
                // Format 1 — Nextcloud Webhooks app format
                $node = $payload['node'];
                $filePath = (string)($node['path'] ?? '');
                $fileId = (string)($node['id'] ?? '');
                // Extract user from path: /admin/files/sol_debug.txt -> admin
                $pathParts = explode('/', ltrim($filePath, '/'));
                $userId = $pathParts[0] ?? 'admin';
            } else {
                // Format 2 — legacy/manual format
                $node = $payload['event']['node'] ?? [];
                $user = $payload['user'] ?? [];
                $fileId = (string)($node['id'] ?? '');
                $filePath = (string)($node['path'] ?? '');
                $userId = (string)($user['uid'] ?? '');
            }

            $fileName = (string)($node['name'] ?? basename($filePath));

            if ($fileId === '' || $filePath === '' || $userId === '') {
                return new JSONResponse([
                    'success' => false,
                    'error' => 'Missing file_id, file_path, or user_id. Got: ' . json_encode($payload),
                ], 400);
            }

            [$fileBytes, $resolvedFileName] = $this->fetchFileFromWebDav($userId, $filePath, $fileName);

            $prediction = $this->callSidecarPredict($fileBytes, $resolvedFileName, $userId, (string)$fileId);
            $predictedTag = (string)($prediction['predicted_tag'] ?? '');
            $confidence = (float)($prediction['confidence'] ?? 0.0);
            $action = (string)($prediction['action'] ?? 'no_tag');

            if ($action === 'auto_apply' && $predictedTag !== '') {
                $this->applyTag((string)$fileId, $predictedTag);
                error_log("SmartFileTagger auto-applied tag '{$predictedTag}' to file {$fileId}");
            } elseif ($action === 'suggest' && $predictedTag !== '') {
                $this->storePendingSuggestion($userId, (string)$fileId, [
                    'file_id' => (string)$fileId,
                    'file_path' => $filePath,
                    'predicted_tag' => $predictedTag,
                    'confidence' => $confidence,
                    'action' => $action,
                    'model_response' => $prediction,
                    'created_at' => gmdate('c'),
                ]);
                $this->pushSuggestionNotification($userId, (string)$fileId, $filePath, $predictedTag, $confidence);
                error_log("SmartFileTagger stored suggestion '{$predictedTag}' for file {$fileId}");
            } else {
                error_log("SmartFileTagger no_tag for file {$fileId}");
            }

            return new JSONResponse([
                'success' => true,
                'file_id' => (string)$fileId,
                'user_id' => $userId,
                'predicted_tag' => $predictedTag,
                'confidence' => $confidence,
                'action' => $action,
                'model_response' => $prediction,
            ]);
        } catch (\Throwable $e) {
            \OC::$server->getLogger()->error(
                'SmartFileTagger predict webhook failed: ' . $e->getMessage(),
                ['app' => 'smartfiletagger', 'exception' => $e]
            );
            return new JSONResponse([
                'success' => false,
                'error' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * @NoAdminRequired
     */
    public function confirm(): JSONResponse {
        $fileId  = (string)$this->request->getParam('fileId');
        $tag     = (string)$this->request->getParam('tag');
        $user = $this->userSession->getUser();
        $userId = $user ? $user->getUID() : '';

        if ($fileId === '' || $tag === '' || $userId === '') {
            return new JSONResponse(['success' => false, 'error' => 'Missing fileId/tag/user'], 400);
        }

        $this->applyTag($fileId, $tag);
        $this->callSidecarJson('/feedback', [
            'file_id'       => $fileId,
            'predicted_tag' => $tag,
            'correct_tag'   => $tag,
            'accepted'      => true,
            'user_id'       => $userId,
        ]);

        return new JSONResponse(['success' => true, 'status' => 'confirmed', 'tag' => $tag]);
    }

    /**
     * @NoAdminRequired
     * @NoCSRFRequired
     */
    public function suggestion(string $fileId): JSONResponse {
        $user = $this->userSession->getUser();
        $userId = $user ? $user->getUID() : '';

        if ($userId === '') {
            return new JSONResponse(['tag' => null], 200);
        }

        // Read from suggestion store
        if (!is_file($this->suggestionStorePath)) {
            return new JSONResponse(['tag' => null], 200);
        }

        $raw = @file_get_contents($this->suggestionStorePath);
        $data = json_decode((string)$raw, true);

        $suggestion = $data[$userId][$fileId] ?? null;
        if (!$suggestion) {
            return new JSONResponse(['tag' => null], 200);
        }

        return new JSONResponse([
            'tag' => $suggestion['predicted_tag'],
            'confidence' => $suggestion['confidence'],
            'explanation' => null,
            'file_id' => $fileId,
        ]);
    }

    /**
     * @NoAdminRequired
     */
    public function reject(): JSONResponse {
        $fileId = (string)$this->request->getParam('fileId');
        $tag = (string)$this->request->getParam('tag');
        $correctTag = (string)$this->request->getParam('correctTag', '');
        $user = $this->userSession->getUser();
        $userId = $user ? $user->getUID() : '';

        if ($fileId === '' || $tag === '' || $userId === '') {
            return new JSONResponse(['success' => false, 'error' => 'Missing fileId/tag/user'], 400);
        }

        $this->callSidecarJson('/feedback', [
            'file_id'       => $fileId,
            'predicted_tag' => $tag,
            'correct_tag'   => $correctTag !== '' ? $correctTag : null,
            'accepted'      => false,
            'user_id'       => $userId,
        ]);

        return new JSONResponse(['success' => true, 'status' => 'rejected']);
    }

    private function fetchFileFromWebDav(string $userId, string $filePath, string $fallbackName): array {
        $normalizedPath = ltrim($filePath, '/');
        $userPrefix = $userId . '/files/';
        if (str_starts_with($normalizedPath, $userPrefix)) {
            $normalizedPath = substr($normalizedPath, strlen($userPrefix));
        } elseif (str_starts_with($normalizedPath, 'files/')) {
            $normalizedPath = substr($normalizedPath, strlen('files/'));
        } elseif (str_starts_with($normalizedPath, $userId . '/')) {
            $normalizedPath = substr($normalizedPath, strlen($userId . '/'));
        }

        $pathParts = array_filter(explode('/', $normalizedPath), static fn(string $part): bool => $part !== '');
        $encodedPath = implode('/', array_map('rawurlencode', $pathParts));
        $url = rtrim($this->nextcloudInternalUrl, '/') . '/remote.php/dav/files/' . rawurlencode($userId);
        if ($encodedPath !== '') {
            $url .= '/' . $encodedPath;
        }

        $ncUser = getenv('NEXTCLOUD_ADMIN_USER') ?: 'admin';
        $ncPass = getenv('NEXTCLOUD_ADMIN_PASSWORD') ?: 'admin';
        $headers = "Authorization: Basic " . base64_encode($ncUser . ':' . $ncPass) . "\r\n";

        $context = stream_context_create([
            'http' => [
                'method' => 'GET',
                'header' => $headers,
                'timeout' => 30,
                'ignore_errors' => true,
            ],
        ]);

        $data = @file_get_contents($url, false, $context);
        $statusCode = $this->extractHttpStatusCode($http_response_header ?? []);

        if ($data === false || $statusCode < 200 || $statusCode >= 300) {
            throw new \RuntimeException('Failed to fetch file from WebDAV (status ' . $statusCode . ')');
        }

        $filename = $fallbackName !== '' ? $fallbackName : basename($normalizedPath);
        return [$data, $filename];
    }

    private function callSidecarPredict(string $fileBytes, string $filename, string $userId, string $fileId): array {
        $url = rtrim($this->sidecarUrl, '/') . '/predict';
        $boundary = '----SmartFileTaggerBoundary' . md5((string)microtime(true));
        $eol = "\r\n";

        $body = '';
        $body .= '--' . $boundary . $eol;
        $body .= 'Content-Disposition: form-data; name="file"; filename="' . addslashes($filename) . '"' . $eol;
        $body .= 'Content-Type: application/octet-stream' . $eol . $eol;
        $body .= $fileBytes . $eol;
        $body .= '--' . $boundary . $eol;
        $body .= 'Content-Disposition: form-data; name="user_id"' . $eol . $eol;
        $body .= $userId . $eol;
        $body .= '--' . $boundary . $eol;
        $body .= 'Content-Disposition: form-data; name="file_id"' . $eol . $eol;
        $body .= (string)$fileId . $eol;
        $body .= '--' . $boundary . '--' . $eol;

        $context = stream_context_create([
            'http' => [
                'method' => 'POST',
                'header' => [
                    'Content-Type: multipart/form-data; boundary=' . $boundary,
                    'Content-Length: ' . strlen($body),
                ],
                'content' => $body,
                'timeout' => 30,
                'ignore_errors' => true,
            ],
        ]);

        $response = @file_get_contents($url, false, $context);
        $statusCode = $this->extractHttpStatusCode($http_response_header ?? []);
        if ($response === false || $statusCode < 200 || $statusCode >= 300) {
            throw new \RuntimeException('Sidecar /predict request failed with status ' . $statusCode);
        }

        $decoded = json_decode($response, true);
        if (!is_array($decoded)) {
            throw new \RuntimeException('Invalid JSON response from sidecar /predict');
        }

        return $decoded;
    }

    private function callSidecarJson(string $endpoint, array $payload): ?array {
        $url = rtrim($this->sidecarUrl, '/') . $endpoint;
        $body = json_encode($payload);

        if ($body === false) {
            return null;
        }

        $context = stream_context_create([
            'http' => [
                'method' => 'POST',
                'header' => [
                    'Content-Type: application/json',
                    'Content-Length: ' . strlen($body),
                ],
                'content' => $body,
                'timeout' => 10,
                'ignore_errors' => true,
            ],
        ]);

        $response = @file_get_contents($url, false, $context);
        $statusCode = $this->extractHttpStatusCode($http_response_header ?? []);
        if ($response === false || $statusCode < 200 || $statusCode >= 300) {
            return null;
        }

        $decoded = json_decode($response, true);
        return is_array($decoded) ? $decoded : null;
    }

    private function applyTag(string $fileId, string $tagName): void {
        try {
            try {
                $tag = $this->tagManager->getTag($tagName, true, true);
            } catch (\OCP\SystemTag\TagNotFoundException $e) {
                $tag = $this->tagManager->createTag($tagName, true, true);
            }

            if (method_exists($this->tagManager, 'assignTags')) {
                $this->tagManager->assignTags((string)$fileId, 'files', [$tag->getId()]);
            } else {
                $this->tagMapper->assignTags((string)$fileId, 'files', [$tag->getId()]);
            }
        } catch (\Throwable $e) {
            \OC::$server->getLogger()->error(
                'SmartFileTagger: failed to apply tag ' . $tagName . ' to file ' . $fileId,
                ['app' => 'smartfiletagger', 'exception' => $e]
            );
        }
    }

    private function storePendingSuggestion(string $userId, string $fileId, array $suggestion): void {
        $data = [];
        if (is_file($this->suggestionStorePath)) {
            $raw = @file_get_contents($this->suggestionStorePath);
            $parsed = json_decode((string)$raw, true);
            if (is_array($parsed)) {
                $data = $parsed;
            }
        }

        if (!isset($data[$userId]) || !is_array($data[$userId])) {
            $data[$userId] = [];
        }
        $data[$userId][$fileId] = $suggestion;

        @file_put_contents($this->suggestionStorePath, json_encode($data, JSON_PRETTY_PRINT), LOCK_EX);
    }

    private function pushSuggestionNotification(
        string $userId,
        string $fileId,
        string $filePath,
        string $tag,
        float $confidence
    ): void {
        $notification = $this->notificationManager->createNotification();
        $notification
            ->setApp('smartfiletagger')
            ->setUser($userId)
            ->setDateTime(new \DateTime())
            ->setObject('file', $fileId)
            ->setSubject('suggested_tag', [
                'tag'        => $tag,
                'confidence' => round($confidence * 100),
                'filePath'   => $filePath,
            ]);

        $this->notificationManager->notify($notification);
    }

    private function extractHttpStatusCode(array $headers): int {
        if (count($headers) === 0) {
            return 0;
        }
        if (preg_match('/HTTP\/\d\.\d\s+(\d{3})/', (string)$headers[0], $matches) === 1) {
            return (int)$matches[1];
        }
        return 0;
    }
}
