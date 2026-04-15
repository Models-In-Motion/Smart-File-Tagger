<?php

namespace OCA\SmartFileTagger\Controller;

use OCP\AppFramework\Controller;
use OCP\AppFramework\Http\JSONResponse;
use OCP\IRequest;
use OCP\IUserSession;
use OCP\SystemTag\ISystemTagManager;
use OCP\SystemTag\ISystemTagObjectMapper;
use OCP\Files\IRootFolder;
use OCP\Notification\IManager as INotificationManager;

/**
 * TagController
 *
 * Sits between Nextcloud and the Python sidecar.
 * Responsibilities:
 *   1. Receive the Flow webhook when a file is uploaded
 *   2. Forward to sidecar POST /predict
 *   3a. If confidence > 0.85 → write tag immediately via ISystemTagManager
 *   3b. If confidence 0.5–0.85 → store pending suggestion, push notification to user
 *   4. Handle user confirm/reject actions from the JS frontend
 */
class TagController extends Controller {

    private ISystemTagManager $tagManager;
    private ISystemTagObjectMapper $tagMapper;
    private IRootFolder $rootFolder;
    private IUserSession $userSession;
    private INotificationManager $notificationManager;
    private string $sidecarUrl;

    public function __construct(
        string $appName,
        IRequest $request,
        ISystemTagManager $tagManager,
        ISystemTagObjectMapper $tagMapper,
        IRootFolder $rootFolder,
        IUserSession $userSession,
        INotificationManager $notificationManager
    ) {
        parent::__construct($appName, $request);
        $this->tagManager          = $tagManager;
        $this->tagMapper           = $tagMapper;
        $this->rootFolder          = $rootFolder;
        $this->userSession         = $userSession;
        $this->notificationManager = $notificationManager;

        // Sidecar URL is injected via environment variable in docker-compose
        // Falls back to localhost for local dev
        $this->sidecarUrl = getenv('SIDECAR_URL') ?: 'http://sidecar:8000';
    }

    /**
     * POST /apps/smartfiletagger/predict
     *
     * Called by Nextcloud Flow on every file upload.
     * Payload from Flow: { "fileId": "123", "filePath": "/admin/files/doc.pdf" }
     *
     * @NoAdminRequired
     * @NoCSRFRequired
     */
    public function predict(): JSONResponse {
        $fileId   = $this->request->getParam('fileId');
        $filePath = $this->request->getParam('filePath');
        $userId   = $this->userSession->getUser()->getUID();

        if (!$fileId || !$filePath) {
            return new JSONResponse(['error' => 'missing fileId or filePath'], 400);
        }

        // Forward to Python sidecar
        $sidecarResponse = $this->callSidecar('/predict', [
            'file_id'   => $fileId,
            'file_path' => $filePath,
            'user_id'   => $userId,
        ]);

        // Sidecar unavailable — fail open, do nothing
        if ($sidecarResponse === null) {
            return new JSONResponse(['status' => 'sidecar_unavailable'], 200);
        }

        $tag        = $sidecarResponse['tag']        ?? null;
        $confidence = $sidecarResponse['confidence'] ?? 0.0;
        $explanation= $sidecarResponse['explanation']?? null;

        // No tag predicted
        if (!$tag) {
            return new JSONResponse(['status' => 'no_prediction'], 200);
        }

        if ($confidence >= 0.85) {
            // High confidence — apply tag immediately
            $this->applyTag($fileId, $tag);
            return new JSONResponse(['status' => 'tagged', 'tag' => $tag]);
        }

        if ($confidence >= 0.50) {
            // Medium confidence — push a suggestion notification to the user
            $this->pushSuggestionNotification($userId, $fileId, $filePath, $tag, $confidence, $explanation);
            return new JSONResponse(['status' => 'suggested', 'tag' => $tag]);
        }

        // Low confidence — do nothing, let user tag manually
        return new JSONResponse(['status' => 'low_confidence'], 200);
    }

    /**
     * POST /apps/smartfiletagger/confirm
     *
     * User clicked "Yes" on a suggestion notification.
     * Applies the tag and logs positive feedback to sidecar.
     *
     * @NoAdminRequired
     */
    public function confirm(): JSONResponse {
        $fileId  = $this->request->getParam('fileId');
        $tag     = $this->request->getParam('tag');
        $userId  = $this->userSession->getUser()->getUID();

        $this->applyTag($fileId, $tag);

        // Log acceptance as positive feedback for retraining
        $this->callSidecar('/feedback', [
            'file_id'       => $fileId,
            'predicted_tag' => $tag,
            'correct_tag'   => $tag,
            'accepted'      => true,
            'user_id'       => $userId,
        ]);

        return new JSONResponse(['status' => 'confirmed', 'tag' => $tag]);
    }

    /**
     * POST /apps/smartfiletagger/reject
     *
     * User clicked "No" on a suggestion notification.
     * Logs negative feedback. User tags manually afterward.
     *
     * @NoAdminRequired
     */
    public function reject(): JSONResponse {
        $fileId      = $this->request->getParam('fileId');
        $tag         = $this->request->getParam('tag');
        $correctTag  = $this->request->getParam('correctTag'); // optional — user may specify
        $userId      = $this->userSession->getUser()->getUID();

        $this->callSidecar('/feedback', [
            'file_id'       => $fileId,
            'predicted_tag' => $tag,
            'correct_tag'   => $correctTag,
            'accepted'      => false,
            'user_id'       => $userId,
        ]);

        return new JSONResponse(['status' => 'rejected']);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Write a tag to a file using Nextcloud's ISystemTagManager.
     * Creates the tag if it doesn't already exist.
     */
    private function applyTag(string $fileId, string $tagName): void {
        try {
            // Get or create the tag (userVisible=true, userAssignable=true)
            try {
                $tag = $this->tagManager->getTag($tagName, true, true);
            } catch (\OCP\SystemTag\TagNotFoundException $e) {
                $tag = $this->tagManager->createTag($tagName, true, true);
            }

            // Map the tag to the file node
            $this->tagMapper->assignTags((string)$fileId, 'files', [$tag->getId()]);
        } catch (\Exception $e) {
            // Log but don't throw — tag failure must never crash the flow
            \OC::$server->getLogger()->error(
                'SmartFileTagger: failed to apply tag ' . $tagName . ' to file ' . $fileId,
                ['exception' => $e]
            );
        }
    }

    /**
     * Push a Nextcloud notification to the user with Accept/Reject actions.
     * The notification appears in the Nextcloud notification bell (top bar).
     */
    private function pushSuggestionNotification(
        string $userId,
        string $fileId,
        string $filePath,
        string $tag,
        float  $confidence,
        ?string $explanation
    ): void {
        $notification = $this->notificationManager->createNotification();
        $notification
            ->setApp('smartfiletagger')
            ->setUser($userId)
            ->setDateTime(new \DateTime())
            ->setObject('file', $fileId)
            ->setSubject('suggestion', [
                'tag'         => $tag,
                'confidence'  => round($confidence * 100),
                'explanation' => $explanation,
                'filePath'    => $filePath,
            ])
            // Two inline actions — rendered as buttons in the notification
            ->addAction(
                $notification->createAction()
                    ->setLabel('confirm')
                    ->setLink(
                        \OC::$server->getURLGenerator()->linkToRoute(
                            'smartfiletagger.tag.confirm'
                        ),
                        'POST'
                    )
                    ->setPrimary(true)
            )
            ->addAction(
                $notification->createAction()
                    ->setLabel('reject')
                    ->setLink(
                        \OC::$server->getURLGenerator()->linkToRoute(
                            'smartfiletagger.tag.reject'
                        ),
                        'POST'
                    )
                    ->setPrimary(false)
            );

        $this->notificationManager->notify($notification);
    }

    /**
     * Make a POST request to the Python sidecar.
     * Returns decoded JSON array on success, null on failure.
     * Failure must always be handled gracefully by the caller (fail-open).
     */
    private function callSidecar(string $endpoint, array $payload): ?array {
        $url = rtrim($this->sidecarUrl, '/') . $endpoint;

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => json_encode($payload),
            CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 5, // never block Nextcloud for more than 5s
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