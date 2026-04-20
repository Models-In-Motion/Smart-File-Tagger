<?php
declare(strict_types=1);

namespace OCA\SmartFileTagger\AppInfo;

use OCP\AppFramework\App;
use OCP\AppFramework\Bootstrap\IBootContext;
use OCP\AppFramework\Bootstrap\IBootstrap;
use OCP\AppFramework\Bootstrap\IRegistrationContext;
use OCP\EventDispatcher\IEventDispatcher;
use OCP\Files\Events\Node\NodeCreatedEvent;
use OCP\Util;

class Application extends App implements IBootstrap {
    public const APP_ID = 'smartfiletagger';

    public function __construct() {
        parent::__construct(self::APP_ID);
    }

    public function register(IRegistrationContext $context): void {}

    public function boot(IBootContext $context): void {
        Util::addScript(self::APP_ID, 'smart-tagger');
        Util::addStyle(self::APP_ID, 'smart-tagger');
    }
}
