import { service, inject } from 'spryly';
import { LoggingService } from './logging';
import { ModuleService } from './module';
import { DeviceService } from './device';
import * as _get from 'lodash.get';
import { bind } from '../utils';

export const healthCheckInterval = 15;
// const healthCheckTimeout = 30;
const healthCheckStartPeriod = 60;
const healthCheckRetries = 3;

export const HealthState = {
    Good: 1,
    Warning: 0,
    Critical: 0
};

@service('health')
export class HealthService {
    @inject('logger')
    private logger: LoggingService;

    @inject('module')
    private module: ModuleService;

    @inject('device')
    private device: DeviceService;

    private heathCheckStartTime = Date.now();
    private failingStreak = 1;

    public async init() {
        this.logger.log(['HealthService', 'info'], 'initialize');
    }

    @bind
    public async checkHealthState(): Promise<number> {
        const moduleHealth = await this.module.getHealth();

        if (moduleHealth < HealthState.Good) {
            this.logger.log(['HealthService', 'warning'], `Health check watch: module:${moduleHealth}`);

            if ((Date.now() - this.heathCheckStartTime) > (1000 * healthCheckStartPeriod) && ++this.failingStreak >= healthCheckRetries) {
                await this.device.restartDevice(10, 'HealthService:checkHealthState');
            }
        }

        return moduleHealth;
    }
}
