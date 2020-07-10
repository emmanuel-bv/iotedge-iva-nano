import { service, inject } from 'spryly';
import { Server } from '@hapi/hapi';
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
    @inject('$server')
    private server: Server;

    @inject('module')
    private module: ModuleService;

    @inject('device')
    private device: DeviceService;

    private heathCheckStartTime = Date.now();
    private failingStreak = 1;

    public async init() {
        this.server.log(['HealthService', 'info'], 'initialize');

        if (_get(process.env, 'LOCAL_DEBUG') === '1') {
            setInterval(async () => {
                const cameraHealth = await this.checkHealthState();

                if (cameraHealth < HealthState.Good) {
                    if ((Date.now() - this.heathCheckStartTime) > (1000 * healthCheckStartPeriod) && ++this.failingStreak >= healthCheckRetries) {
                        await(this.server.methods.device as any).restartDevice('HealthService:checkHealthState');
                    }
                }
                else {
                    this.heathCheckStartTime = Date.now();
                    this.failingStreak = 0;
                }
            }, (1000 * healthCheckInterval));
        }
    }

    @bind
    public async checkHealthState(): Promise<number> {
        this.server.log(['HealthService', 'info'], 'Health check interval');

        const moduleHealth = await this.module.getHealth();

        if (moduleHealth < HealthState.Good) {
            this.server.log(['HealthService', 'warning'], `Health check watch: module:${moduleHealth}`);

            if ((Date.now() - this.heathCheckStartTime) > (1000 * healthCheckStartPeriod) && ++this.failingStreak >= healthCheckRetries) {
                await this.device.restartDevice(10, 'HealthService:checkHealthState');
            }
        }

        return moduleHealth;
    }
}
