import { service, inject } from 'spryly';
import { Server } from '@hapi/hapi';
import { LoggingService } from './logging';
import { DeviceService } from './device';
import { ModuleInfoFieldIds, ModuleState, IoTCentralService } from './iotCentral';
import { HealthState } from './health';
import * as _get from 'lodash.get';
import * as _random from 'lodash.random';
import { bind } from '../utils';

@service('module')
export class ModuleService {
    @inject('$server')
    private server: Server;

    @inject('logger')
    private logger: LoggingService;

    @inject('device')
    private device: DeviceService;

    @inject('iotCentral')
    private iotCentral: IoTCentralService;

    public async init(): Promise<void> {
        this.logger.log(['ModuleService', 'info'], 'initialize');

        this.server.method({ name: 'module.startService', method: this.startService });
    }

    // @ts-ignore (testparam)
    public async route1(testparam: any): Promise<void> {
        return;
    }

    @bind
    public async startService(): Promise<void> {
        this.logger.log(['ModuleService', 'info'], `Starting service...`);

        await this.iotCentral.sendMeasurement({
            [ModuleInfoFieldIds.State.ModuleState]: ModuleState.Active,
            [ModuleInfoFieldIds.Event.VideoStreamProcessingStarted]: 'NVIDIA DeepStream'
        });
    }

    public async getHealth(): Promise<number> {
        const iotCentralHealth = await this.iotCentral.getHealth();

        if (iotCentralHealth < HealthState.Good) {

            this.logger.log(['ModuleService', 'info'], `Health check iot:${iotCentralHealth}`);

            await this.device.restartDevice(10, 'ModuleService:getHealth');

            return HealthState.Critical;
        }

        await this.iotCentral.sendMeasurement({
            [ModuleInfoFieldIds.Telemetry.CameraSystemHeartbeat]: iotCentralHealth
        });

        return HealthState.Good;
    }
}
