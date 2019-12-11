import { inject, RoutePlugin, route } from 'spryly';
import { Request, ResponseToolkit } from '@hapi/hapi';
import { ModuleService } from '../services/module';
import Boom from '@hapi/boom';
import * as _get from 'lodash.get';

export class ModuleRoutes extends RoutePlugin {
    @inject('module')
    private module: ModuleService;

    @route({
        method: 'POST',
        path: '/api/v1/iotcentral/route1',
        options: {
            tags: ['module'],
            description: 'route1 example'
        }
    })
    // @ts-ignore (request)
    public async postRoute1(request: Request, h: ResponseToolkit) {
        try {
            const testparam = _get(request, 'payload.testparam');

            const result = await this.module.route1(testparam);

            return h.response(result).code(201);
        }
        catch (ex) {
            throw Boom.badRequest(ex.message);
        }
    }
}
