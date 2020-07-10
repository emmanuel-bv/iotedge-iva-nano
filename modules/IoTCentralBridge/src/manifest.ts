import { ComposeManifest } from 'spryly';
import { resolve as pathResolve } from 'path';

const DefaultPort = 9014;
const PORT = process.env.PORT || process.env.port || process.env.PORT0 || process.env.port0 || DefaultPort;

export function manifest(config?: any): ComposeManifest {
    return {
        server: {
            port: PORT,
            app: {
                rootDirectory: pathResolve(__dirname, '..'),
                storageRootDirectory: process.env.DATAMISC_ROOT || '/data/misc/storage',
                slogan: 'NVIDIA Jetson Nano local service'
            }
        },
        services: [
            './services'
        ],
        plugins: [
            // ...[
            //     {
            //         plugin: './plugins'
            //     }
            // ],
            ...[
                {
                    plugin: './apis'
                }
            ]
        ]
    };
}
