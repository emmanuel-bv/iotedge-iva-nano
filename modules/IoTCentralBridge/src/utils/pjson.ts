import * as fse from 'fs-extra';
import { resolve } from 'path';

export function pjson(): any {
    let result = {};

    try {
        const packagePath = resolve(__dirname, '..', '..', 'package.json');
        const contents = fse.readFileSync(packagePath);
        if (contents) {
            result = JSON.parse(contents);
        }
    }
    catch (ex) {
        // eat exception
    }

    return result;
}
