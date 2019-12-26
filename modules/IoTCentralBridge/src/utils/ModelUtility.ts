import * as download from 'download';
import * as fs from 'fs';
import * as unzipper from 'unzipper';

//todo: this definitino should probably be hosted somewhere else
let modelDirectory = "/data/misc/storage/";

export async function downloadModel(modelURI:string) {
        try {
            fs.unlinkSync(modelDirectory + "model.zip");
            download(modelURI).then(data => {
                fs.writeFileSync(modelDirectory + "model.zip", data);
            });
        }
        catch (error) {
            // tslint:disable-next-line:no-console
            console.log(`Could not download AI model: ${error.message}`);
        }
    }

export async function unzipModel() {
        try {
        fs.unlinkSync(modelDirectory + "model.onnx");
        fs.unlinkSync(modelDirectory + "labels.txt");
        fs.createReadStream(modelDirectory + "model.zip").pipe(unzipper.Extract({ path: modelDirectory }));
        }   
        catch (error) {
            // tslint:disable-next-line:no-console
            console.log(`Could not unzip AI model: ${error.message}`);
        }
    }