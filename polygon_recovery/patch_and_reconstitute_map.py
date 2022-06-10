import numpy as np
import cv2, os, glob

from dh_segment_torch.config import Params
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.data.annotation import AnnotationWriter
from dh_segment_torch.training import Trainer
from dh_segment_torch.inference import PredictProcess
from dh_segment_torch.post_processing import PostProcessingPipeline
from PIL import Image

classes = [[0, 0, 0], [255, 255, 255]]

def reconstituteMap(image: np.ndarray, prediction_path: str, classes: list, rows: int, cols: int,
                    patch_size: int = 1000, patch_overlap: int = 200, output_pred_size: int = 848):
    ''' Reconstitution of the map based on the prediction on image patches.
    Input(s):
        image: original full-size map
        prediction_path: path to the folder where the predicted patches are found
        classes: color classes used to train the model
        rows: number of rows of patches
        cols: number of columns of patches
        patch_size: side length of each patch
        patch_overlap: number of pixels on which adjacent patches overlap
        output_pred_size: size of the prediction map (CNN output)
    Output(s):
        prediction_map: image of the full_sized predicted segmentation of the map
    '''
    
    core_size = patch_size-patch_overlap
    reconstitution = np.zeros((patch_size+(rows*core_size), patch_size+(cols*core_size), len(classes)))
    
    for row in range(rows):
        for col in range(cols):
            # print("row: " + str(row) + " col: " + str(col))
            probs = np.load(os.path.join(prediction_path, str(row) + '_' + str(col) + '.npy'))
            probs = np.swapaxes(probs, 0, 1)
            probs = np.swapaxes(probs, 1, 2)

            pre_row = pre_col = patch_overlap//4
            post_row = post_col = patch_size - (patch_overlap//4)

            if row == 0:
                pre_row = 0
            elif row == rows-1:
                post_row = patch_size
            if col == 0:
                pre_col = 0
            elif col == cols-1:
                post_col = patch_size

            insert = cv2.resize(probs, dsize=(patch_size, patch_size))

            reconstitution[pre_row+(row*core_size):(row*core_size)+post_row, 
                           pre_col+(col*core_size):(col*core_size)+post_col] = insert[
                pre_row:post_row, pre_col:post_col]
            
    prediction_map = np.zeros((reconstitution.shape[0], reconstitution.shape[1], 3))
    
    labels = np.argmax(reconstitution, axis = 2)
    for i in range(len(classes)):
        prediction_map[labels == i] = classes[i]
            
    prediction_map = prediction_map.astype(np.uint8, copy=False)
    prediction_map = cv2.cvtColor(prediction_map, cv2.COLOR_BGR2RGB)
    prediction_map = prediction_map[:image.shape[0], :image.shape[1]]
    
    prediction_map = cv2.resize(prediction_map, (int(prediction_map.shape[1]//(patch_size/output_pred_size)), 
                                                 int(prediction_map.shape[0]//(patch_size/output_pred_size))))
    
    return prediction_map.astype('uint8')


def makeImagePatches(image: np.ndarray, patches_path: str = '', export: bool = False, patch_size: int = 1000,
                     patch_overlap: int = 200):
    ''' Reconstitution of the map based on the prediction on image patches.
    Input(s):
        image: original full-size map
        patches_path: path to the folder where the image patches will be saved
        patch_size: side length of each patch
        patch_overlap: number of pixels on which adjacent patches overlap
        export: if True, the patches are saved to patches_path
    Output(s):
        rows: number of rows of patches
        cols: number of columns of patches
        patches: if they are not exported, the image patches are returned
    '''
    
    core_size = patch_size-patch_overlap
    rows = 1 + ((image.shape[0]-patch_overlap-1)//core_size)
    cols = 1 + ((image.shape[1]-patch_overlap-1)//core_size)
    
    patches = []
    for row in range(rows):
        for col in range(cols):
            patch = image[0+row*core_size:patch_size+row*core_size, 0+col*core_size:patch_size+col*core_size]
            if patch.shape[:2] != (patch_size, patch_size):
                background = np.zeros((patch_size, patch_size, 3))
                background[0:patch.shape[0], 0:patch.shape[1]] = patch
                patch = background
            
            if export:
                cv2.imwrite(os.path.join(patches_path, str(row) + '_' + str(col) + '.png'), patch)
            else:
                patches.append(patch.astype('uint8'))
    if export:        
        return rows, cols
    else:
        return rows, cols, patches


patch_size = 1024
patch_overlap = 512


if __name__ == '__main__':
    image_path = "../training/additional/images/"
    patches_path = "additional/patches/"
    patches_prediction_path = "additional/patches_prediction/"
    save_path = "additional/prediction_results/"
    img_lists = os.listdir(image_path)

    if not os.path.exists(patches_path):
        os.makedirs(patches_path)

    if not os.path.exists(patches_prediction_path):
        os.makedirs(patches_prediction_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_name in img_lists:
        image = cv2.imread(image_path + img_name)
        np_image = np.asarray(image)
        isPatched = True
        if not os.path.exists(patches_path + img_name[:-4]):
            os.mkdir(patches_path + img_name[:-4])
            isPatched = False
        if not os.path.exists(patches_prediction_path + img_name[:-4]):
            os.mkdir(patches_prediction_path + img_name[:-4])
        
        
        rows, cols = makeImagePatches(np_image, patches_path + img_name[:-4], 
                True, patch_size, patch_overlap)

        print("finished patching {} ! rows: {}, cols: {}".format(img_name, rows, cols))
        
        dataset_params = {
            "type": "folder",
            "folder": patches_path + img_name[:-4],
            "pre_processing": {"transforms": []}
        }

        model_params = {
            "model": {
                    "encoder": "resnet101",
                    "decoder": {"decoder_channels": [512, 256, 128, 64, 32], "max_channels": 512}
                },
                "num_classes": 2,
                "model_state_dict": sorted(glob.glob("../historical_model/best_model_checkpoint_miou=*.pth"))[-1],
                "device": "cuda:1"
        }

        process_params = Params({
            'data': dataset_params,
            'model': model_params,
            'batch_size': 1,
            'num_workers': 2,
            'add_path': True
        })

        predict_annots = PredictProcess.from_params(process_params)
        predict_annots.process_to_probas_files(os.path.join(patches_prediction_path, img_name[:-4]))

        print("finished predicting {} patches !".format(img_name))

        prediction_map = reconstituteMap(np_image, os.path.join(patches_prediction_path, img_name[:-4]), 
                classes, rows, cols, patch_size, patch_overlap, patch_size)
        # prediction_map = Image.fromarray(prediction_map)
        print("shape: " + str(prediction_map.shape[0]) + " " + str(prediction_map.shape[1]) + " " + str(prediction_map.shape[2]))
        cv2.imwrite(os.path.join(save_path, img_name[:-4] + ".png"), prediction_map)
        print("finished reconstituting prediction map of {} !".format(img_name))

