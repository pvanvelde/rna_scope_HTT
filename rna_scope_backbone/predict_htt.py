import numpy as np
import tifffile
import os
import torch
import random
import yaml
import importlib
import h5py
import torch.nn as nn
from skimage.transform import resize
from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model
from skimage.measure import label
import uuid
logger = utils.get_logger('UNet3DPredict')


def show_napari(image):
    import napari
    with napari.gui_qt():
        viewer = napari.view_image(image, title="Detected ROIs")


def resize_image(raw_image, common_shape):
    return resize(raw_image, common_shape, anti_aliasing=True, preserve_range=True).astype(np.uint16)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = config.get('device', None)
    if device == 'cpu':
        logger.warning('CPU mode forced in config, this will likely result in slow training/prediction')
        config['device'] = 'cpu'
    else:
        if torch.cuda.is_available():
            config['device'] = 'cuda'
        else:
            logger.warning('CUDA not available, using CPU')
            config['device'] = 'cpu'

    return config, config_path


def save_prediction_as_tiff(hdf5_path, output_dir, field_name,threshold = 0.95):
    os.makedirs(output_dir, exist_ok=True)
    tiff_path = os.path.join(output_dir, f'{field_name}.tiff')

    with h5py.File(hdf5_path, 'r') as f:
        data = f['predictions'][:]


    return data


def save_prediction_as_tiffv2(hdf5_path, output_dir, field_name, chunk_size=5):
    os.makedirs(output_dir, exist_ok=True)
    tiff_path = os.path.join(output_dir, f'{field_name}.tiff')

    output_data = []

    with h5py.File(hdf5_path, 'r') as f:
        dataset = f['predictions']

        # Process the data in chunks
        for i in range(0, dataset.shape[0], chunk_size):
            chunk = dataset[i:i + chunk_size, :, :]
            # Apply thresholding or other processing here if needed

            output_data.append(chunk)

    # Convert the list of chunks into a single numpy array
    output_data = np.concatenate(output_data, axis=0)

    return output_data
def get_predictor(model, config):

    output_dir = os.path.dirname(config['loaders']['test']['file_paths'][0])

    # if output_dir is not None:
    #     os.makedirs(output_dir, exist_ok=True)
    # else:
    #     output_dir = './predictions_output'
    #     os.makedirs(output_dir, exist_ok=True)
    #     config['output_dir'] = output_dir

    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)
    out_channels = config['model'].get('out_channels')
    return predictor_class(model, output_dir, out_channels, **predictor_config)


# def update_config( model_path, config_path, output_hdf5_path):
#     config = {
#         'model_path': model_path,
#         'model': {
#             'name': 'ResidualUNet3D',
#             'in_channels': 1,
#             'out_channels': 1,
#             'layer_order': 'gcr',
#             'f_maps': [32, 64, 128, 256],
#             'num_groups': 8,
#             'final_sigmoid': True,
#             'is_segmentation': True
#         },
#         'predictor': {
#             'name': 'StandardPredictor'
#             #'name': 'LazyPredictor'
#         },
#         'loaders': {
#             #'batch_size': 1,
#             'batch_size': 2,
#             'raw_internal_path': 'raw',
#             'num_workers': 1,
#             'test': {
#                 'file_paths': [output_hdf5_path],
#                 'slice_builder': {
#                     'name': 'SliceBuilder',
#                     # 'patch_shape': [16, 128, 128],
#                     # # train stride between patches
#                     # 'stride_shape': [16, 128, 128],
#                     'patch_shape': [8, 516,516],
#                     # # train stride between patches
#                     'stride_shape': [8, 514, 514],
#                     #'halo_shape': [32, 32, 32] , # Halo to provide context around borders
#                     'halo_shape': [2, 2, 2],  # Halo to provide context around borders
#
#                 },
#
#                 'transformer': {
#                     'raw': [
#                         {'name': 'Standardize'},
#                         {'name': 'ToTensor', 'expand_dims': True}
#                     ]
#                 }
#             }
#         },
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#         'output_dir': 'output_hdf5_path'
#     }
#     with open(config_path, 'w') as f:
#         yaml.dump(config, f)
#     return config, config_path
#



def function_htt(raw_image,config_path):

    common_shape = raw_image.shape  # Assuming the image is already of the correct shape
    raw_image_resized = resize_image(raw_image, common_shape)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    # check if bigger than image
    config['loaders']['test']['slice_builder']['patch_shape'][0] = min(common_shape[0],  config['loaders']['test']['slice_builder']['patch_shape'][0])
    config['loaders']['test']['slice_builder']['patch_shape'][1] = min(common_shape[1],  config['loaders']['test']['slice_builder']['patch_shape'][1])
    config['loaders']['test']['slice_builder']['patch_shape'][2] = min(common_shape[2],  config['loaders']['test']['slice_builder']['patch_shape'][2])

    # Original file path
    original_file_path = config['loaders']['test']['file_paths'][0]
    dir_path = os.path.dirname(original_file_path)
    unique_id = uuid.uuid4().hex
    # New file name with unique ID
    base_name = "predict"
    new_file_name = f"{base_name}_{unique_id}.h5"
    file_path = os.path.join(dir_path, new_file_name)
    config['loaders']['test']['file_paths'][0] = file_path
    os.makedirs(dir_path, exist_ok=True)

    # Write HDF5 file
    try:
        with h5py.File(file_path, 'w') as hdf5_file:
            raw_dataset = hdf5_file.create_dataset(
                'raw',
                data=raw_image_resized,
                dtype=np.uint16,
            )
        print(f"HDF5 file successfully written to {file_path}")
    except Exception as e:
        print(f"Failed to write HDF5 file: {e}")




    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True

    model = get_model(config['model'])
    #test = model((torch.tensor(raw_image_resized).to('cuda').type(torch.float))[None,...])
    logger.info(f'Loading model from {config["model_path"]}...')
    utils.load_checkpoint(config['model_path'], model)

    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    if torch.cuda.is_available() and not config['device'] == 'cpu':
        model = model.cuda()

    predictor = get_predictor(model, config)

    for test_loader in get_test_loaders(config):
        predictor(test_loader)

    # Assuming the HDF5 file name is generated as 'output_predictions.h5' in the output directory
    predictions_hdf5_path = dir_path+ '/predict_'+unique_id+'_predictions.h5'

    print('save as tiff file...')
    segmented = save_prediction_as_tiffv2(predictions_hdf5_path,dir_path, 'predictions')
    print('done saving as tiff file...')
    #copy_config(config, config_path)
    #show_napari(segmented)
    label_image, num_features = label(segmented > 0.5, return_num=True)

    # Delete the files
    for path in [predictions_hdf5_path, file_path]:
        try:
            os.remove(path)
            print(f"Deleted: {path}")
        except OSError as e:
            print(f"Error deleting {path}: {e}")

    return label_image

if __name__ == '__main__':
    image_file = '/media/grunwaldlab/Extreme SSD/Dropbox (UMass Medical School)/Huntingtin_RNAscope/full_lenght_HTT/training_data_stiched/grunwald test_2024-07-01_17.58.50_channel2_stitched.tif'
    raw_image = tifffile.imread(image_file)
    model_path = '/home/grunwaldlab/Development/rna_scope/full_length_hht_neural_network_training/CHECKPOINT_DIR_HTT/last_checkpoint.pytorch'
    segmented = function_htt(raw_image, model_path)