from src.Model.AutoEncoder.AutoEncoderCNN import AutoEncoderCNN
from src.Process import process, make_grayscale
from src.Model.UNet.UNetCNN import UNetCNN
from src.Config.Args import args

def make_model(name: str, image_type: str):
    """
        Input:
            name       - The name of the model: 'AutoEncoder' or 'UNet'
            image_type - 'rgb' or 'grayscale'
        Output:
            (model_type, path_to_weights, transform_function)
    """
    # Check if model type is valid
    if name not in ['AutoEncoder', 'UNet']:
        raise ValueError('Invalid Model Type')
    
    # Check if image type is valid
    if image_type not in ['rgb', 'grayscale']:
        raise ValueError('Invalid Image Type')

    # Assign model type
    model_type = UNetCNN if name == 'UNet' else AutoEncoderCNN

    # Assign path to model weights
    path_to_weights = f'{args.temp_dir}/model/{image_type}/{name}.pt'

    # Assign transformer function
    transform = (lambda x : process(x)) if image_type == 'rgb' else (lambda x : process(make_grayscale(x)))

    # Return tuple containing specific setup
    return model_type, path_to_weights, transform