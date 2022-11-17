import torch.nn as nn

from .coatnet import CoAtNet


model_dict = {
    0: {
        'num_blocks': [2, 2, 3, 5, 2],
        'channels': [64, 96, 192, 384, 768]
    },
    1: {
        'num_blocks': [2, 2, 6, 14, 2],
        'channels': [64, 96, 192, 384, 768]
    },
    2: {
        'num_blocks': [2, 2, 6, 14, 2],
        'channels': [128, 128, 256, 512, 1026]
    },
    3: {
        'num_blocks': [2, 2, 6, 14, 2],
        'channels': [192, 192, 384, 768, 1536]
    },
    4: {
        'num_blocks': [2, 2, 12, 28, 2],
        'channels': [192, 192, 384, 768, 1536]
    }
}


def load_model(model_ver: int, in_channels: int, image_size: int, num_classes: int,
               block_types: list = ['C', 'C', 'T', 'T']):
               
    model_attributes = model_dict.get(model_ver)
    
    if not model_attributes:
        print('No CoAtNet with this version exists.\nPlease check the version you entered.')
        return
    
    num_blocks = model_attributes.get('num_blocks')
    channels = model_attributes.get('channels')

    model =  CoAtNet((image_size, image_size), in_channels, num_blocks, channels,
                    num_classes, block_types)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
