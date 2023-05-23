import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
# VCR_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'vcr1images')
VCR_IMAGES_DIR = os.path.join("/remote-home/1810864/VCR_dataset/VCR/", 'vcr1images')
VCR_ANNOTS_DIR = os.path.join("/remote-home/1810864/VCR_dataset/VCR/", '')
DATALOADER_DIR = '/remote-home/1810864/projects/graph_transformer_with_attribution/'
BERT_DIR = '/remote-home/1810864/VCR_dataset/'


# VCR_ANNOTS_DIR = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")
