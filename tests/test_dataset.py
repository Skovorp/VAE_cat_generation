import unittest
from src.dataset import CatDataset
import yaml
import os
import torch


class TestCatDataset(unittest.TestCase):

    def setUp(self):
        # Load configuration
        config_path = '/home/ubuntu/image_generation/configs/one_batch_config.yaml'
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize datasets
        self.train_set = CatDataset(**self.cfg['dataset']['general'], **self.cfg['dataset']['train'])
        self.val_set = CatDataset(**self.cfg['dataset']['general'], **self.cfg['dataset']['val'])

    def test_image_size(self):
        # Check one image size from train and val sets
        train_image = self.train_set[0]
        val_image = self.val_set[0]

        self.assertEqual(train_image.shape, (3, 64, 64))
        self.assertEqual(val_image.shape, (3, 64, 64))
        
    def test_image_type(self):
        # Check that the image type is float
        train_image = self.train_set[0]
        val_image = self.val_set[0]
        self.assertTrue(train_image.dtype == torch.float32, "Image type is not float32")
        self.assertTrue(val_image.dtype == torch.float32, "Image type is not float32")


    def test_number_of_images(self):
        # Get the expected sizes
        total_images = len(os.listdir(self.cfg['dataset']['general']['dataset_path']))
        train_size = int(total_images * (1 - self.cfg['dataset']['general']['val_part']))
        val_size = total_images - train_size
        
        train_size = min(self.cfg['dataset']['train']['limit'] or int(1e8), train_size)
        val_size = min(self.cfg['dataset']['val']['limit'] or int(1e8), val_size)

        # Check actual sizes
        self.assertEqual(len(self.train_set), train_size)
        self.assertEqual(len(self.val_set), val_size)

    def test_image_loading(self):
        # Simply loading an image to see if it works without errors
        try:
            _ = self.train_set[0]
            _ = self.val_set[0]
        except Exception as e:
            self.fail(f"Failed to load an image: {e}")

if __name__ == "__main__":
    unittest.main()
