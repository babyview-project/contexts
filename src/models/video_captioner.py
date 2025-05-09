from PIL import Image
import requests
class VideoCaptioner():
    def __init__(self, device):
        self.device = device

    def open_image(self, image_path):
        """
        Open an image from either a URL or a local file path
        
        Args:
            image_path (str): URL or local file path to the image
            
        Returns:
            PIL.Image: The opened image
        """
        # Check if the path is a URL (starts with http:// or https://)
        if image_path.startswith(('http://', 'https://')):
            # Open image from URL
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            # Open image from local file path
            image = Image.open(image_path)
        return image
    
    def open_images(self, image_paths):
        pils = []
        return [self.open_image(image_path) for image_path in image_paths]

    def caption_image(input_image, caption):
        return 