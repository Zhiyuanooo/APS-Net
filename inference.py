import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from models import APSnet18
from train import default_loader


class fenlei:
    def __init__(self):
        pass
    def infer_image(self, model, image_path):
        img_mean = 0.3309
        img_std = 0.1924
        # Load the image and apply transformations
        img = default_loader(image_path, True)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean], std=[img_std])
        ])
        img_tensor = transform(img).unsqueeze(0)

        # Move the image tensor to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)

        # Move the model to the same device as the input tensor
        model.to(device)

        # Perform inference on the image
        with torch.no_grad():
            model.eval()
            out_class, out_seg = model(torch.cat([img_tensor, img_tensor, img_tensor], 1))
            predicted_label = out_class.argmax(dim=1).cpu().item()
            segmentation_mask = F.sigmoid(out_seg).cpu().squeeze().numpy()

        return predicted_label, segmentation_mask

    def run(self, input_dir, output_dir):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        save_root = './temp'

        # Load the trained model
        model = APSnet18()
        model_path = os.path.join(save_root, '13_best_models.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_paths = []
        text_paths = []

        # Perform inference on each image in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(input_dir, filename)
                predicted_label, segmentation_mask = self.infer_image(model, image_path)

                # Save the results
                output_label_file = os.path.join(output_dir, f'{filename}_label.txt')
                with open(output_label_file, 'w') as f:
                    f.write(str(predicted_label))
                text_paths.append(os.path.abspath(output_label_file))

                output_seg_file = os.path.join(output_dir, f'{filename}_seg.png')
                seg_image = Image.fromarray((segmentation_mask * 255).astype(np.uint8))
                seg_image.save(output_seg_file)
                image_paths.append(os.path.abspath(output_seg_file))

        return image_paths, text_paths
