import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import warnings
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import matplotlib.patches as patches

warnings.filterwarnings('ignore')
# %matplotlib inline

torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


class DTUDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotation_file (str): Path to the annotation file in COCO format (e.g., from GitHub).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            return json.load(f)

    def slice_image(self, image_path, tile_size=1024):
        try:
            row, col = map(int, image_path.split('/')[-1].split('.')[0].split('_')[-2:])
            new_path = image_path.split('.')[0][:-4] + '.JPG'
            image = Image.open(new_path).convert('RGB')
            width, height = image.size

            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size
            
            if right > width or lower > height:
                raise ValueError(f"Slice ({row}, {col}) exceeds image bounds: {width}x{height}")
            
            return image.crop((left, upper, right, lower))
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        slc = self.slice_image(img_path)

        if self.transform:
            image = self.transform(slc)

        ann_ids = [ann for ann in self.annotations['annotations'] if ann['image_id'] == img_info['id']]
        boxes = [ann['bbox'] for ann in ann_ids]
        labels = [ann['category_id'] for ann in ann_ids]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        masks = []
        for box in boxes:
            x_min, y_min, width, height = box
            x_max = x_min + width
            y_max = y_min + height
            masks.append(torch.zeros((1024, 1024), dtype=torch.uint8))
            masks[-1][int(y_min):int(y_max), int(x_min):int(x_max)] = 1

        masks = torch.stack(masks) if masks else torch.zeros((0, *image.size), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }

        return image, target
    
annotations_dir = "DTU-annotations-main/"
dataset_zip = "DTU - Drone inspection images of wind turbine/"

if not os.path.exists(annotations_dir):
    os.system('wget https://github.com/imadgohar/DTU-annotations/archive/refs/heads/main.zip')
    os.system('unzip -o main.zip')
    os.system('rm main.zip')

if not os.path.exists(dataset_zip):
    os.system('wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hd96prn3nc-2.zip')
    os.system('unzip -o hd96prn3nc-2.zip')
    os.system('rm hd96prn3nc-2.zip')

os.system('mv "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/Nordtank 2017"/*.JPG "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/"')
os.system('mv "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/Nordtank 2018"/*.JPG "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/"')

root_dir = "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/"
annotation_file = "DTU-annotations-main/re-annotation/D3/train.json"

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

train_dataset = DTUDataset(root_dir=root_dir, annotation_file=annotation_file, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

data_iter = iter(train_dataloader)
images, targets = next(data_iter)

image = images[0]
target = targets[0]
array = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

# fig, ax = plt.subplots(1)
# ax.imshow(array)

# for box in target['boxes']:
#     x_min, y_min, width, height = box
#     rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

# # Plot the image
# plt.imshow(array)
# plt.title(f"Labels: {target['labels']}, Bboxes: {target['boxes']}")
# plt.show()

def train_model(model, optimizer, dataloader, num_epochs=20, device='cuda'):
    """
    Train the Mask R-CNN model with a specified dataloader and optimizer.

    Args:
        model (nn.Module): The Mask R-CNN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        num_epochs (int): Number of training epochs.
        device (str): Device to use for training (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for images, targets in dataloader:
            for target in targets:
                array = target['boxes'].numpy()
                x_min, y_min, width, height = array[0]
                x_max = x_min + width
                y_max = y_min + height
                target['boxes'] = torch.tensor([[
                    x_min, y_min, x_max, y_max
                ]], dtype=torch.float32)

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            try:
                loss_dict = model(images, targets)

                loss = sum(loss for loss in loss_dict.values())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                loss_logs = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                print(f"Batch Loss: {loss.item():.4f} | {loss_logs}")

            except AssertionError as e:
                print(f"Skipping batch due to error: {e}")
                continue

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {epoch_loss:.4f}")

def evaluation(model, dataloader, device='cuda'):
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            for i, prediction in enumerate(predictions):
                print(f"Image {i}:")
                print(f"  Boxes: {prediction['boxes']}")
                print(f"  Labels: {prediction['labels']}")
                print(f"  Scores: {prediction['scores']}")

backbone = resnet_fpn_backbone('resnet50', pretrained=True)

anchor_generator = AnchorGenerator(
sizes=((32,), (64,), (128,), (256,), (512,)),
aspect_ratios=((0.5, 1.0, 2.0),) * 5)

num_classes = 5

model = MaskRCNN(
    backbone=backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

train_model(model, optimizer, train_dataloader, num_epochs=20, device=device)

annotation_file = "DTU-annotations-main/re-annotation/D3/test.json"

test_dataset = DTUDataset(root_dir=root_dir, annotation_file=annotation_file, transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

evaluation(model, test_dataloader, device=device)