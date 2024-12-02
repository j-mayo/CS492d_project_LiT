import os
from PIL import Image
from torchvision import transforms

src_dir = 'src/dir'
tgt_dir = 'tgt/dir'

image_files = [f for f in os.listdir(src_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor() 
])

for image_file in image_files:
    src_path = os.path.join(src_dir, image_file)
    image = Image.open(src_path)
    
    transformed_image = transform(image)

    tgt_path = os.path.join(tgt_dir, image_file)
    
    transformed_image_pil = transforms.ToPILImage()(transformed_image)
    transformed_image_pil.save(tgt_path)
