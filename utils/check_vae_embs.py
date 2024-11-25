import os
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse

from termcolor import cprint
from diagonal_gaussian import DiagonalGaussianDistribution as DGD

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for Stable Diffusion")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=os.path.expanduser("~/diffusion/Project/dataset/train"),
        help="Path to the training images directory."
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path or identifier for the pretrained VAE model."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.expanduser("~/diffusion/Project/dataset_vae/train"),
        help="Directory where the latent tensors will be saved."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="The size to which input images will be resized."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing images."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of subprocesses for data loading."
    )
    return parser.parse_args()

def load_vae(vae_model_path, device):
    """
    Load the pretrained VAE model.
    """
    vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae")
    vae.to(device)
    vae.eval()
    return vae

def encode(vae, x: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = x.shape

    if vae.use_tiling and (width > vae.tile_sample_min_size or height > vae.tile_sample_min_size):
        return vae._tiled_encode(x)

    enc = vae.encoder(x)
    if vae.quant_conv is not None:
        enc = vae.quant_conv(enc)

    return enc

def create_save_directory(save_path):
    """
    Create the save directory if it does not exist.
    """
    os.makedirs(save_path, exist_ok=True)
    print(f"Latent tensors will be saved to: {save_path}")

def get_image_files(train_data_path):
    """
    Retrieve a list of image file paths from the training data directory.
    """
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = []
    for root, _, files in os.walk(train_data_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(root, file))
    return image_files

def preprocess_image(image_size):
    """
    Define the image preprocessing pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])

def encode_and_save(vae, image_files, preprocess, save_path, device, batch_size=16):
    """
    Encode images using VAE and save the latent tensors.
    """
    num_images = len(image_files)
    print(f"Total images to process: {num_images}")
    
    for i in tqdm(range(0, num_images, batch_size), desc="Encoding images"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        valid_files = []
        
        for file in batch_files:
            try:
                image = Image.open(file).convert("RGB")
                image = preprocess(image)
                batch_images.append(image)
                valid_files.append(file)
            except Exception as e:
                print(f"Error loading image {file}: {e}")
                continue
        
        if not batch_images:
            continue
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            #latents = vae._encode(batch_tensor) #.latent_dist.sample()
            #latents = latents * vae.config.scaling_factor  # Scale the latents
            latents = encode(vae, batch_tensor)
            cprint(latents.size(), color='cyan')

        latents = latents.cpu()

        sample = DGD(latents).sample()
        sample = sample.cpu()

        cprint(sample.size(), color='red')
        cprint(vae.config.scaling_factor, color='yellow')
        #for j, latent in enumerate(latents):
        #    file_path = valid_files[j]
        #    file_name = os.path.splitext(os.path.basename(file_path))[0]
        #    save_file = os.path.join(save_path, f"{file_name}.pt")
        #    torch.save(latent, save_file)

def main():
    args = parse_args()
    
    # 설정된 장치 사용 (GPU가 없으면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # VAE 모델 로드
    vae = load_vae(args.vae_model_path, device)
    
    # 저장 디렉토리 생성
    create_save_directory(args.save_path)
    
    # 이미지 파일 리스트 가져오기
    image_files = get_image_files(args.train_data_path)
    if not image_files:
        print("No images found in the training data path.")
        return
    
    # 이미지 전처리 정의
    preprocess = preprocess_image(args.image_size)
    
    # 이미지 인코딩 및 저장
    encode_and_save(
        vae=vae,
        image_files=image_files[:2],
        preprocess=preprocess,
        save_path=args.save_path,
        device=device,
        batch_size=args.batch_size
    )
    
    print("Latent tensors precomputation completed.")

if __name__ == "__main__":
    main()
