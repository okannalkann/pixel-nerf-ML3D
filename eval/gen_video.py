import sys
import os

# Bu satır custom_models.py dosyasının bulunduğu dizini belirtir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from custom_models import CustomEncoder, CustomDecoder  # Import custom models
from scipy.interpolate import CubicSpline
import tqdm

def extra_args(parser):
    parser.add_argument("--subset", "-S", "-V", type=int, default=0, help="Subset in data to use")
    parser.add_argument("--split", type=str, default="train", help="Split of data to use train | val | test")
    parser.add_argument("--source", "-P", type=str, default="64", help="Source view(s) in image, in increasing order. -1 to do random")
    parser.add_argument("--num_views", type=int, default=40, help="Number of video frames (rotated views)")
    parser.add_argument("--elevation", type=float, default=-10.0, help="Elevation angle (negative is above)")
    parser.add_argument("--scale", type=float, default=1.0, help="Video scale relative to input size")
    parser.add_argument("--radius", type=float, default=0.0, help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)")
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    # Check if --visual_path and --name are already defined
    if not any(arg.dest == "visual_path" for arg in parser._actions):
        parser.add_argument("--visual_path", type=str, default="visuals", help="Path to save the video")
    if not any(arg.dest == "name" for arg in parser._actions):
        parser.add_argument("--name", type=str, default="output", help="Name of the output video")
    return parser

args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

# Debug bilgisi ekleyelim
print(f"Loading dataset from {args.datadir} with format {args.dataset_format} and split {args.split}")

dset = get_split_dataset(args.dataset_format, args.datadir, want_split=args.split, training=False)

# Veri kümesindeki toplam örnek sayısını kontrol etme
total_samples = len(dset)
print(f"Total samples in the dataset: {total_samples}")

if total_samples == 0:
    raise ValueError("The dataset contains no samples. Please check the dataset path and format.")

data = dset[args.subset]
data_path = data["path"]
print("Data instance loaded:", data_path)

images = data["images"]  # (NV, 3, 300, 400)
poses = data["poses"]  # (NV, 4, 4)
focal = data["focal"]
# z_near ve z_far değerlerini doğrudan ayarlıyoruz
z_near = 1.2
z_far = 4.0

# focal tensorunu listeye çeviriyoruz
if isinstance(focal, torch.Tensor):
    focal = focal.tolist()

# Debug: Giriş görüntülerinin boyutlarını kontrol edelim
print(f"Input image size: {images.shape}")
print(f"z_near: {z_near}, z_far: {z_far}")
print(f"focal: {focal}")

# Initialize custom encoder and decoder
encoder = CustomEncoder(input_height=images.shape[2], input_width=images.shape[3]).to(device)
decoder = CustomDecoder(output_size=(images.shape[2], images.shape[3])).to(device)

# Encode images
encoded_images = encoder(images.to(device))

# Decode images
decoded_images = decoder(encoded_images)

# Initialize renderer and model
renderer = NeRFRenderer.from_conf(conf)
renderer = renderer.to(device)  # Ensure the renderer is moved to the correct device

# Assume `model` is defined and loaded properly
model = make_model(conf["model"]).to(device)

# Bind the renderer to the model
wrapped_renderer = renderer.bind_parallel(model, gpus=[args.gpu_id], simple_output=True)

# Render the novel views
all_rgb_frames = []
for pose in tqdm.tqdm(poses, desc="Rendering frames"):
    rays = util.gen_rays(pose, focal, z_near, z_far, images.shape[2], images.shape[3])  # Generate rays for the given pose
    rgb, _ = wrapped_renderer(rays.to(device))  # Perform rendering
    all_rgb_frames.append(rgb.cpu().numpy())

# Save the video
output_path = os.path.join(args.visual_path, f"{args.name}.mp4")
imageio.mimwrite(output_path, all_rgb_frames, fps=args.fps)
print(f"Video saved to {output_path}")
