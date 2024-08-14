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
<<<<<<< HEAD
from model.encoder_decoder import EncoderDecoderModel
=======
from custom_models import CustomEncoder, CustomDecoder  # Import custom models
>>>>>>> c64aab0d01e0521b9e1dc910c19773b8f70f0c36
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
<<<<<<< HEAD
    parser.add_argument(
        "--encoder_mode",
        type=str,
        choices=["active", "passive"],
        default="passive",
        help="Set encoder-decoder mode to active or passive",
    )
=======
    # Check if --visual_path and --name are already defined
    if not any(arg.dest == "visual_path" for arg in parser._actions):
        parser.add_argument("--visual_path", type=str, default="visuals", help="Path to save the video")
    if not any(arg.dest == "name" for arg in parser._actions):
        parser.add_argument("--name", type=str, default="output", help="Name of the output video")
>>>>>>> c64aab0d01e0521b9e1dc910c19773b8f70f0c36
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

<<<<<<< HEAD
# Load the encoder-decoder model based on the mode
encoder_decoder = EncoderDecoderModel().to(device)
encoder_decoder.load_state_dict(torch.load(os.path.abspath('eval/encoder_decoder.pth'), map_location=device))

if args.encoder_mode == "active":
    encoder_decoder.train()  # Set to active mode
else:
    encoder_decoder.eval()  # Set to passive mode

net = make_model(conf["model"]).to(device=device)
net.load_weights(args)
=======
# Encode images
encoded_images = encoder(images.to(device))
>>>>>>> c64aab0d01e0521b9e1dc910c19773b8f70f0c36

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

<<<<<<< HEAD
if dtu_format:
    print("Using DTU camera trajectory")
    # Use hard-coded pose interpolation from IDR for DTU

    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    pose_quat = torch.tensor(
        [
            [0.9698, 0.2121, 0.1203, -0.0039],
            [0.7020, 0.1578, 0.4525, 0.5268],
            [0.6766, 0.3176, 0.5179, 0.4161],
            [0.9085, 0.4020, 0.1139, -0.0025],
            [0.9698, 0.2121, 0.1203, -0.0039],
        ]
    )
    n_inter = args.num_views // 5
    args.num_views = n_inter * 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    q_new = torch.from_numpy(q_new).float()

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        new_q = new_q.unsqueeze(0)
        R = util.quat_to_rot(new_q)
        t = R[:, :, 2] * scale
        new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        new_pose[:, :3, :3] = R
        new_pose[:, :3, 3] = t
        render_poses.append(new_pose)
    render_poses = torch.cat(render_poses, dim=0)
else:
    print("Using default (360 loop) camera trajectory")
    if args.radius == 0.0:
        radius = (z_near + z_far) * 0.5
        print("> Using default camera radius", radius)
    else:
        radius = args.radius

    # Use 360 pose sequence from NeRF
    render_poses = torch.stack(
        [
            util.pose_spherical(angle, args.elevation, radius)
            for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    focal * args.scale,
    z_near,
    z_far,
    c=c * args.scale if c is not None else None,
).to(device=device)
# (NV, H, W, 8)

focal = focal.to(device=device)

source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1
assert not (source >= NV).any()

if renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.n_coarse = 64
    renderer.n_fine = 128

with torch.no_grad():
    print("Encoding source view(s)")
    if random_source:
        src_view = torch.randint(0, NV, (1,))
    else:
        src_view = source

    net.encode(
        images[src_view].unsqueeze(0),
        poses[src_view].unsqueeze(0).to(device=device),
        focal,
        c=c,
    )

    print("Rendering", args.num_views * H * W, "rays")
    all_rgb_fine = []
    for rays in tqdm.tqdm(
        torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
    ):
        rgb, _depth = render_par(rays[None])
        all_rgb_fine.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    # rgb_fine (V*H*W, 3)

    if args.encoder_mode == "active":
        # Obtain original image dimensions from the dataset
        NV, _, H_orig, W_orig = images.shape  # NV: Number of views, H_orig: Original height, W_orig: Original width

        # Apply scaling to the original dimensions if a scale factor is provided
        if args.scale != 1.0:
            H = int(H_orig * args.scale)  # Scaled height
            W = int(W_orig * args.scale)  # Scaled width
        else:
            H = H_orig  # Use original height if no scaling
            W = W_orig  # Use original width if no scaling

        # Debugging information to check dimensions
        print(f"Original H: {H_orig}, W: {W_orig}")  # Output the original dimensions
        print(f"Scaled H: {H}, W: {W}")  # Output the scaled dimensions (after applying args.scale)
        print(f"rgb_fine.shape: {rgb_fine.shape}")  # Output the shape of rgb_fine (number of pixels and channels)
        print(f"rgb_fine.numel(): {rgb_fine.numel()}, expected: {H * W * 3 * rgb_fine.shape[0]}")  # Compare the actual and expected number of elements

        # Calculate the expected number of elements based on the scaled dimensions and the number of frames
        expected_numel = H * W * 3 * rgb_fine.shape[0]

        # Check if the actual number of elements matches the expected number
        '''if rgb_fine.numel() != expected_numel:
            raise RuntimeError(f"Mismatch in reshaping. rgb_fine has {rgb_fine.numel()} elements, expected {expected_numel}")'''

        # Reshape rgb_fine to match the expected dimensions for frames
        frames = rgb_fine.view(-1, H, W, 3)

        # Pass the frames through the encoder-decoder model for upsampling, if needed
        frames = encoder_decoder(frames.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    
    else:
        frames = rgb_fine.view(-1, H, W, 3)

print("Writing video")
vid_name = "{:04}".format(args.subset)
if args.split == "test":
    vid_name = "t" + vid_name
elif args.split == "val":
    vid_name = "v" + vid_name
vid_name += "_v" + "_".join(map(lambda x: "{:03}".format(x), source))
vid_path = os.path.join(args.visual_path, args.name, "video" + vid_name + ".mp4")
viewimg_path = os.path.join(
    args.visual_path, args.name, "video" + vid_name + "_view.jpg"
)

imageio.mimwrite(
    vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=args.fps, quality=8
)

img_np = (data["images"][src_view].permute(0, 2, 3, 1) * 0.5 + 0.5).numpy()
img_np = (img_np * 255).astype(np.uint8)
img_np = np.hstack((*img_np,))
imageio.imwrite(viewimg_path, img_np)

print("Wrote to", vid_path, "view:", viewimg_path)
=======
# Save the video
output_path = os.path.join(args.visual_path, f"{args.name}.mp4")
imageio.mimwrite(output_path, all_rgb_frames, fps=args.fps)
print(f"Video saved to {output_path}")
>>>>>>> c64aab0d01e0521b9e1dc910c19773b8f70f0c36
