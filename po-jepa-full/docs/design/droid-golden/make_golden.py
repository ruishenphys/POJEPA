"""Golden references from the OFFICIAL code paths (vjepa2 @204698b) on real DROID samples."""
import sys, os, json, glob, hashlib, importlib.util, random
import numpy as np, h5py, cv2, torch
from scipy.spatial.transform import Rotation
sys.path.insert(0, sys.argv[1])
import src.datasets.utils.video.transforms as T
assert T.__file__.startswith(sys.argv[1]), T.__file__
from notebooks.utils.mpc_utils import compute_new_pose  # official state integration used by the AC planner
assert sys.modules["notebooks.utils.mpc_utils"].__file__.startswith(sys.argv[1])
out_path = sys.argv[2]; eps = sys.argv[3:]

def poses_to_diffs(poses):  # verbatim from app/vjepa_droid/droid.py @204698b
    xyz = poses[:, :3]
    thetas = poses[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]
    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32) * 255.0
std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32) * 255.0
golden = dict(source="vjepa2@204698b45b3712590f06245fbfba32d3be539812 app/vjepa_droid/{droid,transforms}.py, src/datasets/utils/video/transforms.py",
              stride_frames=15, episodes=[])
for ep in eps:
    m = json.load(open(os.path.join(ep, "metadata.json")))
    mp4 = os.path.join(ep, os.path.basename(m["left_mp4_path"]))
    with h5py.File(os.path.join(ep, "trajectory.h5"), "r") as f:
        states = np.concatenate([f["observation/robot_state/cartesian_position"][()],
                                 f["observation/robot_state/gripper_position"][()][:, None]], axis=1)
        serial = os.path.basename(m["left_mp4_path"]).split(".")[0]
        cap_ts = f[f"observation/timestamp/cameras/{serial}_estimated_capture"][()]
        skip = f["observation/timestamp/skip_action"][()]
    cap = cv2.VideoCapture(mp4); frames = []
    container_fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    vlen = len(frames); Th5 = len(states)
    idx = np.arange(0, vlen, 15)
    st = states[idx]
    acts = poses_to_diffs(st)
    # official preprocessing of two frames (float32 [C,T,H,W] -> crop+bilinear -> normalize)
    buf = torch.tensor(np.stack([frames[0], frames[15]]), dtype=torch.float32).permute(3, 0, 1, 2)
    random.seed(0); np.random.seed(0)
    cropped = T.random_resized_crop(buf, 256, 256, scale=(1.777, 1.777), ratio=(0.75, 1.35))
    C, Tn, H, W = cropped.shape
    norm = cropped.view(C, -1).permute(1, 0).sub(mean).div(std).permute(1, 0).reshape(C, Tn, H, W)
    # independent re-derivation: slice [0:720,154:1126] + F.interpolate bilinear(align_corners=False)
    manual = torch.nn.functional.interpolate(buf[:, :, 0:720, 154:1126], size=(256, 256), mode="bilinear", align_corners=False)
    golden["episodes"].append(dict(
        uuid=m["uuid"], left_mp4=os.path.basename(m["left_mp4_path"]),
        h5_steps=Th5, video_frames=vlen, container_fps=container_fps,
        capture_dt_ms_median=float(np.median(np.diff(cap_ts))), skip_action_count=int(skip.sum()),
        stride15_indices=idx.tolist(), stride15_states=st.tolist(), official_actions=acts.tolist(),
        decoded_frame_sha256={str(i): sha(frames[i]) for i in (0, 15, 30, vlen - 1)},
        official_crop_shape=list(cropped.shape), official_crop_sha256=sha(cropped.numpy()),
        manual_crop_equal=bool(torch.equal(manual, cropped)),
        normalized_sha256=sha(norm.numpy()), normalized_sample=[float(norm[0, 0, 0, 0]), float(norm[1, 1, 128, 128]), float(norm[2, 0, 255, 255])],
        action_abs_max=np.abs(acts).max(0).tolist(),
        official_new_pose=compute_new_pose(torch.from_numpy(st[:-1, None]), torch.from_numpy(acts[:, None]))[:, 0].numpy().tolist()))
    print(os.path.basename(ep), "vlen", vlen, "T", Th5, "fps", container_fps, "frames@15", len(idx), "manual==official", bool(torch.equal(manual, cropped)),
          "max|a|", np.round(np.abs(acts).max(0), 3).tolist())
json.dump(golden, open(out_path, "w"), indent=1)
