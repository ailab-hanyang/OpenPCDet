import glob
from pathlib import Path
import open3d as o3d
import numpy as np
import shutil
from rich import print
from rich.progress import track
from multiprocessing import Pool, cpu_count

def print_config(root_path, save_path, scenarios, sample_rate):
    print("[bold blue]Configuration[/bold blue]:")
    print(f"[bold green]Root path:[/bold green] {root_path}")
    print(f"[bold green]Save path:[/bold green] {save_path}")
    print(f"[bold green]Scenarios:[/bold green] {scenarios}")
    print(f"[bold green]Sample rate:[/bold green] {sample_rate}")

def load_label_files(root_path, scenarios):
    label_file_list = []
    for scenario in scenarios:
        label_file_list += glob.glob(str(root_path / scenario / '**/*.txt'), recursive=True)
    label_file_list.sort()
    return label_file_list

def copy_label_files(label_file_list, root_path, save_path, scenarios):
    for label_file in track(label_file_list, description="Copying label files..."):
        scenario = next(s for s in scenarios if f'/{s}/' in label_file)
        save_label_file = str(label_file).replace(str(root_path), str(save_path)).replace(f'/{scenario}', '/labels')
        save_label_file = Path(save_label_file)
        save_label_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(label_file, save_label_file)

def convert_pcd_to_npy(label_file, root_path, save_path, scenarios):
    lidar_file = str(label_file).replace('labels', 'points').replace('.txt', '.pcd')
    pcd = o3d.t.io.read_point_cloud(lidar_file)
    xyz = np.asarray(pcd.point.positions.numpy(), dtype=np.float32)
    intensity = np.asarray(pcd.point.intensity.numpy(), dtype=np.float32)
    points = np.concatenate([xyz, intensity], axis=1)
    
    scenario = next(s for s in scenarios if f'/{s}/' in label_file)
    save_npy_file = str(label_file).replace(str(root_path), str(save_path)).replace(f'/{scenario}', '/points').replace('.txt', '.npy')
    save_npy_file = Path(save_npy_file)
    save_npy_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_npy_file, points) 

def convert_and_save_pcd(label_file_list, root_path, save_path, scenarios):
    with Pool(cpu_count()) as pool:
        pool.starmap(convert_pcd_to_npy, [(label_file, root_path, save_path, scenarios) for label_file in label_file_list])

def create_image_sets(label_file_list, save_path, sample_rate, split='test'):
    sampled_label_file_list = label_file_list[::sample_rate]
    ImageSets = save_path / 'ImageSets'
    ImageSets.mkdir(parents=True, exist_ok=True)
    with open(ImageSets / (split+'.txt'), 'w') as f:
        for label_file in sampled_label_file_list:
            f.write(label_file.split('/')[-1].replace('.txt', '') + '\n')
    print(f"Saved into {ImageSets / (split+'.txt')}")

def main():
    # V2 Configuration
    # root_path = Path("/home/ailab/AILabDataset/02_Custom_Dataset/21_RSCube_Dataset/V2/Raw")
    # save_path = Path("/home/ailab/AILabDataset/02_Custom_Dataset/21_RSCube_Dataset/V4/Processed")
    # scenarios = ['test']
    # split = {'test': ['urban', 'highway']}
    # sample_rate = 2

    # V3 Configuration
    root_path = Path("/home/ailab/AILabDataset/02_Custom_Dataset/21_RSCube_Dataset/V3/Raw")
    save_path = Path("/home/ailab/AILabDataset/02_Custom_Dataset/21_RSCube_Dataset/V3/Processed")
    scenarios = ['train', 'val']
    split = {'train': ['1_1', '2_6', '2_8', '2_9', '2_10', '2_11', '2_12', '2_13'],
             'val': ['1_2', '2_7', '2_14']}
    sample_rate = 1

    print_config(root_path, save_path, scenarios, sample_rate)
    for scenario in scenarios:
        label_file_list = load_label_files(root_path, split[scenario])
        label_file_list.sort()
        create_image_sets(label_file_list, save_path, sample_rate, split=scenario)
        copy_label_files(label_file_list, root_path, save_path, split[scenario])
        convert_and_save_pcd(label_file_list, root_path, save_path, split[scenario])
    
    # Multiprocessing
    # convert_and_save_pcd(label_file_list, root_path, save_path, scenarios)
    
    # Single processing
    # for label_file in label_file_list:
    #     convert_pcd_to_npy(label_file, root_path, save_path, scenarios)
    
if __name__ == "__main__":
    main()
