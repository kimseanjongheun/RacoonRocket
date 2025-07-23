import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

# 폴더 경로 지정
gt_folder_path = '../Data/leaderboard'
model_name = input('enter the model name(ex. Varnet_0707_1): ')
recon_folder_path = f'../result/{model_name}/reconstructions_leaderboard'

gt_image_type_list = ['acc4/image', 'acc8/image']
recon_image_type_list = ['acc4', 'acc8']
body_types = ['brain', 'knee']  # ✅ 공통 리스트로 통일

# ✅ model_name 하위 폴더 생성
output_dir = os.path.join('visualization', model_name)
os.makedirs(output_dir, exist_ok=True)

slice_idx = 0
max_count = 2  # ✅ 각 acc, body type 조합 당 최대 저장 수

for gt_subdir, recon_subdir in zip(gt_image_type_list, recon_image_type_list):
    gt_path = os.path.join(gt_folder_path, gt_subdir)
    recon_path = os.path.join(recon_folder_path, recon_subdir)

    gt_filenames = sorted([f for f in os.listdir(gt_path) if f.endswith('.h5')])
    recon_filenames = sorted([f for f in os.listdir(recon_path) if f.endswith('.h5')])

    # ✅ (acc, body_type) 조합 별 카운트 초기화
    count_map = {body: 0 for body in body_types}

    for gt_file, recon_file in zip(gt_filenames, recon_filenames):
        # 파일 이름에서 body type 판단
        matched_type = None
        for body in body_types:
            if body in gt_file.lower():
                matched_type = body
                break

        if matched_type is None or count_map[matched_type] >= max_count:
            continue

        gt_file_path = os.path.join(gt_path, gt_file)
        recon_file_path = os.path.join(recon_path, recon_file)

        with h5py.File(gt_file_path, 'r') as gt_f:
            target = gt_f['image_label'][slice_idx]
            vmax = gt_f.attrs['max']

        with h5py.File(recon_file_path, 'r') as recon_f:
            recon = recon_f['reconstruction'][slice_idx]

        target_norm = target / vmax
        recon_norm = recon / vmax

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(target_norm, cmap='gray')
        axs[0].set_title(f'GT ({matched_type})')
        axs[0].axis('off')

        axs[1].imshow(recon_norm, cmap='gray')
        axs[1].set_title('Reconstruction')
        axs[1].axis('off')

        recon_tag = recon_subdir.replace('/', '_')
        base_name = os.path.splitext(gt_file)[0]
        output_filename = f'comparison_{recon_tag}_{matched_type}_{base_name}_slice{slice_idx}.png'
        output_path = os.path.join(output_dir, output_filename)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        count_map[matched_type] += 1
        print(f'✅ 저장 완료: {output_filename}')
