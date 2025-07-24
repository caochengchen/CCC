
"""从索引解码重建动作"""

import torch
import numpy as np
from model import VQVAE
from tqdm import tqdm
import os


def decode_from_indices(model, indices, config):
    model.eval()
    with torch.no_grad():
        indices = torch.LongTensor(indices).to(config.device)
        one_hot = torch.zeros(indices.size(0), config.num_embeddings, device=config.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        quantized = torch.matmul(one_hot, model.vq.embedding.weight)
        quantized = quantized.unsqueeze(0)
        mask = torch.ones(1, indices.size(0), device=config.device)
        reconstruction = model.decoder(quantized, mask)
        reconstruction = reconstruction.squeeze(0)

    return reconstruction.cpu().numpy()


def check_sequence_lengths(original_file, decoded_file):
    print("\nChecking sequence lengths...")

    # 读取原始文件
    with open(original_file, 'r') as f:
        original_sequences = f.readlines()

    # 读取解码后的文件
    with open(decoded_file, 'r') as f:
        decoded_sequences = f.readlines()

    # 确保文件行数相同
    if len(original_sequences) != len(decoded_sequences):
        print(f"Error: Number of sequences doesn't match!")
        print(f"Original file has {len(original_sequences)} sequences")
        print(f"Decoded file has {len(decoded_sequences)} sequences")
        return

    mismatch_count = 0
    for i, (orig, decoded) in enumerate(zip(original_sequences, decoded_sequences)):
        # 将字符串转换为数字列表
        orig_values = orig.strip().split()
        decoded_values = decoded.strip().split()

        # 计算原始序列的帧数（每帧42+42+16个特征）
        orig_frames = len(orig_values) // 100
        # 计算解码序列的帧数
        decoded_frames = len(decoded_values) // 100

        if orig_frames != decoded_frames:
            mismatch_count += 1
            print(f"Sequence {i}: Length mismatch!")
            print(f"Original frames: {orig_frames}, Decoded frames: {decoded_frames}")

    if mismatch_count == 0:
        print("All sequences have matching lengths!")
    else:
        print(f"\nFound {mismatch_count} sequences with mismatched lengths")
        print(f"Total sequences checked: {len(original_sequences)}")


def main():
    # 设置文件路径
    model_path = 'checkpoints/whole_body_data_vqvae_checkpoints_256_60_512/best_model.pt'
    indices_file = '/mnt/sda/caochengcheng/一条龙/演示/4-可视化/test_codebook_indices-ccc.txt'
    output_file = '/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/output/自回归生成-全部-test_codebook_indices_decoder.txt'
    original_file = 'datas/test_whole_body_downsampled_data.txt'

    # 检查输出文件是否存在
    if os.path.exists(output_file):
        print(f"输出文件已存在: {output_file}")
        print("直接进行序列长度检查...")
        check_sequence_lengths(original_file, output_file)
        return

    # 如果文件不存在，进行解码过程
    print("开始解码过程...")

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cuda:3' if torch.cuda.is_available() else 'cpu')
    config = checkpoint['config']

    model = VQVAE(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 读取codebook indices并解码
    reconstructed_sequences = []
    with open(indices_file, 'r') as f:
        for line in tqdm(f, desc='Decoding sequences'):
            indices = [int(x) for x in line.strip().split()]
            reconstruction = decode_from_indices(model, indices, config)
            reconstructed_sequences.append(reconstruction)

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存重建的序列
    with open(output_file, 'w') as f:
        for seq in reconstructed_sequences:
            seq_str = ' '.join(map(str, seq.flatten()))
            f.write(seq_str + '\n')

    print(f"重建的序列已保存到 {output_file}")

    # 检查序列长度
    check_sequence_lengths(original_file, output_file)


if __name__ == '__main__':
    main()





"""动作可视化"""
import numpy as np
import cv2
import multiprocessing
import os


def read_motion_data(filename):
    """读取动作序列数据"""
    sequences = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == 'ERROR':
                sequences.append(None)
                continue
            try:
                # 将每行数据转换为numpy数组
                data = np.fromstring(line.strip(), sep=' ', dtype=np.float32)
                # 重塑数据为 (帧数, 50, 2) 的形状
                frames = len(data) // 100
                sequence = data.reshape(frames, 50, 2)
                sequences.append(sequence)
            except:
                sequences.append(None)
    return sequences


def get_colorful_skeleton_connections():
    """定义彩色骨骼连接"""
    connections = [
        # 第一组 红色 - 包含中点连接
        [(0, 1), (2, 3), (2, 4), (3, 5), (4, 6), (5, 7), (-1, -2)],  # -1和-2是中点标记
        # 第二组 绿色
        [(8, 9), (9, 10), (10, 11), (11, 12)],
        # 第三组 蓝色
        [(8, 13), (13, 14), (14, 15), (15, 16)],
        # 第四组 黄色
        [(8, 17), (17, 18), (18, 19), (19, 20)],
        # 第五组 紫色
        [(8, 21), (21, 22), (22, 23), (23, 24)],
        # 第六组 青色
        [(8, 25), (25, 26), (26, 27), (27, 28)],
        # 第七组 橙色
        [(29, 30), (30, 31), (31, 32), (32, 33)],
        # 第八组 粉色
        [(29, 34), (34, 35), (35, 36), (36, 37)],
        # 第九组 棕色
        [(29, 38), (38, 39), (39, 40), (40, 41)],
        # 第十组 灰色
        [(29, 42), (42, 43), (43, 44), (44, 45)],
        # 第十一组 深绿
        [(29, 46), (46, 47), (47, 48), (48, 49)]
    ]

    colors = [
        (0, 0, 255),  # 红色
        (0, 255, 0),  # 绿色
        (255, 0, 0),  # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 紫色
        (255, 255, 0),  # 青色
        (0, 165, 255),  # 橙色
        (180, 105, 255),  # 粉色
        (42, 42, 165),  # 棕色
        (128, 128, 128),  # 灰色
        (0, 100, 0)  # 深绿
    ]

    return connections, colors


def normalize_points(points, target_width=210, target_height=260):
    """
    标准化和缩放关键点
    假设输入点在某个坐标系统中，需要映射到目标图像尺寸
    """
    # 找到x和y的最小和最大值
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    # 处理极端情况（所有点都在同一位置）
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1

    # 缩放和平移
    scaled_x = ((points[:, 0] - x_min) / x_range) * (target_width - 10) + 5
    scaled_y = ((points[:, 1] - y_min) / y_range) * (target_height - 10) + 5

    return np.column_stack([scaled_x, scaled_y]).astype(np.int32)


def draw_skeleton_frame(frame):
    """绘制单帧骨骼"""
    connections, colors = get_colorful_skeleton_connections()
    img = np.full((260, 210, 3), 255, dtype=np.uint8)

    if frame is not None:
        # 标准化和缩放关键点
        scaled_points = normalize_points(frame)

        # 计算中点
        midpoint1 = ((scaled_points[0] + scaled_points[1]) // 2).astype(np.int32)
        midpoint2 = ((scaled_points[2] + scaled_points[3]) // 2).astype(np.int32)

        # 将中点添加到scaled_points中
        scaled_points = np.vstack([scaled_points, midpoint1, midpoint2])

        # 绘制彩色骨骼连接
        for conn_group, color in zip(connections, colors):
            for start, end in conn_group:
                # 处理中点连接
                if start == -1 and end == -2:
                    start_point = tuple(scaled_points[-2])  # 第一个中点
                    end_point = tuple(scaled_points[-1])  # 第二个中点
                else:
                    start_point = tuple(scaled_points[start])
                    end_point = tuple(scaled_points[end])
                cv2.line(img, start_point, end_point, color, 2)

        # 绘制关键点
        for point in scaled_points[:-2]:  # 不绘制中点
            cv2.circle(img, tuple(point), 3, (0, 0, 0), -1)  # 黑色点

        # 绘制中点（用不同的颜色标识）
        cv2.circle(img, tuple(midpoint1), 3, (0, 255, 0), -1)  # 绿色
        cv2.circle(img, tuple(midpoint2), 3, (0, 255, 0), -1)  # 绿色

    return img


def process_sequence(args):
    """处理单个序列,将每一帧保存为PNG图像"""
    sequence, output_folder = args
    os.makedirs(output_folder, exist_ok=True)

    # 检查是否已经有足够的帧图片
    existing_frames = len([f for f in os.listdir(output_folder) if f.startswith('frame_') and f.endswith('.png')])
    if existing_frames > 0 and sequence is not None and existing_frames == len(sequence):
        return output_folder

    if sequence is None:
        # 如果序列无效,创建一个带有ERROR信息的图像
        error_img = np.full((260, 210, 3), 255, dtype=np.uint8)
        cv2.putText(error_img, "ERROR", (70, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_folder, "error.png"), error_img)
    else:
        for i, frame in enumerate(sequence):
            img = draw_skeleton_frame(frame)
            cv2.imwrite(os.path.join(output_folder, f"frame_{i:04d}.png"), img)

    return output_folder


def check_output_folders_exist(base_path, max_sequences):
    for i in range(max_sequences):
        folder_path = os.path.join(base_path, f"{i}")
        if not os.path.exists(folder_path):
            return False
    return True


def process_all_sequences(input_filename, output_prefix, max_sequences=None):
    """并行处理所有序列,每个序列的每一帧保存为PNG图像"""
    # 检查输出文件夹是否已存在
    if check_output_folders_exist(output_prefix, max_sequences):
        print(f"输出文件夹 {output_prefix} 已存在，跳过处理")
        return

    sequences = read_motion_data(input_filename)

    # 如果指定了最大序列数
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    # 准备参数
    tasks = [(seq, f"{output_prefix}_{i}") for i, seq in enumerate(sequences)]

    # 使用进程池并行处理
    with multiprocessing.Pool() as pool:
        results = pool.map(process_sequence, tasks)

    for result in results:
        print(f"已生成图像序列: {result}")


if __name__ == "__main__":
    # 处理预测数据
    pred_filename = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/output/自回归生成-全部-test_codebook_indices_decoder.txt"
    pred_output_prefix = "/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/全部生成图片/"
    os.makedirs(pred_output_prefix, exist_ok=True)
    process_all_sequences(pred_filename, pred_output_prefix, max_sequences=642)

    # 处理真实数据
    real_filename = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/datas/test_whole_body_downsampled_data.txt"
    real_output_prefix = "/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/全部真实图片/"
    os.makedirs(real_output_prefix, exist_ok=True)
    process_all_sequences(real_filename, real_output_prefix, max_sequences=642)







"""生成对比视频"""
import os
from PIL import Image, ImageDraw, ImageFont


def check_gif_exists(folder_name, output_base_path):
    gif_path = os.path.join(output_base_path, f'{folder_name}_comparison.gif')
    return os.path.exists(gif_path)


def main():
    # 定义路径
    original_base_path = '/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/全部真实图片/'
    generated_base_path = '/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/全部生成图片/'
    output_base_path = '/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/全部对比视频/'

    # 确保输出目录存在
    os.makedirs(output_base_path, exist_ok=True)

    # 获取所有原始和生成图片的文件夹
    original_folders = sorted(
        [f for f in os.listdir(original_base_path) if os.path.isdir(os.path.join(original_base_path, f))])
    generated_folders = sorted(
        [f for f in os.listdir(generated_base_path) if os.path.isdir(os.path.join(generated_base_path, f))])

    # 尝试加载支持中文的字体
    try:
        # 尝试几种常见的中文字体路径
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驿微米黑
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Droid字体
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
            '/usr/share/fonts/truetype/arphic/uming.ttc',  # AR PL UMing
        ]

        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 16)
                break

        # 如果没有找到任何字体，使用默认字体
        if font is None:
            # 使用英文字符作为替代
            use_english = True
            print("警告：未找到支持中文的字体，将使用英文标题")
        else:
            use_english = False

    except Exception as e:
        print(f"字体加载错误: {e}")
        use_english = True
        font = None

    # 处理每个对应的文件夹
    for folder_name in original_folders:
        if folder_name not in generated_folders:
            print(f"警告: 生成图片中没有对应的文件夹 {folder_name}，跳过")
            continue

        # 检查GIF是否已存在
        if check_gif_exists(folder_name, output_base_path):
            print(f"GIF文件 {folder_name}_comparison.gif 已存在，跳过")
            continue

        original_folder_path = os.path.join(original_base_path, folder_name)
        generated_folder_path = os.path.join(generated_base_path, folder_name)

        # 获取两个文件夹中的图片文件
        original_images = sorted([f for f in os.listdir(original_folder_path) if f.endswith('.png')])
        generated_images = sorted([f for f in os.listdir(generated_folder_path) if f.endswith('.png')])

        # 检查图片数量是否一致
        if len(original_images) != len(generated_images):
            print(
                f"警告: 文件夹 {folder_name} 中图片数量不一致 (原始: {len(original_images)}, 生成: {len(generated_images)})")

        # 创建一个列表来存储合并后的帧
        combined_frames = []

        # 合并每一帧
        for original_img_name, generated_img_name in zip(original_images, generated_images):
            # 打开原始视频帧和生成视频帧
            original_image = Image.open(os.path.join(original_folder_path, original_img_name))
            generated_image = Image.open(os.path.join(generated_folder_path, generated_img_name))

            # 创建一个新的图像，宽度为原始视频和生成视频的宽度之和
            combined_image = Image.new('RGB', (original_image.width + generated_image.width, original_image.height))

            # 将原始视频帧和生成视频帧粘贴到新图像中
            combined_image.paste(original_image, (0, 0))
            combined_image.paste(generated_image, (original_image.width, 0))

            # 添加标题
            title_image = Image.new('RGB', (combined_image.width, 30), (255, 255, 255))
            title_image_draw = ImageDraw.Draw(title_image)

            # 根据字体情况选择显示的文字
            left_title = "Original Video" if use_english else "原始视频"
            right_title = "Generated Video" if use_english else "生成视频"

            title_image_draw.text((10, 5), left_title, fill=(0, 0, 0), font=font)
            title_image_draw.text((original_image.width + 10, 5), right_title, fill=(0, 0, 0), font=font)

            # 将标题和合并后的图像合并
            final_image = Image.new('RGB', (combined_image.width, combined_image.height + 30))
            final_image.paste(title_image, (0, 0))
            final_image.paste(combined_image, (0, 30))

            # 将最终图像添加到帧列表中
            combined_frames.append(final_image)

        # 保存为GIF
        output_gif_path = os.path.join(output_base_path, f'{folder_name}_comparison.gif')

        # 如果没有任何帧，跳过
        if not combined_frames:
            print(f"警告: 文件夹 {folder_name} 中没有有效的帧，跳过")
            continue

        # 保存GIF，设置适当的帧率
        combined_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=80,  # 每帧显示时间(毫秒)
            loop=0,  # 无限循环
            optimize=True
        )

        print(f"已生成对比GIF: {output_gif_path}")

    print("所有对比GIF生成完成")


if __name__ == "__main__":
    main()



