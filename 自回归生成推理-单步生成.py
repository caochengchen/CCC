

"""输入一个句子获取文本编码向量"""
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


def extract_v3b_embedding(text, model_path, output_path):
    """
    使用V3b模型提取文本嵌入向量并保存到文件

    参数:
    text (str): 输入文本
    model_path (str): V3b模型路径
    output_path (str): 输出文件路径
    """
    print(f"加载模型: {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)

    print(f"处理文本: '{text}'")
    # 编码句子获取嵌入向量
    embedding = model.encode([text])[0]  # 获取第一个（也是唯一一个）句子的嵌入向量

    print(f"嵌入向量维度: {embedding.shape}")

    # 保存嵌入向量到文件
    print(f"保存嵌入向量到: {output_path}")
    with open(output_path, 'w') as f:
        # 将向量写入文件，每个值用空格分隔
        f.write(' '.join(map(str, embedding)))

    print("嵌入向量提取完成!")
    return embedding


def main():
    parser = argparse.ArgumentParser(description="使用V3b模型提取文本嵌入向量")
    parser.add_argument("--text", type=str, required=True, help="输入文本")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/German_Semantic_V3b",
                        help="V3b模型路径")
    parser.add_argument("--output", type=str, default="/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/test_v3b_embedding.txt",
                        help="输出文件路径")

    args = parser.parse_args()

    # 提取并保存嵌入向量
    embedding = extract_v3b_embedding(args.text, args.model_path, args.output)

    # 显示向量的一部分
    preview_length = min(10, len(embedding))
    print(f"\n嵌入向量预览 (前{preview_length}个值):")
    print(embedding[:preview_length])


if __name__ == "__main__":
    main()








"""获取MLP输出"""
import torch
import os
import numpy as np
from tqdm import tqdm
from 文本编码正负样本.model import TransformerModel
from 文本编码正负样本.config import Config
import torch.nn.functional as F


def extract_text_features(model_path, text_path, output_dir="./extracted_features"):
    """
    加载预训练模型并提取文本特征

    Args:
        model_path: 模型路径
        text_path: 文本特征路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载配置
    config = Config()

    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading model from {model_path}")
    model = TransformerModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 如果有多个GPU，使用DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for feature extraction")
        model = torch.nn.DataParallel(model)

    # 直接加载文本特征，不使用MotionTextDataset避免加载动作特征
    print(f"Loading text features from {text_path}")
    text_features = []
    with open(text_path, 'r') as f:
        for line in f.readlines():
            text_features.append(np.array([float(x) for x in line.strip().split()]))

    # 提取特征
    raw_features = []
    normalized_features = []

    print("Extracting features...")
    with torch.no_grad():
        for i, text_feature in enumerate(tqdm(text_features)):
            # 转换为张量并移到设备
            text = torch.FloatTensor(text_feature).unsqueeze(0).to(device)

            # 提取原始和归一化特征
            if isinstance(model, torch.nn.DataParallel):
                # 对于DataParallel模型，先获取module
                # 注意：DataParallel包装的模型无法直接访问text_mlp，所以需要一些特殊处理
                raw_feature = model.module.text_mlp(text)
                # 调用encode_text方法会同时设置raw_text_features和返回归一化特征
                normalized_feature = model.module.encode_text(text)
            else:
                raw_feature = model.text_mlp(text)
                normalized_feature = model.encode_text(text)

            # 保存特征
            raw_features.append(raw_feature.cpu().numpy()[0])
            normalized_features.append(normalized_feature.cpu().numpy()[0])

    # 保存特征到文件
    raw_output_path = os.path.join(output_dir, "归一化前文本特征.txt")
    norm_output_path = os.path.join(output_dir, "归一化后文本特征.txt")

    print(f"Saving raw features to {raw_output_path}")
    with open(raw_output_path, 'w') as f:
        for feature in raw_features:
            f.write(' '.join([str(x) for x in feature]) + '\n')

    print(f"Saving normalized features to {norm_output_path}")
    with open(norm_output_path, 'w') as f:
        for feature in normalized_features:
            f.write(' '.join([str(x) for x in feature]) + '\n')

    print("Feature extraction completed!")


if __name__ == "__main__":
    # 直接在代码中设置参数
    model_path = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/文本编码正负样本/checkpoints/model_epoch_33.pth"  # 修改为您的模型路径
    text_path = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/test_v3b_embedding.txt"  # 修改为您想要处理的文本特征文件
    output_dir = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/自回归生成的MLP输出特征/test"  # 修改为您想要的输出目录

    extract_text_features(model_path, text_path, output_dir)







"""自回归生成推理"""
import os
import torch
import numpy as np
from tqdm import tqdm
from 自回归生成.model import Text2Motion_Transformer
from 自回归生成.config import Config
from collections import OrderedDict


def generate_motion_from_features(model_path, text_feature_path, output_path):
    """
    从文本特征文件生成动作序列

    Args:
        model_path: 训练好的模型路径
        text_feature_path: 文本特征文件路径
        output_path: 输出文件路径
    """
    # 加载配置
    config = Config()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading model from {model_path}")
    model = Text2Motion_Transformer(
        num_vq=config.num_vq,
        embed_dim=config.embed_dim,
        clip_dim=config.clip_dim,
        block_size=config.block_size,
        num_layers=config.num_layers,
        n_head=config.n_head,
        drop_out_rate=config.drop_out_rate,
        fc_rate=config.fc_rate,
        max_action_length=config.max_action_length
    ).to(device)

    try:
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)

        # 检查是否是DataParallel或DDP保存的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("检测到DataParallel/DDP保存的模型，调整键名...")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # 移除'module.'前缀
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("尝试非严格加载模式...")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded with non-strict mode!")
        except Exception as e2:
            print(f"Non-strict loading also failed: {e2}")
            return

    model.eval()

    # 加载文本特征
    print(f"Loading text features from {text_feature_path}")
    text_features = []
    with open(text_feature_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            text_features.append(values)

    print(f"Loaded {len(text_features)} text features")

    # 生成动作序列
    all_generated = []

    with torch.no_grad():
        for i, text_feature in enumerate(tqdm(text_features, desc="Generating motions")):
            # 转换为张量
            text_feature = torch.tensor(text_feature, dtype=torch.float32).unsqueeze(0).to(device)

            try:
                # 生成序列
                generated_seq = model.sample(text_feature, if_categorial=True)

                if generated_seq is None:
                    print(f"Warning: model.sample returned None for sample {i}")
                    generated_seq = torch.tensor([[1]], device=device)  # 使用默认值

                # 将生成的索引转换为numpy数组并减1（恢复原始范围）
                generated_seq = generated_seq.cpu().numpy()[0]
                generated_seq = [x - 1 for x in generated_seq]

                all_generated.append(generated_seq)
                print(f"Generated sequence {i + 1} with length {len(generated_seq)}")
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                all_generated.append([0])  # 添加一个默认序列

    # 保存生成的序列
    print(f"Saving generated sequences to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for seq in all_generated:
            f.write(' '.join(map(str, seq)) + '\n')

    print(f"Generated {len(all_generated)} sequences")
    print("Generation completed!")


if __name__ == "__main__":
    # 设置路径
    model_path = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/自回归生成/output/text2motion_model.pt"  # 训练好的模型路径
    text_feature_path = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/自回归生成的MLP输出特征/test/归一化后文本特征.txt"  # 文本特征文件路径
    output_path = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/test_codebook_indices-generated.txt"  # 输出文件路径

    # 生成动作序列
    generate_motion_from_features(model_path, text_feature_path, output_path)














"""从索引解码生成动作"""
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


def main():
    # 设置文件路径
    model_path = 'checkpoints/whole_body_data_vqvae_checkpoints_256_60_512/best_model.pt'
    indices_file = '/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/test_codebook_indices-generated.txt'
    output_file = '/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/test_codebook_indices_decoder.txt'

    # 直接进行解码过程，不检查文件是否存在
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

    # 保存重建的序列（直接覆盖已存在的文件）
    with open(output_file, 'w') as f:
        for seq in reconstructed_sequences:
            seq_str = ' '.join(map(str, seq.flatten()))
            f.write(seq_str + '\n')

    print(f"重建的序列已保存到 {output_file}")


if __name__ == '__main__':
    main()






"""动作可视化（仅生成部分）"""
import numpy as np
import cv2
import multiprocessing
import os
from PIL import Image


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

    # 删除原有检查逻辑，直接重新生成所有帧
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
    # 删除检查输出文件夹是否存在的逻辑
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


def create_gif_from_frames(frames_folder, output_gif_path, duration=80):
    """从帧图片创建GIF动画"""
    # 获取所有帧图片
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.startswith('frame_') and f.endswith('.png')])

    if not frame_files:
        print(f"警告: 文件夹 {frames_folder} 中没有有效的帧图片")
        return

    # 打开所有帧图片
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = Image.open(frame_path)
        frames.append(frame)

    # 保存为GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print(f"已生成GIF: {output_gif_path}")


def create_all_gifs(generated_base_path, output_base_path):
    """为所有生成的序列创建GIF动画"""
    os.makedirs(output_base_path, exist_ok=True)

    # 获取所有生成图片的文件夹
    generated_folders = sorted(
        [f for f in os.listdir(generated_base_path) if os.path.isdir(os.path.join(generated_base_path, f))])

    for folder_name in generated_folders:
        output_gif_path = os.path.join(output_base_path, f'{folder_name}.gif')

        # 直接创建GIF，不检查是否已存在
        frames_folder = os.path.join(generated_base_path, folder_name)
        create_gif_from_frames(frames_folder, output_gif_path)


if __name__ == "__main__":
    # 处理预测数据（生成图片）
    pred_filename = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/test_codebook_indices_decoder.txt"
    pred_output_prefix = "/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/单步生成图片/"
    os.makedirs(pred_output_prefix, exist_ok=True)
    process_all_sequences(pred_filename, pred_output_prefix, max_sequences=642)

    # 为生成的图片创建GIF动画
    gif_output_path = "/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/单步生成视频/"
    create_all_gifs(pred_output_prefix, gif_output_path)


