import os
import shutil
import re

def process_npy_files(refined_bbox_path):
    # 确保输入路径存在
    if not os.path.exists(refined_bbox_path):
        print(f"路径不存在: {refined_bbox_path}")
        return
    
    # 获取路径中的所有文件
    files = os.listdir(refined_bbox_path)
    
    # 过滤出.npy文件
    npy_files = [f for f in files if f.endswith('.png')]
    
    # 为每个相机ID创建目录
    created_dirs = set()
    
    for file in npy_files:
        # 使用正则表达式提取帧号和相机ID
        match = re.match(r'(\d{6})_(\d+)\.png', file)
        if not match:
            print(f"文件名格式不符合预期: {file}")
            continue
        
        frame_str, cam_id = match.groups()
        frame = int(frame_str)  # 转换为整数，去掉前导零
        
        # 创建目标目录
        target_dir = os.path.join(os.path.dirname(refined_bbox_path), f'refined_bbox_{cam_id}')
        if target_dir not in created_dirs:
            os.makedirs(target_dir, exist_ok=True)
            created_dirs.add(target_dir)
        
        # 生成新文件名
        new_filename = f'0017{frame:03d}.png'
        
        # 源文件和目标文件的完整路径
        source_path = os.path.join(refined_bbox_path, file)
        target_path = os.path.join(target_dir, new_filename)
        
        # 复制文件
        shutil.copy2(source_path, target_path)
        print(f"已处理: {file} -> {target_dir}/{new_filename}")

if __name__ == "__main__":
    refined_bbox_path = 'data/waymo_scenes/0017085/refined_bbox'
    process_npy_files(refined_bbox_path)
