"""
HuaTool - ComfyUI 图像批量处理工具包
=====================================

核心功能：
- 从文件夹批量加载图片（保持顺序）
- 批量保存图片（保持原文件名，支持多格式）

安装：将本文件夹复制到 ComfyUI/custom_nodes/ 目录下，重启 ComfyUI
"""

import os
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import folder_paths


# ==================== 节点1：从文件夹加载图片 ====================

class LoadImagesFromFolder:
    """从指定文件夹加载所有图片，返回图片批次和对应的文件名列表"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "input"}),
                "max_images": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "display": "number"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames_json", "count")
    FUNCTION = "load_images"
    CATEGORY = "HuaTool/image"

    def load_images(self, folder_path, max_images, start_index):
        # 解析路径
        if not os.path.isabs(folder_path):
            base_path = folder_paths.base_path
            folder = os.path.join(base_path, folder_path)
        else:
            folder = os.path.expanduser(folder_path)

        print(f"[HuaTool] 扫描文件夹: {folder}")

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tiff', '.tif'}
        
        if not os.path.exists(folder):
            raise FileNotFoundError(f"文件夹不存在: {folder}")
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"路径不是文件夹: {folder}")

        # 收集所有图片文件
        image_files = []
        folder_path_obj = Path(folder)
        for ext in valid_extensions:
            image_files.extend(folder_path_obj.glob(f"*{ext}"))
            image_files.extend(folder_path_obj.glob(f"*{ext.upper()}"))
        
        # 去重并排序（按文件名字母顺序）
        image_files = sorted(list(set(image_files)), key=lambda x: x.name.lower())
        
        if not image_files:
            raise ValueError(f"文件夹中没有图片: {folder}")

        total = len(image_files)
        print(f"[HuaTool] 找到 {total} 个图片文件")

        # 根据起始索引和最大数量截取
        end_index = start_index + max_images if max_images > 0 else None
        selected_files = image_files[start_index:end_index]
        
        if not selected_files:
            raise ValueError(f"索引范围无效: start={start_index}, max={max_images}, total={total}")

        # 加载图片
        images = []
        filenames = []
        
        for img_path in selected_files:
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)
                filenames.append(img_path.name)
                
            except Exception as e:
                print(f"[HuaTool] 警告: 无法加载 {img_path.name}: {e}")
                continue
        
        if not images:
            raise ValueError("没有成功加载任何图片")

        # 转换为tensor
        batch = np.stack(images, axis=0)
        image_tensor = torch.from_numpy(batch)
        filenames_json = json.dumps(filenames, ensure_ascii=False)
        
        print(f"[HuaTool] 成功加载 {len(images)} 张图片 (从第 {start_index} 张开始)")
        
        return (image_tensor, filenames_json, len(images))


# ==================== 节点2：批量保存图片（保持原文件名） ====================

class SaveImagesWithOriginalName:
    """批量保存图片，保持原文件名，支持PNG/JPG/WEBP格式"""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_folder": ("STRING", {"default": "output"}),
            },
            "optional": {
                "filenames_json": ("STRING", {"default": "", "multiline": False}),
                "image_format": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "add_suffix": ("BOOLEAN", {"default": False}),
                "suffix": ("STRING", {"default": "_processed"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("saved_paths", "count")
    FUNCTION = "save_images"
    CATEGORY = "HuaTool/save"
    OUTPUT_NODE = True

    def save_images(self, images, output_folder, filenames_json="", 
                   image_format="png", quality=95, add_suffix=False, suffix="_processed",
                   prompt=None, extra_pnginfo=None):

        # 解析保存路径
        if not os.path.isabs(output_folder):
            save_dir = os.path.join(self.output_dir, output_folder)
        else:
            save_dir = os.path.expanduser(output_folder)

        os.makedirs(save_dir, exist_ok=True)
        print(f"[HuaTool] 保存到: {save_dir}")

        # 确保是批次格式 [B,H,W,C]
        if images.dim() == 3:
            images = images.unsqueeze(0)

        batch_size = images.shape[0]

        # 解析文件名列表
        original_filenames = []
        if filenames_json and filenames_json.strip():
            try:
                parsed = json.loads(filenames_json)
                if isinstance(parsed, list):
                    original_filenames = parsed
            except Exception as e:
                print(f"[HuaTool] 警告: 无法解析filenames_json: {e}")

        # 处理后缀
        suffix_str = ""
        if add_suffix and suffix and suffix.strip():
            suffix_str = suffix.strip()
            if not suffix_str.startswith(('_', '-')):
                suffix_str = '_' + suffix_str

        # 生成保存文件名
        save_filenames = []
        if original_filenames and len(original_filenames) >= batch_size:
            # 使用原文件名
            for i in range(batch_size):
                original_name = original_filenames[i]
                name_without_ext = Path(original_name).stem
                new_name = f"{name_without_ext}{suffix_str}.{image_format}"
                save_filenames.append(new_name)
        else:
            # 使用默认命名
            for i in range(batch_size):
                new_name = f"image_{i:05d}{suffix_str}.{image_format}"
                save_filenames.append(new_name)
            if not original_filenames:
                print(f"[HuaTool] 提示: 未提供原文件名，使用默认命名 image_00000.{image_format} 等")

        # 保存图片
        saved_paths = []
        for i in range(batch_size):
            img_tensor = images[i]
            img_np = img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            save_path = os.path.join(save_dir, save_filenames[i])

            # 根据格式保存
            if image_format.lower() == "png":
                self._save_png(img_pil, save_path, prompt, extra_pnginfo)
            elif image_format.lower() in ["jpg", "jpeg"]:
                self._save_jpg(img_pil, save_path, quality)
            elif image_format.lower() == "webp":
                self._save_webp(img_pil, save_path, quality)

            saved_paths.append(save_path)
            print(f"[HuaTool] 已保存: {save_filenames[i]}")

        print(f"[HuaTool] 成功保存 {len(saved_paths)} 张图片")
        return (json.dumps(saved_paths, ensure_ascii=False), len(saved_paths))

    def _save_png(self, img_pil, save_path, prompt, extra_pnginfo):
        """保存为PNG格式，保留ComfyUI元数据"""
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])
        
        if metadata:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            for key, value in metadata.items():
                pnginfo.add_text(key, value)
            img_pil.save(save_path, format='PNG', optimize=True, compress_level=3, pnginfo=pnginfo)
        else:
            img_pil.save(save_path, format='PNG', optimize=True, compress_level=3)

    def _save_jpg(self, img_pil, save_path, quality):
        """保存为JPG格式"""
        # RGBA转RGB
        if img_pil.mode == 'RGBA':
            background = Image.new('RGB', img_pil.size, (255, 255, 255))
            background.paste(img_pil, mask=img_pil.split()[3])
            img_pil = background
        elif img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        img_pil.save(save_path, format='JPEG', quality=quality, optimize=True)

    def _save_webp(self, img_pil, save_path, quality):
        """保存为WEBP格式"""
        img_pil.save(save_path, format='WEBP', quality=quality, method=6)


# ==================== 节点注册 ====================

NODE_CLASS_MAPPINGS = {
    "LoadImagesFromFolder": LoadImagesFromFolder,
    "SaveImagesWithOriginalName": SaveImagesWithOriginalName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesFromFolder": "HuaTool 从文件夹加载图片",
    "SaveImagesWithOriginalName": "HuaTool 批量保存图片(保持原文件名)",
}

print(f"[HuaTool] 节点包加载完成，包含 {len(NODE_CLASS_MAPPINGS)} 个节点")