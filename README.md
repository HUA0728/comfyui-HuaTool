# comfyui-HuaTool
极简实用的 ComfyUI 图像批量处理工具包，专注于解决两个核心痛点：批量加载文件夹图片 和 保持原文件名保存。无需复杂设置，连接两个节点即可实现成百上千张图片的自动化处理 workflow，完美适配批量放大、重绘、格式转换等场景。


🔹 节点详解


🔹 从文件夹加载图片

folder_path: 文件夹路径（相对ComfyUI根目录或绝对路径）

max_images: 最大加载数量（0=全部）

start_index: 起始索引（从第几张开始加载）

输出: 图片批次、文件名列表(JSON)、数量


🔹 批量保存图片(保持原文件名)

images: 输入图片批次

output_folder: 输出文件夹

filenames_json: 连接"从文件夹加载图片"的文件名输出，保持原文件名

image_format: 格式选择（png/jpg/webp）

quality: JPG/WEBP质量（1-100）

add_suffix: 是否添加后缀

suffix: 自定义后缀（如 _processed）



🔹 使用示例


🔹 场景1：批量放大图片并保持原名

将图片放入 ComfyUI/input/my_images

"从文件夹加载图片" 设置 folder_path 为 input/my_images

连接放大节点（如 UltimateSDUpscale）

"批量保存图片" 设置 output_folder 为 output/upscaled

连接 filenames_json 以保持原文件名

结果：photo.jpg → photo.png（或保持原扩展名取决于格式选择）


🔹 场景2：添加处理后缀

设置 add_suffix=true, suffix="_hd"

结果：photo.jpg → photo_hd.png


<img width="750" height="229" alt="ScreenShot_2026-03-17_162307_532" src="https://github.com/user-attachments/assets/370126a3-de39-4dcf-87b9-f3d264c449e4" />

两个节点通过 filenames_json 连接即可实现原文件名对应保存！



