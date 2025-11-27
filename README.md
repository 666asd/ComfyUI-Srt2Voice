# ComfyUI-Srt2Voice

基于 [IndexTTS](https://huggingface.co/IndexTeam/IndexTTS-1.5) 的 ComfyUI 插件，用于将 `.srt` 字幕文件转换为语音。支持参考音频复刻语音风格，并自动调整语速以匹配字幕时长。适用于视频配音、语音生成等中英文场景。

## ✨ 功能特色

- 🎙️ 支持输入 `.srt` 字幕内容，生成语音输出  
- 🧬 支持上传参考音频，克隆说话人音色  
- ⏱️ 自动语速调整以匹配字幕时间  
- 🌐 支持中英文
- 🔧 无需编程，直接在 ComfyUI 节点中操作
 - ⚙️ 兼容 IndexTTS-2.0（除了原有的 IndexTTS-1.5），可以选择不同模型版本进行合成
 - 💓 支持从“原音频”中自动截取每条字幕对应的“情绪音频段”（仅在 IndexTTS-2.0 时启用），用于提升合成时的情感一致性
 - 🧩 srt_to_voice 接口支持更多可选推理参数（temperature、top_p、top_k、repetition_penalty、seed 等），用户可在节点中微调合成效果
 - 🔍 改进的 SRT 解析：优先尝试解析标准 SRT；若解析失败，会退化为“一行一句”的简易解析，提升兼容性
 - ⏱️ 文本长度自适应生成长度（避免短句生成过长音频），并自动按字幕时间插入静音以对齐时序

---

## 🛠️ 安装步骤

### 1. 如果还没有安装comfyui，先下载安装

comfyui下载地址：

[https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

### 2. 克隆插件代码

进入 ComfyUI 的 custom_nodes 目录，执行以下命令克隆插件代码：

`cd ComfyUI/custom_nodes`

`git clone https://github.com/bluewing82/ComfyUI-Srt2Voice`

### 3. 安装 pynini

查看 Python 版本：

`..\..\python_embeded\python.exe --version`

从项目的 `pynini安装文件` 文件夹选择对应的 pynini 安装包进行安装：

python 3.10 : `..\..\python_embeded\python.exe -m pip install ".\ComfyUI-Srt2Voice\pynini_files\pynini-2.1.6.post1-cp310-cp310-win_amd64.whl"`

python 3.11 : `..\..\python_embeded\python.exe -m pip install ".\ComfyUI-Srt2Voice\pynini_files\pynini-2.1.6.post1-cp311-cp311-win_amd64.whl"`

python 3.12 : `..\..\python_embeded\python.exe -m pip install ".\ComfyUI-Srt2Voice\pynini_files\pynini-2.1.6.post1-cp312-cp312-win_amd64.whl"`

python 3.13 : `..\..\python_embeded\python.exe -m pip install ".\ComfyUI-Srt2Voice\pynini_files\pynini-2.1.6.post1-cp313-cp313-win_amd64.whl"`

安装 WeTextProcessing：

`..\..\python_embeded\python.exe -m pip install WeTextProcessing --no-deps`

### 4. 安装环境依赖

安装插件所需的依赖：

`..\..\python_embeded\python.exe -m pip install -r .\ComfyUI-Srt2Voice\requirements.txt`

### 5. 下载 IndexTTS 模型

从 Hugging Face 下载 IndexTTS 模型文件，链接：

[https://huggingface.co/IndexTeam/IndexTTS-1.5/tree/main](https://huggingface.co/IndexTeam/IndexTTS-1.5/tree/main)

将模型文件全部放到这个目录下面（如果该目录不存在，请手动创建）：

`ComfyUI/models/IndexTTS-1.5/`

---

## 🚀 使用方法

启动 ComfyUI，添加节点 **SRT to Voice**。

配置参数说明：

- `srt_text`：输入 `.srt` 文件中的字幕内容（纯文本）  
- `reference_audio`：上传参考音频，用于克隆说话人的音色

可选参数与说明：

- `model_version`：选择 TTS 模型版本（可选：`IndexTTS-1.5` 或 `IndexTTS-2.0`，默认 `IndexTTS-1.5`）。
- `original_audio`：（可选）上传原始音频文件，只有在使用 `IndexTTS-2.0` 时会尝试按字幕时间自动截取情绪音频段并作为 emo_prompt 提供给模型，从而提升情绪一致性。
- `temperature` / `top_p` / `top_k` / `repetition_penalty`：推理采样控制参数（节点中可调），用于微调合成风格与多样性。
- `seed`：随机种子（-1 表示不固定），用于可重复性控制。

行为细节（实现说明）：

- 当提供标准 SRT 文本时，插件会优先解析时间戳并按时序合成；若解析失败会退化为逐行文本处理。
- 对于短句子，插件会自动限制生成长度（通过动态 max_mel_tokens），避免生成比预期更长的音频片段。
- 若使用 `IndexTTS-2.0` 且提供 `original_audio`，插件会在每条字幕对应的时段内截取音频片段并传给模型以作为“情绪音频提示（emo_audio_prompt）”。

运行工作流，即可生成对应的语音输出。

---

## 🙏 鸣谢

本项目基于以下开源项目构建：

- **IndexTTS**：高质量文本转语音模型  
- **ComfyUI**：强大的可视化工作流平台  

特别感谢 IndexTTS 项目的开源贡献！

---

## 📄 许可证

本插件遵循 MIT 许可证，详情请见 LICENSE 文件。

⚠️ 本项目使用了 IndexTTS，该项目同样使用 MIT 许可证。请确保在使用本插件过程中遵守其许可证要求。

---

如需进一步帮助或反馈建议，欢迎提交 Issue！

## ⚠️ 兼容性及测试说明

- 参考 https://github.com/idiap/coqui-ai-TTS/commit/8555c60b21f2665a6f64f9f165153dd62c7639c1 ，该提交为 coqui-ai-TTS 引入了对更高版本 transformers 的支持。
- 本插件在 transformers 4.56.2 下进行了测试；注意在 IndexTTS-2.0 下生成的声音在某些场景中与参考音频并不完全一致（即“声音和参考声音不一样”），如遇到问题请提交 Issue 并附上复现示例。
