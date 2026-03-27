# MultiVision

Multimodal Visual Intelligence Platform - 基于 Django + Hugging Face 的计算机视觉应用

## 功能特性

- **目标检测** - 使用 DETR 模型检测 80 类 COCO 对象
- **手势识别** - 使用 MediaPipe 检测手部关键点和手势
- **风格迁移** - 支持素描、水彩、油画、动漫等多种风格
- **动作识别** - 基于光流法的实时动作分析

## 技术栈

- **后端**: Django 5.0 + Django REST Framework
- **模型**: Hugging Face Transformers + MediaPipe
- **前端**: Django Templates + 原生 JavaScript
- **实时**: WebSocket 支持摄像头实时推理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行迁移

```bash
python manage.py migrate
```

### 3. 启动服务

```bash
python manage.py runserver
```

访问 http://localhost:8000

## 可选依赖

手势识别需要安装 MediaPipe：
```bash
pip install mediapipe
```

## 添加新模型

在 `vision/models_handler.py` 中：

1. 创建新的模型类继承 `BaseVisionModel`
2. 实现 `process()` 方法
3. 在 `load_models()` 中注册

```python
class MyModel(BaseVisionModel):
    def __init__(self):
        super().__init__("my_model")

    def process(self, image, **kwargs):
        # 处理逻辑
        return {"result": "..."}

# 注册
_models["my_model"] = MyModel()
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/models` | GET | 列出可用模型 |
| `/api/infer` | POST | 图片推理 |
| `/api/camera` | POST | 摄像头帧处理 |
| `/health` | GET | 健康检查 |

## 项目结构

```
MultiVision/
├── manage.py              # Django 管理脚本
├── requirements.txt       # 依赖
├── multivision/           # 项目配置
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── vision/                # 主应用
│   ├── models_handler.py  # 模型处理核心
│   ├── views.py           # 视图函数
│   └── urls.py            # URL 路由
└── templates/
    └── vision/
        └── index.html     # 前端页面
```