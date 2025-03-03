# 一、YOLO框架讲解

YOLO框架可以分为以下几个主要部分：

### 1. YOLO框架概述
- **背景与目标**：介绍YOLO（You Only Look Once）系列算法的起源、目标以及其在目标检测领域的定位。
- **核心思想**：强调YOLO将目标检测视为一个回归问题，通过单个卷积神经网络直接从图像中预测边界框和类别概率。

### 2. 网络架构设计
- **整体架构**：YOLO网络通常由Backbone（特征提取部分）、Neck（特征融合部分）和Head（预测部分）组成。

- **Backbone**：负责从输入图像中提取特征，常见的Backbone包括Darknet-53、CSPDarknet53等。

- **Neck**：用于细化和组合不同层次的特征，例如YOLOv4中的SPP（空间金字塔池化）和PAN（路径聚合网络）。

- **Head**：负责最终的预测，输出边界框、类别概率和置信度。

- ![主干网络篇 | YOLOv8更换主干网络之ShuffleNetV2（包括完整代码+添加步骤+网络结构图）](https://img-blog.csdnimg.cn/direct/14146aad294a481b8b570fe76dca815d.png)

  ![img](https://i-blog.csdnimg.cn/blog_migrate/1ebb040adac1e258c00d6e0868727559.jpeg)

### 3. 关键技术与改进
- **版本演进**：从YOLOv1到YOLOv11，每一代模型在架构、性能和效率上的改进。

- **优化技术**：例如Leaky ReLU激活函数、Dropout正则化、数据增强等。

- **多尺度预测**：通过不同尺度的特征图进行预测，以更好地检测不同大小的目标。

  ![img](https://i0.hdslb.com/bfs/new_dyn/watermark/7a9a9265ac13f3bfeeb19e9544faeae0297582922.png)

### 4. 训练与推理流程
- **数据预处理**：包括图像尺寸调整、格式转换、数据增强等。
- **训练过程**：介绍损失函数的计算（如位置损失、置信度损失）以及训练参数的设置。
- **推理过程**：从模型加载到结果输出的完整流程，包括数据预处理、模型推理和结果后处理。

### 5. 性能评估与优化
- **性能指标**：如平均精度（mAP）、每秒帧数（FPS）等。
- **优化策略**：包括模型剪枝、量化、GPU加速等。

### 6. 应用场景与案例
- **目标检测**：用于物体定位和分类。
- **图像分割**：结合分割任务的扩展。
- **姿态估计**：人体关键点检测与行为分析。



# 二、模型部署（Android 或者 RK3588）

模型部署是将训练好的机器学习模型应用于实际生产环境的过程。为了确保模型能够高效、稳定地运行，模型部署通常可以分为以下几个主要部分：

### 1. **模型准备**
   - **模型选择与优化**：
     - 确定适合部署的模型版本（例如，是否需要轻量化、是否需要量化等）。
     - 对模型进行优化，如剪枝、量化、蒸馏等，以提高推理速度和降低资源消耗。
   - **模型转换**：
     - 将训练好的模型转换为适合部署的格式（如ONNX、TensorRT、OpenVINO等）。
     - 这里选择ncnn，将权重转换为 NCNN 格式
     - 确保模型在目标硬件上的兼容性。

### 2. **硬件与环境准备**
   - **硬件选择**：
     - 根据模型的计算需求选择合适的硬件平台，如CPU、GPU、FPGA、ASIC或边缘设备。
     - 这里可以选择自己的手机。
     
   - **软件环境搭建**：
     
     - 安装操作系统、深度学习框架（如TensorFlow、PyTorch）和推理引擎（如TensorRT、OpenVINO）。
     
     - 配置依赖库和环境变量。
     
  - **开发环境搭建**：
  - Android Studio安装
  - Android SDK选择手机的Android版本（通常是最新）
  - Gradle jdk选择jdk17或更新版本
  - 将 NCNN 格式权重放入文件夹中
    放置best.bin并best.param放入文件夹中app\src\main\assets\

### 3. **推理设计和模型剪裁（涉及到代码修改）**
   - **推理引擎选择**：
     - 根据硬件和模型选择合适的推理引擎，如TensorFlow Serving、TorchScript、ONNX Runtime等。
   - **将权重转换为ONNX格式**：
     - 修改ultralytics/ultralytics/nn/modules.py如下内容：
     ```python
     class C2f(nn.Module):

    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
     
    def forward(self, x):
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))
     
        print("ook")
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1)) ```
        - 修改ultralytics/ultralytics/nn/head.py
        ```python 
        def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # y = torch.cat((dbox, cls.sigmoid()), 1)
        # return y if self.export else (y, x)
     
        print("ook")
        return torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).permute(0, 2, 1) ```


​        
   - **修改项目代码**：
     - 修改yolo.cpp
		根据您的自定义数据集修改app\src\main\jni\yolo.cpp's 。num_class
     - 根据您的自定义数据集修改app\src\main\jni\yolo.cpp's 。class_names
     - 根据你的app\src\main\jni\yolo.cpp情况修改。layer_namebest.param
     - 修改app\src\main\jni\yolo.cpp的weights name↳

   - **虚拟手机运行程序**：
     
     - 效果如下就成功了
     
       ![image-20250226181702815](C:\Users\zhj\AppData\Roaming\Typora\typora-user-images\image-20250226181702815.png)
     
       **性能优化**：
     
       - 优化推理流程，如批处理、多线程、异步处理等。
       - 进行性能测试，确保模型在实际场景中的响应时间和吞吐量满足要求。
     

### 4. **数据预处理与后处理**
   - **数据预处理**：
     - 将输入数据转换为模型所需的格式，如归一化、裁剪、缩放等。
     - 确保预处理流程与训练时一致。
   - **数据后处理**：
     - 对模型输出进行处理，如解码、阈值过滤、非极大值抑制（NMS）等。
     - 将模型输出转换为用户可理解的结果。

### 5. **部署与测试**
   - **部署方式选择**：
     - 选择部署方式，如本地服务器、云平台、边缘设备，这里选择Android**部署。
     
   - **配置**[Andriod](https://so.csdn.net/so/search?q=Andriod&spm=1001.2101.3001.7020) Studio
     
       ##### 1. 安装Google USB Driver
     
       ​	在Android Studio主界面中，点击SDK Manager，进入Android SDK管理界面
     ​     
       ​	在Android SDK界面下，选择SDK Tools，然后勾选Google USB Driver，再点击OK
     ​     
     
     ##### 2. 安装与手机对应Android平台
     
   - **将手机用数据线连接电脑**：
     
        - 安装手机驱动程序
        - 按下【Win+X】组合键呼出快捷菜单，点击【设备管理器】；
        - 进入电脑的设备管理器界面，并点开便携设备，找到你的手机图标
        - 3.点击你的手机图标，进入属性界面。
        - 4.点击浏览我的电脑以查找驱动程序
          右击你的手机——>更新驱动程序——>浏览我的计算机以查找驱动程序软件
          选择刚才下载的usb driver 的下载路径，一般自动下载到你sdk安装包sdk\extras\google\usb_driver
          一直下一步到完成就可以了
      
   - **配置手机设置**：
     
     - 进入开发者模式
     打开设置——>关于手机，疯狂点击版本号，直到出现提示你正在开发者模式，返回上层菜单进入开发者选项，把usb调试打开
        - 运行程序
     在Android Studio 上选择你的手机点击run，然后手机会自动提示你，安装程序。安装完成后，即可运行。
     
   - **测试与验证**：
     - 进行功能测试，确保模型输出正确。
     - 进行性能测试，确保服务的响应时间和吞吐量满足要求。
     - 进行压力测试，确保系统在高负载下的稳定性。



# 三、模型量化

模型量化是将高精度的浮点数（如float32）转换为低精度的整数（如int8）的过程，旨在减少模型的存储需求和计算量，同时尽量保持模型性能。模型量化可以分为以下几个主要部分：

### 1. **量化分类**
根据量化实施的阶段和方式，模型量化通常分为以下几类：
- **动态量化（Dynamic Quantization）**：在模型推理时动态计算量化参数，不需要额外的校准步骤。

- **静态量化（Static Quantization）**：在模型推理前静态计算量化参数，需要校准步骤。

- **量化感知训练（Quantization-Aware Training, QAT）**：在训练过程中模拟量化效果，通过引入量化噪声来训练模型，使模型能够在训练时就适应量化带来的变化。

- **训练后量化（Post Training Quantization, PTQ）**：在模型训练完成后，使用少量校准数据对模型参数进行量化，适用于追求高易用性和缺乏训练资源的场景。

  **方法选择**：通常，建议对 RNN 和基于 Transformer 的模型使用动态量化，对 CNN 模型使用静态量化

### 2. **量化方法**
根据量化参数的计算方式和范围，量化方法可以分为：
- **对称量化（Symmetric Quantization）**：量化范围是对称的，零点为0，只计算缩放因子（Scale）。适用于数据分布较为对称的情况。
- **非对称量化（Asymmetric Quantization）**：量化范围不对称，需要计算缩放因子和零点（Zero Point），能够更灵活地适应数据的实际分布。

### 3. **量化粒度**
根据量化参数的共享范围，量化粒度可以分为：
- **逐层量化（Layer-wise Quantization）**：整个层的所有权重共享同一组缩放因子和偏移量。
- **逐组量化（Group-wise Quantization）**：每个权重组有自己的一组缩放因子和偏移量。
- **逐通道量化（Channel-wise Quantization）**：每个通道的权重都有单独的缩放因子和偏移量，适用于卷积神经网络。

### 4. **量化步骤**
模型量化的过程通常包括以下步骤：
1. **权重量化**：在模型训练完成后，将权重从浮点数转换为低精度整数，并计算对应的缩放因子和零点。
2. **激活值量化**：在推理时，输入数据也会被量化为低精度整数，激活值的量化需要输入样本数据来计算范围。
3. **整数运算**：推理计算完全在量化后的整数级别上进行，中间操作（如加法、乘法）基于整数。
4. **反量化（可选）**：在推理的最终阶段，输出结果会被反量化为浮点数，以便进行后续处理。

### 5. **量化框架与工具**
目前，多种深度学习框架和工具支持模型量化，例如：
- **TensorRT**：NVIDIA提供的高效推理优化工具，支持多种量化方式。
- **ONNX Runtime**：支持模型优化和量化，适用于跨平台部署。
- **OpenVINO**：Intel提供的推理优化工具，适用于边缘设备。

### 6. **量化实践**
#### - 准备环境

首先，确保已安装ONNX Runtime和必要的Python库，包括ONNX和onnxconverter-common等。可以使用pip命令安装：

```pip
pip install onnx onnxruntime onnxconverter-common
```

#### 导出YOLOv8模型为ONNX格式

使用[PyTorch](https://cloud.baidu.com/product/wenxinworkshop)等工具训练好的YOLOv8模型，可以通过TorchScript或PyTorch自带的导出工具转换为ONNX格式。确保Opset版本不低于10，因为低于10的版本可能不支持量化操作。

#### 进行模型量化

使用ONNX Runtime提供的量化工具进行模型量化。以下是一个使用动态量化将YOLOv8模型转换为INT8格式的示例代码：

```python 
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
# 加载FP32模型
model_fp32 = 'yolov8_fp32.onnx'
model_quant = 'yolov8_int8.onnx'
# 进行动态量化
quantized_model = quantize_dynamic(model_fp32, model_quant, quantization_type=QuantType.QInt8)
# 保存量化后的模型
onnx.save_model(quantized_model, model_quant)
```

### INT8 量化模型转换

ONNX Runtime 提供了量化工具 `onnxruntime.quantization`，可以将 FP32 模型量化为 INT8 模型。

```python
import onnx
from onnxruntime.quantization import QuantizationMode, quantize_static

# 模型路径
fp32_model_path = 'yolov8n.onnx'
int8_model_path = 'yolov8n_int8.onnx'

# 校准数据集（需要提供一组代表性的输入数据）
calibration_data_path = 'calibration_data.npy'

# 加载 FP32 模型
model = onnx.load(fp32_model_path)

# 量化配置
quantization_mode = QuantizationMode.INT8

# 量化模型
quantize_static(
    model_input=fp32_model_path,
    model_output=int8_model_path,
    mode=quantization_mode,
    calibration_data=calibration_data_path,
    per_channel=False,
    optimize_model=True
)
```

**校准数据**：

- 量化过程中需要提供一组代表性的输入数据（`calibration_data.npy`），用于计算量化参数。

- 你可以使用以下代码生成校准数据：

  ```python
  import numpy as np
  calibration_data = np.random.rand(10, 3, 640, 640).astype(np.float32)
  np.save('calibration_data.npy', calibration_data)
  ```

#### 验证量化模型

使用测试数据集验证量化后的模型精度和性能。比较FP32模型和INT8模型在相同数据集上的表现，确保量化后的模型精度损失在可接受范围内。



# 四、其他模型优化

模型优化是一个广泛的概念，涵盖了从训练到部署的多个环节，目的是提高模型的性能、效率和可扩展性。除了模型量化之外，模型优化还可以分为以下几个主要部分：

### 1. **模型结构优化**
   - **网络架构设计**：
     
     - 选择或设计高效的网络架构，如MobileNet、EfficientNet、ShuffleNet等，这些架构通过轻量化设计减少计算量。
     - 使用深度可分离卷积（Depthwise Separable Convolution）或注意力机制（Attention Mechanism）来提高效率。
     
   - **剪枝（Pruning）** **(模型稀疏化)**：
     
     - 去除神经网络中不重要的权重或神经元，减少模型的参数量和计算复杂度。
     
     - 剪枝可以是结构化的（如剪掉整个通道）或非结构化的（剪掉单个权重）。
     
        **剪枝步骤概述**：
     
       1. **安装`ultralytics`、`torch`、`torch_pruning`等必要的库**。
       
       2. **加载模型**：使用`ultralytics`库加载 YOLOv8n 模型。
       
       3. **模型剪枝**：使用`torch_pruning`库对模型进行剪枝操作。
       
       4. **导出为 ONNX**：将剪枝后的模型导出为 ONNX 格式。
       
       5. 示例代码：
       
          ```python
          import torch
          import torch_pruning as tp
          from ultralytics import YOLO
          
          # 步骤1: 安装必要的库
          # 如果你还没有安装这些库，可以使用以下命令进行安装
          # pip install ultralytics torch torch_pruning
          
          # 步骤2: 加载模型
          model = YOLO('yolov8n.pt')  # 从.pt文件加载模型
          model = model.model  # 获取底层的PyTorch模型
          
          # 步骤3: 模型剪枝
          # 初始化剪枝策略
          strategy = tp.strategy.L1Strategy()  # 基于L1范数的剪枝策略
          
          # 计算剪枝的比例（这里设置为0.2，表示剪掉20%的通道）
          pruning_ratio = 0.2
          
          # 构建剪枝工具
          DG = tp.DependencyGraph()
          DG.build_dependency(model, example_inputs=torch.randn(1, 3, 640, 640))
          
          # 定义需要剪枝的层
          ignored_layers = []
          for m in model.modules():
              if isinstance(m, torch.nn.BatchNorm2d):
                  ignored_layers.append(m)
          
          # 获取需要剪枝的通道
          pruning_idxs = strategy(model.conv1.weight, amount=pruning_ratio)  # 以第一个卷积层为例
          pruning_plan = DG.get_pruning_plan(model.conv1, tp.prune_conv, idxs=pruning_idxs)
          
          # 执行剪枝
          pruning_plan.exec()
          
          # 步骤4: 导出为ONNX
          input_shape = (1, 3, 640, 640)
          dummy_input = torch.randn(input_shape)
          
          # 导出剪枝后的模型为ONNX格式
          torch.onnx.export(
              model,
              dummy_input,
              "yolov8n_pruned.onnx",
              export_params=True,
              opset_version=11,
              do_constant_folding=True,
              input_names=['input'],
              output_names=['output'],
              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
          )
          
          print("剪枝后的模型已导出为 yolov8n_pruned.onnx")
          ```
       
          - 初始化剪枝策略`tp.strategy.L1Strategy()`，这里使用基于 L1 范数的剪枝策略。
          - 构建依赖图`tp.DependencyGraph()`，用于分析模型的层间依赖关系。
          - 定义需要剪枝的层，这里以第一个卷积层`model.conv1`为例，计算需要剪枝的通道索引。
          - 生成剪枝计划`pruning_plan`并执行剪枝操作
     
   - **知识蒸馏（Knowledge Distillation）**：
     
     - 将复杂模型（教师模型）的知识迁移到轻量化模型（学生模型），以提高轻量化模型的性能。
     
     - 通过软目标（Soft Targets）或中间特征蒸馏来保留教师模型的性能。
     
       **步骤概述**:
     
       1. **安装`ultralytics`库**。
     
       2. **准备教师模型和学生模型**：选择一个性能较好的 YOLOv8 模型比如x作为教师模型，一个较小的 YOLOv8 模型作为学生模型n。
     
       3. **配置数据**：准备好用于训练的数据集。
     
       4. **定义蒸馏损失函数**：结合学生模型的预测损失和学生模型与教师模型预测结果之间的蒸馏损失。
     
       5. **训练学生模型**：使用定义好的损失函数对学生模型进行训练。
     
       6. **代码实例：**
     
          ```python
          from ultralytics import YOLO
          import torch
          import torch.nn as nn
          
          # 步骤1: 安装必要的库
          # 如果你还没有安装ultralytics库，可以使用以下命令进行安装
          # pip install ultralytics
          
          # 步骤2: 准备教师模型和学生模型
          teacher_model = YOLO('yolov8x.pt')  # 使用较大的YOLOv8x作为教师模型
          student_model = YOLO('yolov8n.pt')  # 使用较小的YOLOv8n作为学生模型
          
          # 步骤3: 配置数据
          data = 'path/to/your/data.yaml'  # 替换为你的数据集配置文件路径
          
          # 步骤4: 定义蒸馏损失函数
          class DistillationLoss(nn.Module):
              def __init__(self, alpha=0.5, temperature=4):
                  super(DistillationLoss, self).__init__()
                  self.alpha = alpha
                  self.temperature = temperature
                  self.ce_loss = nn.CrossEntropyLoss()
                  self.kl_loss = nn.KLDivLoss(reduction='batchmean')
          
              def forward(self, student_output, teacher_output, labels):
                  # 学生模型的预测损失
                  student_loss = self.ce_loss(student_output / self.temperature, labels)
          
                  # 学生模型与教师模型预测结果之间的蒸馏损失
                  distillation_loss = self.kl_loss(
                      nn.functional.log_softmax(student_output / self.temperature, dim=1),
                      nn.functional.softmax(teacher_output / self.temperature, dim=1)
                  ) * (self.temperature ** 2)
          
                  # 组合损失
                  total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                  return total_loss
          
          distillation_loss = DistillationLoss()
          
          # 步骤5: 训练学生模型
          epochs = 10  # 训练的轮数
          batch_size = 16  # 批量大小
          
          # 训练循环
          for epoch in range(epochs):
              for batch in student_model.loader(data, batch_size=batch_size):
                  images, labels = batch
          
                  # 教师模型的预测
                  with torch.no_grad():
                      teacher_output = teacher_model(images)[0].probs
          
                  # 学生模型的预测
                  student_output = student_model(images)[0].probs
          
                  # 计算蒸馏损失
                  loss = distillation_loss(student_output, teacher_output, labels)
          
                  # 反向传播和优化
                  student_model.model.optimizer.zero_grad()
                  loss.backward()
                  student_model.model.optimizer.step()
          
              print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
          
          # 保存训练好的学生模型
          student_model.export(format='pt')
          ```
     
          

### 2. **训练过程优化**
   - **优化算法选择**：
     - 使用高效的优化算法（如Adam、SGD with Momentum、AdaGrad等）以加速模型收敛。
     - 调整学习率策略（如学习率衰减、Warm-up策略）以提高训练效果。
   - **正则化技术**：
     - 使用Dropout、L2正则化、Batch Normalization等技术防止过拟合。
   - **数据增强与预处理**：
     - 通过数据增强（如旋转、裁剪、翻转、颜色变换等）增加模型的泛化能力。
     - 优化数据预处理流程以提高训练效率。

### 3. **推理优化**
   - **模型量化**：
     - 将浮点数模型转换为低精度整数模型，减少存储需求和计算量。
   - **推理引擎优化**：
     - 使用高效的推理引擎（如TensorRT、ONNX Runtime、OpenVINO）来加速模型推理。
     - 优化推理流程，如批处理、多线程、异步推理等。
     - 1.读图 2.推理 3.后处理 4. 图片显示
   - **硬件加速**：
     - 利用GPU、TPU、FPGA等硬件加速推理过程。
     - 针对特定硬件平台进行优化（如使用CUDA、OpenCL）。

### 4. **模型压缩**
   - **参数共享**：
     - 在某些模型中，通过参数共享减少模型的存储需求。
   - **矩阵分解**：
     - 使用奇异值分解（SVD）等技术对权重矩阵进行分解，减少参数量。
   - **稀疏化**：
     - 将模型的权重稀疏化，减少非零元素的数量，从而提高计算效率。

### 5. **混合精度训练**
   - **混合精度技术**：
     - 在训练过程中同时使用浮点数（如float32）和低精度格式（如float16）以加速训练并减少内存占用。
   - **自动混合精度（AMP）**：
     - 使用框架提供的工具（如NVIDIA的AMP）自动选择合适的精度进行训练。

### 6. **模型部署优化**
   - **服务架构优化**：
     - 使用容器化（如Docker）、微服务化（如Kubernetes）部署模型，提高可扩展性和灵活性。
   - **缓存机制**：
     - 对常见请求的结果进行缓存，减少重复计算。
   - **负载均衡**：
     - 分散请求到多个服务实例，提高系统的吞吐量和稳定性。
