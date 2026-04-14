# 《Sanders classification of calcaneal fractures in CT images with deep learning and differential data augmentation techniques》算法实现步骤整理

## 1. 文档目的

本文档根据论文 **Sanders classification of calcaneal fractures in CT images with deep learning and differential data augmentation techniques** 的方法部分，整理出一套尽可能贴近论文描述的、可落地执行的算法实现步骤。

这篇论文的核心任务是：

- 输入患者跟骨 CT 图像
- 对图像进行筛选、预处理和增强
- 使用 **PCANet** 完成 **Sanders 分型四分类**
- 输出骨折类别：**Type I / Type II / Type III / Type IV**

> 说明：论文给出了总体流程、主要参数和实验设计，但没有公开完整源码，也没有给出 PCANet 的全部底层超参数。因此，本文档分为两部分：
>
> 1. **论文明确给出的实现步骤**
> 2. **工程落地时需要补充的实现细节**（会明确标注为“工程补充”）

---

## 2. 算法总体流程

论文中的识别算法可以整理为如下主流程：

```text
原始 DICOM CT 数据
→ 2D 图像转换
→ 数据筛选与人工标注
→ 图像预处理
→ 训练集数据增强
→ PCANet 特征提取
→ 四分类识别（Sanders I/II/III/IV）
→ 性能评估
```

如果按工程模块划分，可拆成以下 6 个模块：

1. 数据采集与格式转换
2. 数据筛选与标签建立
3. 图像预处理
4. 数据增强
5. PCANet 训练与分类
6. 测试与评估

---

## 3. 数据准备阶段

## 3.1 原始数据来源

论文中使用的数据来自：

- 台湾 Show Chwan Memorial Hospital
- 共 **14 名患者** 的跟骨 CT DICOM 数据

原始数据特征：

- 每个病例包含多个切片，深度范围约 **314 ~ 1000 slices**
- 原始图像分辨率为：
  - **512 × 512**
  - 或 **912 × 912**
- 包含 3 个观察平面：
  - Axial / Transversal
  - Coronal
  - Sagittal

---

## 3.2 DICOM 转 2D 图像

### 论文描述
论文先将 3D DICOM 数据转换成 2D JPG 图像，再进行筛选与分类。

### 实现步骤

1. 读取每个患者的 DICOM 序列。
2. 提取其中每个 slice 的图像矩阵。
3. 保存为 2D 图像文件（论文中使用的是 `jpg`）。
4. 统一调整图像大小为：
   - **360 × 360 像素**

### 推荐实现方式（工程补充）
可使用：

- `pydicom` 读取 DICOM
- `opencv-python` 或 `PIL` 保存 JPG/PNG

示例流程：

```python
for patient in patients:
    dicom_series = load_dicom_series(patient)
    for i, dcm in enumerate(dicom_series):
        img = dcm.pixel_array
        img = normalize_to_uint8(img)
        img = cv2.resize(img, (360, 360))
        cv2.imwrite(f"{patient}_{i:04d}.jpg", img)
```

---

## 4. 数据筛选与标签构建

## 4.1 筛选规则

论文给出了两个明确的数据筛选条件：

1. **必须是 coronal view（冠状位）**
2. **图中必须出现 calcaneus bone（跟骨）**

也就是说，不是所有切片都会用于训练。只有满足上述条件的图像才会进入后续流程。

---

## 4.2 筛选逻辑

### 论文描述
对于一个病例，如果该病例某个方向有很多切片，并不是所有切片都包含清晰的跟骨区域。例如：

- 如果某位患者的某一视图有 100 张切片
- 真正包含跟骨主体的可能只有中间的约 35 张

因此论文会人工筛掉不包含目标骨结构的切片。

### 实现步骤

1. 只保留 **coronal view** 的切片。
2. 检查每一张切片是否包含跟骨区域。
3. 删除不包含跟骨或跟骨过小、信息不足的切片。
4. 保留有效切片进入标注阶段。

### 工程补充建议
由于论文本身未实现自动分割，这一步大概率是 **人工筛选** 或半自动筛选完成的。

可选实现方式：

- 人工逐张浏览并勾选有效图像
- 编写简单的阈值/面积规则过滤明显无骨组织切片
- 结合医生或标注人员进行审核

---

## 4.3 标签标注

论文采用 **Sanders classification system** 对图像进行四分类标注：

- Type I
- Type II
- Type III
- Type IV

标注依据为：

- 骨折线数量
- 骨折线在 posterior calcaneal facet（跟骨后关节面）的位置

### 实现步骤

1. 由具备医学知识的标注者查看合格的 CT 图像。
2. 根据 Sanders 分类标准给每张图像赋标签。
3. 将图像路径与标签保存成结构化数据，例如 CSV：

```csv
image_path,label
patient01_0032.jpg,Type_I
patient01_0033.jpg,Type_I
patient02_0101.jpg,Type_III
```

### 论文最终筛选结果
从约 **6131 张**切片中，得到 **760 张**有效标注图像：

- Type I: **153**
- Type II: **221**
- Type III: **178**
- Type IV: **208**

---

## 5. 图像预处理模块

这部分是论文算法的重要组成部分。预处理的目标是：

- 降低背景噪声
- 提取主要骨组织区域
- 让输入图像更加稳定
- 提升后续分类效果

论文给出的预处理流程包括：

1. 二值化
2. 腐蚀
3. 膨胀
4. 轮廓检测
5. 将主要区域映射回原图并居中输出

---

## 5.1 输入图像

输入是上一阶段筛选后的 **灰度 CT 图像**。

如果当前图像不是灰度图，应先转灰度：

```python
gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

---

## 5.2 阈值二值化

### 论文参数
- 阈值：**50**

### 实现步骤

1. 对灰度图应用固定阈值。
2. 大于等于 50 的像素设为白色（255）。
3. 小于 50 的像素设为黑色（0）。

```python
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
```

### 作用
- 将骨组织与背景粗略分离
- 为后续形态学操作提供输入

---

## 5.3 形态学腐蚀（Erosion）

### 论文参数
- 腐蚀核大小：**10**

### 实现步骤

1. 构造一个大小约为 `10 × 10` 的结构元素。
2. 对二值图执行腐蚀。

```python
kernel = np.ones((10, 10), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)
```

### 作用
- 去掉小噪声块
- 缩小白色区域
- 去除骨组织内部或外部的小型伪影

论文中描述为：

- 白色区域变细
- 黑色背景区域变大
- 对象变小

---

## 5.4 形态学膨胀（Dilation）

### 论文参数
- 膨胀核大小：**10**

### 实现步骤

1. 使用与腐蚀相同大小的结构元素。
2. 对腐蚀结果执行膨胀。

```python
dilated = cv2.dilate(eroded, kernel, iterations=1)
```

### 作用
- 恢复主要结构的连通性
- 扩大亮区域
- 让主骨组织区域更容易被轮廓检测捕获

---

## 5.5 轮廓检测

### 论文描述
论文使用 OpenCV，并参考 **Suzuki 方法** 做轮廓检测。

### 实现步骤

1. 在膨胀后的图像上寻找轮廓。
2. 计算每个轮廓对应的面积。
3. 选取 **最大白色区域** 作为主要组织区域。

```python
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
main_contour = max(contours, key=cv2.contourArea)
```

### 作用
- 在复杂背景中定位主要骨组织
- 去掉外围无关组织和背景噪声

---

## 5.6 生成主组织掩膜并映射回原图

### 实现步骤

1. 创建与原图同尺寸的黑色 mask。
2. 将主轮廓填充为白色。
3. 用该 mask 从原始灰度图中提取主组织区域。

```python
mask = np.zeros_like(gray)
cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)
roi = cv2.bitwise_and(gray, gray, mask=mask)
```

### 作用
- 保留最重要的骨结构信息
- 去掉无关背景

---

## 5.7 居中放置到黑色背景

### 论文描述
提取出的主组织区域会被放到新的黑色背景中央，生成最终输出图像。

### 实现步骤

1. 根据主轮廓计算 bounding box。
2. 裁剪主组织区域。
3. 创建目标大小的黑底图像。
4. 将裁剪区域贴到中心位置。

```python
x, y, w, h = cv2.boundingRect(main_contour)
crop = roi[y:y+h, x:x+w]
canvas = np.zeros((360, 360), dtype=np.uint8)
start_x = (360 - w) // 2
start_y = (360 - h) // 2
canvas[start_y:start_y+h, start_x:start_x+w] = crop
```

### 最终输出
这张输出图就是后续分类器的输入图像。

---

## 6. 数据增强模块

由于原始医学图像数量有限，论文在训练阶段引入了 **Augmentor** 数据增强流水线。

核心思想是：

- 不改变标签含义
- 人工生成更多训练样本
- 提高模型泛化能力
- 缓解 leave-one-out 条件下训练样本不足的问题

---

## 6.1 使用工具

论文使用：

- **Augmentor**

其特点：

- 支持基于流水线的图像增强
- 每次图像通过管道时可随机触发变换
- 可持续采样直到生成指定数量的新图像

---

## 6.2 增强顺序

论文明确给出增强顺序：

1. **rotation**
2. **distortion**
3. **horizontal flip**

即：

```text
原始图像
→ 旋转
→ 随机形变
→ 水平翻转
→ 得到增强图像
```

---

## 6.3 旋转（Rotation）

### 论文参数
- 角度范围：**-10° 到 +10°**
- 概率：**0.7**

### 实现逻辑
每张图像进入增强管道时：

- 以 0.7 的概率执行旋转
- 旋转角度在 -10 到 +10 度之间随机采样

### Augmentor 风格示例

```python
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
```

---

## 6.4 随机形变（Random Distortion）

### 论文参数
- 宽度方向 distortion：**10**
- 高度方向 distortion：**10**
- 概率：**0.5**

### 实现逻辑
- 以 0.5 的概率对图像施加随机形变
- 形变程度在高和宽两个方向都设置为 10

### 示例

```python
p.random_distortion(probability=0.5, grid_width=10, grid_height=10, magnitude=10)
```

> 说明：论文写的是 “degree of distortion in both height and width are 10”，工程实现时通常会对应到网格或形变量级别参数，具体映射依所用库接口而定。

---

## 6.5 水平翻转（Horizontal Flip）

### 论文参数
- 概率：**0.5**

### 示例

```python
p.flip_left_right(probability=0.5)
```

---

## 6.6 增强样本生成方式

论文的做法不是对每张图像只增强一次，而是：

- 将图像反复送入增强流水线
- 每次随机执行一组变换
- 一直生成到指定样本数量为止

### 论文设置的增强数据规模

| 实验集 | 总图像数 | 每类图像数 |
|---|---:|---:|
| Baseline | 760 | 不均衡原始数据 |
| Augmented A | 2000 | 500 |
| Augmented B | 4000 | 1000 |
| Augmented C | 8000 | 2000 |
| Augmented D | 10000 | 2500 |
| Augmented E | 20000 | 5000 |

### 实现思路

1. 将每类原始图像分别组织到对应文件夹。
2. 对每一类单独进行增强。
3. 为每一类生成目标数量的新图像，使类别分布均衡。

---

## 7. PCANet 分类模块

这是论文中真正执行“识别”的核心模块。

---

## 7.1 为什么选 PCANet

论文指出，PCANet 的优势是：

- 结构简单
- 运算代价低
- 适合图像分类
- 基本组件容易实现

PCANet 并不是标准意义上依赖大规模反向传播训练的深层 CNN，而是使用较为简单的统计学习方式构造卷积滤波器。

---

## 7.2 PCANet 的三层核心处理逻辑

论文明确给出 PCANet 的三个核心步骤：

1. **PCA filters**
2. **binary hashing**
3. **block-wise histograms**

可将其理解为：

```text
输入图像
→ PCA卷积滤波
→ 二值化编码
→ 分块直方图池化
→ 分类特征向量
→ 输出类别
```

---

## 7.3 Step 1：PCA 滤波器学习

### 原理
PCANet 不直接学习传统卷积核，而是：

1. 从训练图像中提取局部 patch
2. 对 patch 集合做 PCA
3. 取主成分作为卷积滤波器

### 工程实现步骤（补充）

1. 收集所有训练图像。
2. 从每张图像中提取固定大小的小块 patch。
3. 将 patch 展平后组成矩阵。
4. 对矩阵做去均值。
5. 执行 PCA。
6. 选取前 `k` 个主成分作为滤波器。

伪代码：

```python
patches = collect_patches(train_images, patch_size)
patches = patches - patches.mean(axis=0)
pca = PCA(n_components=k)
pca.fit(patches)
filters = pca.components_.reshape(k, patch_h, patch_w)
```

> 注意：论文没有明确写出 patch 大小、每层滤波器数、PCANet stage 数量细节，但图示说明其使用的是 **two-stage PCANet**。

---

## 7.4 Step 2：卷积响应计算

### 实现逻辑
将每个 PCA 滤波器与输入图像卷积，得到多个响应图。

```python
feature_maps = [cv2.filter2D(img, -1, f) for f in filters]
```

若为两级 PCANet，则第二级的输入通常是第一级输出的特征图，再重复：

- patch 提取
- PCA 学习
- 卷积生成响应图

---

## 7.5 Step 3：Binary Hashing

### 论文描述
非线性层采用最简单的 **binary quantization / hashing**。

### 实现逻辑
对于同一位置上的多个响应图：

- 若响应值 > 0，则记为 1
- 否则记为 0

然后把多个二值结果按位组合成整数编码。

例如有 8 个滤波器：

```text
[1, 0, 1, 1, 0, 0, 1, 0] -> 二进制编码 -> 十进制整数
```

伪代码：

```python
binary_maps = [(fm > 0).astype(np.uint8) for fm in feature_maps]
hash_code = np.zeros_like(binary_maps[0], dtype=np.uint8)
for i, bm in enumerate(binary_maps):
    hash_code += bm << i
```

---

## 7.6 Step 4：Block-wise Histogram Pooling

### 论文描述
PCANet 最终使用 **block-wise histograms of the binary codes** 作为输出特征。

### 实现步骤

1. 将编码图划分成多个小块（blocks）。
2. 对每个 block 统计哈希值直方图。
3. 将所有 block 的直方图拼接成最终特征向量。

伪代码：

```python
features = []
for block in split_into_blocks(hash_code, block_size):
    hist, _ = np.histogram(block, bins=256, range=(0, 256))
    features.extend(hist)
feature_vector = np.array(features, dtype=np.float32)
```

### 作用
- 保留局部空间分布信息
- 将图像映射为可供分类的固定长度特征向量

---

## 7.7 Step 5：分类输出

论文在摘要和实验部分重点描述的是分类准确率，没有展开说明最后一层分类器使用的具体形式。

### 工程补充建议
PCANet 的经典实现中，常见做法是：

- 使用 **SVM**
- 或简单线性分类器
- 或最近邻分类器

如果希望尽量贴近 PCANet 文献传统实现，可优先尝试：

- `LinearSVC`
- `SVM(kernel='linear')`

示例：

```python
clf = LinearSVC()
clf.fit(train_features, train_labels)
pred = clf.predict(test_features)
```

> 注意：该分类器类型并未在本文论文正文中明确写出，因此这里属于工程补充，不应当当作论文原文结论。

---

## 8. 训练策略与实验设置

论文做了两类实验。

---

## 8.1 实验 1：随机划分训练集/测试集

### 目的
验证模型是否能对 Sanders 四分类进行有效识别。

### 做法
- 从所有患者图像中随机抽样
- 训练集和测试集随机划分
- 不限制同一患者样本同时出现在训练集和测试集

### 结论
- 准确率约 **92% ~ 95%**

### 解读
这个设置更容易得到高分，因为训练和测试数据分布更接近，甚至可能共享同一患者的相似样本。

---

## 8.2 实验 2：Leave-One-Out

### 目的
验证模型面对“未见过的新患者”时的泛化能力。

### 做法
- 每次留出一个患者的全部样本作为测试集
- 剩余患者样本作为训练集
- 测试集患者不出现在训练集中

### 意义
这更接近真实临床使用场景。

### 结果

#### 不使用增强图像
- 平均准确率约 **35%**

#### 使用增强图像
- 平均准确率提升到约 **62% ~ 72%**
- 当增强样本量从 **2000** 提高到 **10000** 时
- 准确率约从 **62%** 提高到 **72%**

---

## 9. 性能评估方式

论文采用的指标是 **Accuracy（准确率）**。

公式为：

```text
accuracy = n_correct / n
```

其中：

- `n_correct`：正确分类样本数
- `n`：总测试样本数

工程实现：

```python
accuracy = (pred == y_true).sum() / len(y_true)
```

---

## 10. 按工程实现整理的完整步骤清单

下面给出一份可直接照着做的落地步骤。

## Step 1：读取原始 CT 数据

- 收集 14 名患者的 DICOM 序列
- 按患者分文件夹存放

## Step 2：转换为 2D 图像

- 遍历 DICOM slice
- 提取像素矩阵
- 转成灰度 JPG
- resize 到 `360 × 360`

## Step 3：筛选有效样本

- 仅保留 coronal view
- 去除不含跟骨的切片
- 保留跟骨结构清晰、可用于分型的切片

## Step 4：标注 Sanders 类型

- 人工根据骨折线数量和位置标注为 I / II / III / IV
- 保存标签文件

## Step 5：执行预处理

对每张图像：

1. 灰度读取
2. 固定阈值 50 二值化
3. `10×10` 腐蚀
4. `10×10` 膨胀
5. OpenCV 轮廓检测
6. 取最大轮廓
7. 生成 mask
8. 将主组织映射回原图
9. 放到黑底中央
10. 保存预处理结果

## Step 6：构建训练/测试集

### 方案 A：随机划分
- 用于复现实验 1

### 方案 B：leave-one-out
- 用于复现实验 2
- 每次留一位患者作为测试集

## Step 7：生成增强训练集

在训练集上使用 Augmentor：

1. rotation，`prob=0.7`，角度 `[-10, +10]`
2. random distortion，`prob=0.5`，height/width = 10
3. horizontal flip，`prob=0.5`
4. 持续采样直到生成目标数量图像

## Step 8：训练 PCANet

- 使用训练图像学习 PCA 滤波器
- 构建 two-stage PCANet
- 对每张图像提取特征：
  - PCA filters
  - binary hashing
  - block-wise histograms

## Step 9：训练分类器

- 将 PCANet 输出特征向量输入分类器
- 输出四分类结果

## Step 10：测试与统计准确率

- 在测试集上预测 Sanders 类型
- 统计 accuracy
- 对比：
  - 原始训练集
  - 增强训练集
  - 不同增强规模

---

## 11. 一份更接近代码执行顺序的伪代码

```python
# 1. 读取并转换数据
images = load_dicom_and_convert_to_jpg(dicom_root)
images = resize_all(images, (360, 360))

# 2. 筛选 coronal view 且包含 calcaneus 的切片
selected = select_coronal_calcaneus_images(images)

# 3. 人工标注 Sanders 类型
labeled_data = manual_label(selected)

# 4. 图像预处理
processed = []
for img, label, patient_id in labeled_data:
    binary = threshold(img, thresh=50)
    eroded = erosion(binary, kernel_size=10)
    dilated = dilation(eroded, kernel_size=10)
    contour = largest_contour(dilated)
    roi = mask_original_image(img, contour)
    centered = center_to_black_canvas(roi, size=(360, 360))
    processed.append((centered, label, patient_id))

# 5. 划分训练/测试集
train_set, test_set = leave_one_out_split(processed, patient_id=holdout_id)

# 6. 数据增强（仅训练集）
augmented_train = augment_with_pipeline(
    train_set,
    rotate_prob=0.7,
    rotate_range=(-10, 10),
    distortion_prob=0.5,
    distortion_degree=(10, 10),
    hflip_prob=0.5,
    target_size=10000
)

# 7. PCANet 特征提取
train_features = pcanet_extract_features(augmented_train.images)
test_features = pcanet_extract_features(test_set.images)

# 8. 分类器训练与预测
clf = train_classifier(train_features, augmented_train.labels)
pred = clf.predict(test_features)

# 9. 计算准确率
acc = compute_accuracy(pred, test_set.labels)
print(acc)
```

---

## 12. 论文中未完全公开但实现时必须明确的点

下面这些点是论文没有给全、但你真正复现时必须补齐的：

1. **PCANet 的 patch size**
2. **每个 stage 的滤波器个数**
3. **block-wise histogram 的 block 大小**
4. **最终分类器类型**
5. **训练集/测试集每次划分的随机种子**
6. **图像归一化方式**（仅 resize 还是还做强度归一化）
7. **DICOM 到 JPG 的窗宽窗位处理**

因此，如果你的目标是“严格复现论文数值结果”，需要进一步：

- 查论文补充材料
- 查作者是否公开代码
- 参考被引用的 PCANet 原论文实现

---

## 13. 可直接复现的最小实现建议

如果你的目标不是逐点完全复现实验，而是先做出一版能运行的实现，建议采用下面的最小版本：

### 数据处理
- DICOM -> PNG/JPG
- resize 到 `360×360`
- threshold=50
- erosion/dilation kernel=`10×10`
- 最大轮廓提取
- 居中到黑背景

### 数据增强
- rotation: ±10°, p=0.7
- random distortion: p=0.5
- horizontal flip: p=0.5

### 特征与分类
- 使用两级 PCANet
- 最后接线性 SVM

### 评估
- 先做随机划分
- 再做 leave-one-out
- 记录 accuracy

---

## 14. 总结

这篇论文的算法本质上不是一个特别复杂的端到端深度 CNN，而是一套由以下部分组成的识别流程：

1. **基于规则的图像筛选**
2. **基于形态学和轮廓检测的预处理**
3. **基于 Augmentor 的数据增强**
4. **基于 PCANet 的特征提取**
5. **基于分类器的 Sanders 四分类输出**

它的关键贡献点不只是 PCANet 本身，而是：

- 在样本少的医学影像场景中
- 通过增强数据扩充训练集
- 让模型在 leave-one-out 设置下从约 **35%** 提高到约 **72%**

因此，如果你要复现论文，真正需要重点把握的是：

- **筛选规则是否一致**
- **预处理是否一致**
- **增强参数是否一致**
- **PCANet 特征实现是否一致**
- **训练/测试划分是否一致**

---

## 15. 后续可扩展方向

如果继续往下做，可以进一步补充：

1. 将本文档再展开成 **可执行 Python 项目结构说明**
2. 给出 **OpenCV + Augmentor + PCANet + SVM** 的完整代码框架
3. 补一份 **论文算法流程图**
4. 将本文档拆成：
   - 数据预处理说明
   - 数据增强说明
   - PCANet 实现说明
   - 复现实验说明

如果需要，我可以继续把这份 MD 文档再往下扩成“可直接编码实现”的版本。
