import torchvision.datasets as datasets
from torchvision import transforms

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DINOv3默认输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet标准化参数
        std=[0.229, 0.224, 0.225]
    )
])

# 下载训练集（会自动下载到指定目录）
train_dataset = datasets.EuroSAT(
    root='./data/eurosat',
    split='train',
    download=True,
    transform=transform
)

# 下载测试集
test_dataset = datasets.EuroSAT(
    root='./data/eurosat',
    split='test',
    download=True,
    transform=transform
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"类别数: {len(train_dataset.classes)}")
print(f"类别: {train_dataset.classes}")