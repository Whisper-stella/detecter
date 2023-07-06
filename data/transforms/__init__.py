from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:   #数据增强方法 构建数据时会使用
        transform = [      
            ConvertFromInts(),       # 转换成浮点型
            PhotometricDistort(),    # 像素级变换

            Expand(cfg.INPUT.PIXEL_MEAN),  # 随机扩展
            RandomSampleCrop(),            # 随机裁剪
            RandomMirror(),                # 随机镜像

            ToPercentCoords(),             # 这是因为几何变换后图像尺寸改变了，box也要改变
            Resize(cfg.INPUT.IMAGE_SIZE),  # 将输入图片resize成统一尺寸
            SubtractMeans(cfg.INPUT.PIXEL_MEAN), # 零均值化，便于运算
            ToTensor(),                    # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
        ]
    else:  # 测试的图片就是原模原样
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)   # 初始化数据增强方法，后面用transform统一调用
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
