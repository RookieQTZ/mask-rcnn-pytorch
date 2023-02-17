import os.path

import numpy as np
import cv2

'''
将mask rcnn预测的截图保存下来png
'''


# (h, w)
def save_kou_img(org_img,
                 masks,
                 filename,
                 out_path,
                 scores,
                 box_thresh: float = 0.1,
                 mask_thresh: float = 0.5):
    '''
    :param org_img:原始图片 (w, h)
    :param masks:预测的掩码数据 (batch, h, w)
    :param filename:原始图像名
    :param scores:预测分数
    :param out_path:保存路径
    :param thresh:置信阈值，score大于此值才会截图
    '''
    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    if masks is not None:
        masks = masks[idxs]
    masks = np.where(masks > mask_thresh, True, False)
    '''
    image为正常的RGB图像，alpha_array为和image尺寸大小（image.shape）相同的array，alpha_array里面的值范围也是0-1，0为完全透明，1为完全不透明。
    随后使用plt.savefig()保存图像，其中参数transparent=True，这个设置会让坐标轴，以及图像补丁（也就是alpha为0的位置）都变为透明；
    bbox_inches和pad_inches的设置是为了保存图像时删除图像的白边。
    其余地方保留原始像素
    '''
    # (c, h, w)
    h = masks.shape[1]
    w = masks.shape[2]
    np_org_img = np.array(org_img)  #(h, w, c)
    np_img = np.empty((4, h, w), dtype=int)  #(c, h, w)
    np_alpha = np.ones((h, w))

    for n, mask in enumerate(masks):
        for i in range(0, h):
            for j in range(0, w):
                if mask[i, j]:
                    # 不透明
                    # 完全透明的像素应该将alpha设置为0，完全不透明的像素应该将alpha设置为255/65535。
                    np_img[:, i, j] = [np_org_img[i, j, 0], np_org_img[i, j, 1], np_org_img[i, j, 2], 255]
                else:
                    np_img[:, i, j] = [255, 255, 255, 0]

    # rgba -> bgra
    r = np_img[0, :, :]
    g = np_img[1, :, :]
    b = np_img[2, :, :]
    a = np_img[3, :, :]
    np_img = np.stack((b, g, r, a), axis=2)  # h w c

    # 格式转换操作，确保可以从numpy数组转换为img
    # imwrite函数只支持保存8位（或16位无符号）
    np_img = np_img.astype(np.uint8)
    # c h w -> h w c
    # np_img = np.uint8(np_img.transpose(1, 2, 0))
    base = filename.split(".")[0]
    cv2.imwrite(os.path.join(out_path, "kou_" + base + ".png"), np_img)
