import os
import os.path as osp
import sys
import shutil
import argparse

'''
筛选出未经过数据增强的绝缘子图像
原始图：002.jpg
增强图：002_1.jpg IMG_003_2.jpg
'''


def select_origin_img(args):
    img_path = args.input_dir
    out_path = args.output_dir

    img_dir = os.listdir(img_path)
    org_dir = []
    for filename in img_dir:
        # 筛选掉不是jpg的文件
        if not filename.endswith(".jpg") and not filename.endswith(".JPG"):
            continue

        base = filename.split(".")[0]
        if base.startswith("IMG"):
            base = base[4:]

        if "_" not in base:
            org_dir.append(filename)

    for filename in org_dir:
        print(filename)
        shutil.copyfile(os.path.join(img_path, filename), os.path.join(out_path, filename))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", help="input annotated directory")
    parser.add_argument("--output_dir", help="output dataset directory")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    print("| Creating output dir:", args.output_dir)

    # 把训练集标签转化为COCO的格式，并将标签对应的图片保存到目录 /train2017/
    print("—" * 50)
    print("| Select images:")
    select_origin_img(args)


if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)
