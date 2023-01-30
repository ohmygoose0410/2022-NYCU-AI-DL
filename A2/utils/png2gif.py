import glob
from PIL import Image
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create and save animated GIF with Python')
    parser.add_argument('-img_dir', type=str, required=True)
    parser.add_argument('-target_path', type=str, required=True)

    args = parser.parse_args()
    # fp_in = "./result/linear_prob/*.png"
    # fp_out = "./result/linear_prob.gif"

    imgs = (Image.open(f) for f in sorted(glob.glob(os.path.join(*[args.img_dir, "*.png"])), key=len))
    img = next(imgs)
    img.save(fp=args.target_path, format='GIF', append_images=imgs,
            save_all=True, duration=20, loop=0)