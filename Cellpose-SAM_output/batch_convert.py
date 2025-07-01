import glob
import tifffile
import imageio

for tif_path in glob.glob('/home/xinx/data/Colony_detection_SAM_3.0/Cellpose-SAM_output/*_cp_masks.tif'):
    mask = tifffile.imread(tif_path)
    mask2d = mask[0] if mask.ndim==3 else mask
    out_png = tif_path.replace('_cp_masks.tif', '_cp_masks.png')
    imageio.imwrite(out_png, (mask2d).astype('uint8'))
    print('Converted', tif_path, '→', out_png)
