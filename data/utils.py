import glob

import albumentations as A
import cv2
import numpy as np
import pandas as pd


def define_transforms():
    train_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),
        A.HorizontalFlip(p=0.5),
    ])
    val_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),
        A.HorizontalFlip(p=0.5),
    ])
    test_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0)
    ])

    return train_transform, val_transform, test_transform


def get_dataframe(root_path):

    mask_files = glob.glob(root_path + '*/*_mask*')
    image_files = [f.replace('_mask', '') for f in mask_files]

    def diagnosis(mask_path):
        return 1 if np.max(cv2.imread(mask_path)) > 0 else 0

    df = pd.DataFrame({"image_path": image_files,
                    "mask_path": mask_files,
                    "diagnosis": [diagnosis(x) for x in mask_files]})
    
    df.to_csv('data/data.csv', index=False)
    return df
