## ImageNet-1K Dataset

https://huggingface.co/datasets/HadlayZ/ImageNet-1K-ffcv

Generated from `split_and_upload.py`. Merge by running:
```Shell
cat train_chunk_* > train_complete.ffcv

cat val_chunk_* > val_complete.ffcv
```

and make sure results from:
```Shell
sha256sum train_500_0.50_90.ffcv

sha256sum val_500_0.50_90.ffcv
```

match with `checksum.txt`.