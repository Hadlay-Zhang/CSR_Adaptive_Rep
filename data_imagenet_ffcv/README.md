## ImageNet-1K Dataset

ImageNet-1K pre-processed `.ffcv` data files. [link](https://huggingface.co/datasets/HadlayZ/ImageNet-1K-ffcv)

Split by `split_and_upload.py` due to single file size limit on huggingface (Validation Set is complete, no need for merging). Run:
```Shell
huggingface-cli download "HadlayZ/ImageNet-1K-ffcv" --local-dir data_imagenet_ffcv/ --repo-type dataset
```
to download from HF repo, and then merge multiple chunks by:
```Shell
cat train_chunk_* > train_complete.ffcv
```

and make sure results from:
```Shell
sha256sum train_500_0.50_90.ffcv
```

match with `checksum.txt`.