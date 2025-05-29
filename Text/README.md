## Download Datasets
```Shell
mkdir datasets
huggingface-cli download "mteb/banking77" --local-dir datasets/banking77/ --repo-type dataset
huggingface-cli download "mteb/mtop_intent" --local-dir datasets/mtop_intent/ --repo-type dataset
huggingface-cli download "mteb/tweet_sentiment_extraction" --local-dir datasets/tweet_sentiment_extraction/ --repo-type dataset

huggingface-cli download "mteb/biorxiv-clustering-p2p" --local-dir datasets/biorxiv-clustering-p2p/ --repo-type dataset
huggingface-cli download "mteb/biorxiv-clustering-s2s" --local-dir datasets/biorxiv-clustering-s2s/ --repo-type dataset
huggingface-cli download "mteb/twentynewsgroups-clustering" --local-dir datasets/twentynewsgroups-clustering/ --repo-type dataset

huggingface-cli download "mteb/fiqa" --local-dir datasets/fiqa/ --repo-type dataset
huggingface-cli download "mteb/nfcorpus" --local-dir datasets/nfcorpus/ --repo-type dataset
huggingface-cli download "mteb/scifact" --local-dir datasets/scifact/ --repo-type dataset
```