python lstm_demo.py --label_type anomaly --feature_type sequentials --topk 10 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_1.0_tar
python lstm_demo.py --label_type anomaly --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_1.0_tar
torchrun --nproc-per-node=gpu lstm_demo.py --label_type anomaly --feature_type sequentials --dataset HDFS --data_dir ../data/processed/HDFS_100k/hdfs_1.0_tar --cache
