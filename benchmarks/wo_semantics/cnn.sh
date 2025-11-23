python cnn_demo.py --label_type anomaly --feature_type sequentials --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_1.0_tar
python cnn_demo.py --label_type anomaly --feature_type sequentials --dataset BGL --data_dir ../data/processed/BGL/bgl_1.0_tar
torchrun --nproc-per-node=gpu cnn_demo.py --label_type anomaly --feature_type sequentials --dataset HDFS --data_dir ../data/processed/HDFS_100k/hdfs_1.0_tar
