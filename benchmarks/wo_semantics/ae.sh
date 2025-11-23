python ae_demo.py --feature_type sequentials --anomaly_ratio 0.03 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_0.0_tar
python ae_demo.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar
torchrun --nproc-per-node=gpu ae_demo.py --feature_type sequentials --dataset HDFS --data_dir ../data/processed/HDFS_100k/hdfs_0.0_tar --anomaly_ratio 0.1
