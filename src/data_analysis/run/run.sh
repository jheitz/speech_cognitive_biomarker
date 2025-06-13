datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)

python -u run.py --config configs/test.yaml --name test
