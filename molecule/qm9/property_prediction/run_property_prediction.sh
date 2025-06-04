python -u main_qm9.py --num_workers 2 --lr 1e-3 --property gap --exp_name exp_1_gap

python -u main_qm9.py --num_workers 2 --lr 1e-3 --property homo --exp_name exp_1_homo
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property lumo --exp_name exp_1_lumo


python -u main_qm9.py --num_workers 2 --lr 5e-4 --property mu --exp_name exp_1_mu
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property Cv --exp_name exp_1_Cv