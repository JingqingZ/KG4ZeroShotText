# TODO: to run diff settings, by Peter
python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model vwvc --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 1
python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model vcvkg --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 1

python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vwvc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vcvkg --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model kgonly --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1

# TODO: to run baselines, by Jingqing
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model cnnfc --ns 4 --ni 2 --sepoch 4 --rgidx 1 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model rnnfc --ns 4 --ni 2 --sepoch 4 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 1

# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vwvkg --ns 1 --ni 1 --sepoch 5 --rgidx 7 --train 1

# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 10000 --model vwvcvkg --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 1
