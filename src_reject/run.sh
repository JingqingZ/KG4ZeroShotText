python3 train_seen.py --data dbpedia --unseen 0.5 --model vw --ns 0 --ni 0 --sepoch 1 --rgidx 1 --train 1
python3 train_unseen.py --data dbpedia --unseen 0.25 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
