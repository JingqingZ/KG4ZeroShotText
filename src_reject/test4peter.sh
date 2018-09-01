python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model vcvkg --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 0 --basepeoch 10 --fulltest 1
python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vcvkg --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1
python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model kgonly --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1 --baseepoch 10 --fulltest 1

# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model vcvkg --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 0 --basepeoch 10 --fulltest 1
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model vcvkg --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model kgonly --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1 --baseepoch 10 --fulltest 1
