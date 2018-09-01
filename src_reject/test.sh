# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model kgonly --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 0 --baseepoch 10

# TODO: currently only confidence of unseen classes are calculated on documents of unseen classes
# running

# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --gpu 0.5
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1 --threshold 0.5

# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model vwvcvkg --ns 2 --ni 2 --sepoch 2 --rgidx 1 --train 0 --baseepoch 9
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model vwvcvkg --ns 2 --ni 2 --sepoch 2 --rgidx 1 --train 0 --baseepoch 9 --fulltest 1

# waiting
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 20 --rgidx 1 --train 1

# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model vwvcvkg --ns 1 --ni 4 --sepoch 1 --rgidx 1 --train 1
# python3 error.py --data 20news --unseen 0.5 --aug 0 --model vwvcvkg --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 1

# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vwvkg --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 6 --gpu 0.5
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model vwvkg --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 6 --fulltest 1

# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 1 --gpu 0.5

# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 20 --rgidx 1 --train 0 --baseepoch 10
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 20 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1

# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model vwvcvkg --ns 3 --ni 3 --sepoch 2 --rgidx 1 --train 0

# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model rnnfc --ns 5 --ni 3 --sepoch 10 --rgidx 1 --train 1 --threshold 0.9

# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model vwvc --ns 3 --ni 3 --sepoch 2 --rgidx 3 --train 1
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model vwvkg --ns 6 --ni 3 --sepoch 2 --rgidx 1 --train 1

# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model vwvcvkg --ns 3 --ni 3 --sepoch 2 --rgidx 1 --train 1

# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 20 --rgidx 1 --train 0

##################
# waiting for full testing once more time
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 15 --gpu 0.5
# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 15 --fulltest 1 --threshold 0.5

# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --gpu 0.5
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1 --threshold 0.5

# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 20 --rgidx 1 --train 0 --baseepoch 10 --gpu 0.5
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 20 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1 --threshold 0.5

# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 0 --baseepoch 10 --gpu 0.5
# python3 error.py --data 20news --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1 --threshold 0.5

##################
# done
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 0 --baseepoch 10 --gpu 0.5
# python3 error.py --data 20news --unseen 0.5 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1 --threshold 0.5
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 4 --train 0 --baseepoch 10 --fulltest 1

# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --gpu 0.5
# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1 --threshold 0.5
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1

# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model vwvcvkg --ns 2 --ni 2 --sepoch 10 --rgidx 1 --train 0 --baseepoch 5
# python3 error.py --data 20news --unseen 0.5 --aug 0 --model vwvcvkg --ns 2 --ni 2 --sepoch 10 --rgidx 1 --train 0 --baseepoch 5 --fulltest 1

# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model vwvcvkg --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 0 --baseepoch 9
python3 error.py --data 20news --unseen 0.25 --aug 0 --model vwvcvkg --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 0 --baseepoch 1
