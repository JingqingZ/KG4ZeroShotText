# TODO: to run diff settings, by Peter
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model vwvc --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model vcvkg --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 1

# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vwvc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vcvkg --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model kgonly --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1

# TODO: to run baselines, by Jingqing

# running

# if have time can run
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model vwvcvkg --ns 3 --ni 3 --sepoch 2 --rgidx 1 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model vwvcvkg --ns 3 --ni 3 --sepoch 2 --rgidx 1 --train 0 --baseepoch 5
# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model vwvcvkg --ns 3 --ni 3 --sepoch 2 --rgidx 1 --train 0 --baseepoch 5 --fulltest 1

# waiting for testing

###############################
# give up
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 10000 --model vwvcvkg --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 10000 --model vwvcvkg --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model vwvc --ns 3 --ni 3 --sepoch 2 --rgidx 3 --train 1

###############################
# waiting for full test
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model vwvkg --ns 6 --ni 3 --sepoch 1 --rgidx 1 --train 0

###############################
# all done
# python3 train_seen.py --data dbpedia --unseen 0.5 --aug 0 --model vw --ns 0 --ni 0 --sepoch 1 --rgidx 1 --train 1
# python3 train_seen.py --data 20news --unseen 0.5 --aug 0 --model vw --ns 0 --ni 0 --sepoch 5 --rgidx 1 --train 1
# dbpedia vwvcvkg is saved in a file with special name
# 20news vwvcvkg is save in a file with special namme
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 20 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model cnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 4 --train 1
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 10 --rgidx 1 --train 1 --gpu 0.5
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 4 --train 1 --gpu 0.5
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model rnnfc --ns 1 --ni 1 --sepoch 5 --rgidx 1 --train 1 --gpu 0.5
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model vwvcvkg --ns 2 --ni 2 --sepoch 10 --rgidx 1 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model vwvcvkg --ns 2 --ni 2 --sepoch 2 --rgidx 1 --train 0
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model vwvkg --ns 1 --ni 1 --sepoch 2 --rgidx 2 --train 1
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model vcvkg --ns 1 --ni 1 --sepoch 2 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model kgonly --ns 1 --ni 1 --sepoch 2 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model vwvkg --ns 1 --ni 1 --sepoch 5 --rgidx 7 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model kgonly --ns 5 --ni 3 --sepoch 2 --rgidx 1 --train 0 --baseepoch 10
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model vcvkg --ns 6 --ni 3 --sepoch 1 --rgidx 9 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model kgonly --ns 6 --ni 3 --sepoch 1 --rgidx 2 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model vcvkg --ns 6 --ni 3 --sepoch 1 --rgidx 1 --train 0 --baseepoch 10
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model kgonly --ns 6 --ni 3 --sepoch 1 --rgidx 1 --train 0 --baseepoch 10
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model vcvkg --ns 6 --ni 3 --sepoch 1 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model kgonly --ns 6 --ni 3 --sepoch 1 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1

# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 4 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.5 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 4 --rgidx 1 --train 0 --baseepoch 10
# python3 error.py --data 20news --unseen 0.5 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 4 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1

# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 4 --rgidx 1 --train 1
# python3 train_unseen.py --data 20news --unseen 0.25 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 4 --rgidx 1 --train 0 --baseepoch 10
# python3 error.py --data 20news --unseen 0.25 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 4 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1

# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 2 --rgidx 2 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.5 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 2 --rgidx 1 --train 0 --baseepoch 10
# python3 error.py --data dbpedia --unseen 0.5 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 2 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1

# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 2 --rgidx 1 --train 1
# python3 train_unseen.py --data dbpedia --unseen 0.25 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 2 --rgidx 1 --train 0 --baseepoch 10
# python3 error.py --data dbpedia --unseen 0.25 --aug 0 --model autoencoder --ns 0 --ni 0 --sepoch 2 --rgidx 1 --train 0 --baseepoch 10 --fulltest 1
