python train.py ja en -datadir ./corpus_data/tanaka -savedir ./model/sample -edim 512 -nhid 1024 -gensim_mode load -lr 0.01 -gpunum 6 -epoch 1 -useDropout -dlr 0.2 -batch 50 -pooling 100 -genlimit 60 -useSeed -seed_num 7 -name sample