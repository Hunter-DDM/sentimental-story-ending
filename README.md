# sentimental-story-ending
The codes and data for paper "Learning to Control the Fine-grained Sentiment for Story Ending Generation (ACL 2019)". 

## HOW TO USE THIS CODE

There are three directories for our three proposed methods. 

### For model "SIC-Seq2Seq + DA"

1. enter the "SIC-Seq2Seq + DA" and sentiment analyzer directory

        cd SIC_Seq2Seq_DA/transfer_emotion

2. train sentiment analyzer

        python3 train.py -gpus 1 -config transfer.yaml -log sentiment_analyzer

3. estimate sentiments for story datasets

        python3 train.py -mode eval -gpus 1 -config transfer.yaml -restore ./experiments/transfer/sentiment_analyzer/checkpoint.pt -log sentiment_analyzer
        
4. enter the and sentimental generator directory

        cd ../story_generation/raw_data
        
5. partition dataset of sentimental generator

        python3 partition.py
        
6. move the estimated sentiments into sentimental generator directory

        cp ../../transfer_emotion/data/train.tgt.emotion story_data/
        cp ../../transfer_emotion/data/valid.tgt.emotion story_data/
        
7. preprocessing for sentimental generator

        cd ..
        python3 preprocess.py -share -load_data raw_data/story_data/ -save_data data/

8. train sentimental generator

        python3 train.py -gpus 1 -config story.yaml -log sentimental_generator

9. test sentimental generator

        python3 train.py -mode eval -gpus 1 -config story.yaml -restore ./experiments/story/sentimental_generator/checkpoint.pt -log sentimental_generator

### For model "SIC-Seq2Seq + RM"

1. enter the "SIC-Seq2Seq + RM" and sentiment analyzer directory

        cd SIC_Seq2Seq_RM/transfer_emotion

2. train sentiment analyzer

        python3 train.py -gpus 1 -config transfer.yaml -log sentiment_analyzer

3. estimate sentiments for story datasets

        python3 train.py -mode eval -gpus 1 -config transfer.yaml -restore ./experiments/transfer/sentiment_analyzer/checkpoint.pt -log sentiment_analyzer
        
4. enter the and sentimental generator directory

        cd ../story_generation/raw_data
        
5. partition dataset of sentimental generator

        python3 partition.py
        
6. move the estimated sentiments into sentimental generator directory

        cp ../../transfer_emotion/data/train.tgt.emotion story_data/
        cp ../../transfer_emotion/data/valid.tgt.emotion story_data/
        
7. preprocessing for sentimental generator

        cd ..
        python3 preprocess.py -share -load_data raw_data/story_data/ -save_data data/

8. train sentimental generator

        python3 train.py -gpus 1 -config story.yaml -log sentimental_generator

9. test sentimental generator

        python3 train.py -mode eval -gpus 1 -config story.yaml -restore ./experiments/story/sentimental_generator/checkpoint.pt -log sentimental_generator

### For model "SIC-Seq2Seq + RB"

1. enter the "SIC-Seq2Seq + RB" and sentimental generator directory

        cd SIC_Seq2Seq_RB/story_generation

2. partition dataset of sentimental generator

        cd raw_data/
        python3 partition.py
        
3. preprocessing for sentimental generator

        cd ..
        python3 preprocess.py -share -load_data raw_data/story_data/ -save_data data/

4. train sentimental generator

        python3 train.py -gpus 1 -config story.yaml -log sentimental_generator

5. test sentimental generator

        python3 train.py -mode eval -gpus 1 -config story.yaml -restore ./experiments/story/sentimental_generator/checkpoint.pt -log sentimental_generator

## PROVIDED CHECKPOINTS

 We provided five checkpoints of this project. you can download them from:
 
 https://drive.google.com/open?id=1VPopk1F_3e3KkDQ2BcnKSG4KDJHlKCUw
 
 DA_SA is the checkpoint of sentiment analyzer for model "SIC-Seq2Seq + DA"
 
 RM_SA is the checkpoint of sentiment analyzer for model "SIC-Seq2Seq + RM"
 
 DA_SG is the checkpoint of sentimental generator for model "SIC-Seq2Seq + DA"
 
 RM_SG is the checkpoint of sentimental generator for model "SIC-Seq2Seq + RM"
 
 RB_SG is the checkpoint of sentimental generator for model "SIC-Seq2Seq + RB"

