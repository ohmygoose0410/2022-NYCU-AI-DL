## Seq2Seq

**Lab Objective:** 

In this assignment, you will need to implement a seq2seq LSTM encoder-decoder network with recurrent units for English spelling correction.

**Requirements:**

1.  Track A. Implement a seq2seq LSTM model by yourself
Track B. Or Use sample code (sample.py)
    - Modify encoder, decoder, and training functions.
    - Implement the evaluation function and the dataloader.
    - Note: This sample code is provided for those who have problem constructing their own project. If you wish to improve the performance, we strongly recommend Track A for you!
2.  Plot the CrossEntropy training loss and BLEU-4 testing score curves during
training. And Output the correction results from test.json and new_test.json
3.  (Optional parts) Compare performance changes due to different parameters and
model structures and write them into reports (Bonus)

## Training Loss Curve

<img src="img/training_loss.png" width="360">

## BLEU-4 Testing Score Curve

<img src="img/testing_score.png" width="360">

## Example

1.  Dataset - Test

    <img src="img/test_output.png" width="300">

2.  Dataset - New Test

    <img src="img/new_test_output.png" width="300">