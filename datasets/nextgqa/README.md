<b>train.csv</b>: QA annotations that are equal to nextqa train.csv. Note that we have replaced the correct answer_id with the corresponding text answer.

<b>test.csv/val.csv</b>: QA annotations that are subset of nextqa test.csv/val.csv. We exclude the questions that rely on global video content and those in the descriptive group.

<b>gsub_test.json/gsub_val.json</b>: time span annotations corresponding to the QAs in test.csv/val.csv

<b>frame2time_test.json/frame2time_val.json</b>: map the frame id into time seconds.

<b>map_vid_vidorID</b>: map the video_id in QA annotation file into video path.

<b>upbd_test.json/upbd_val.json</b>: sampled video timestamps for each video.

