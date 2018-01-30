# handwriting

handwriting is a work-in-progress handwriting recognition system. My initial goal is transcription of pages of my handwriting (I use pen on college-ruled paper and scan at 300 dpi) that is accurate enough to be faster than retyping.

Currently, I'm using traditional image processing techniques for extracting lines and words from the scanned page. Finding the positions of characters within words and classifying the extracted character images is performed using convolutional neural networks.

![QBF 2018-01-29](https://s3.us-east-2.amazonaws.com/bdzimmer-public/photos/qbf_20180129.png)

### License
This code is currently published under the 3-clause BSD license. See [LICENSE](LICENSE) for further information.