# AIM327: Amazon SageMaker and Apache MXNet: Tips & Tricks (re:Invent builder session)

This repository contains the notebooks used in the [AIM327](https://www.portal.reinvent.awsevents.com/connect/sessionDetail.ww?SESSION_ID=88717) [Builder Sessions](https://reinvent.awsevents.com/learn/builders-sessions/). The materials in this
repository are derived from notebooks in [GluonCV](https://github.com/dmlc/gluon-cv) and [GluonNLP](https://github.com/dmlc/gluon-nlp).

# Setup

We will follow the [SageMaker Get Started guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html) and create a new notebook instance of ml.p3.2xlarge type, with at least 10GB of volume size.
  - We won't need S3 for today's session so feel free to skip step 1.2.
  - Follow step 2.1. Leave the "Create notebook" unchanged. In "Start notebook", paste the following script:
      ```bash
      #!/bin/bash

      set -e

      pip install 'gluonnlp[extras]' gluoncv
      python -m spacy download en
      python -m spacy download de
      python -m nltk.downloader all
      ```
  - Once the instance is InService, click on the "Open Jupyter", and create a `conda_mxnet_p36` instance. In the empty cell, type and run:
      ```
      !git clone https://github.com/hetong007/reinvent2018-buildersession
      ```
  - Go back to the jupyter file browser. You should see the content of this repository.

# Resources
- [Apache MXNet](https://beta.mxnet.io)
- [Dive into Deep Learning](https://en.diveintodeeplearning.org/)
- [GluonCV](https://github.com/dmlc/gluon-cv)
- [GluonNLP](https://github.com/dmlc/gluon-nlp)
- [MXNet on SageMaker](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/mxnet/README.rst)
