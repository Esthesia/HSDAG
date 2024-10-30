Set up the conda environment with following command:
conda env create -f environment.yml 

To use the RNN baseline, open RNN_bert.ipynb and RNN_resnet_inceptionv3.ipynb, edit the input data shape specified by your model to match your XML file, and run all cells in each notebook. 

To use Placeto, unzip PlacetoOpenVINO.zip in your directory and update the location of your XML file and appropriate input. To run the code, install and use our TensorFlow and OpenVINO Conda environment by executing the following commands:
conda env create -f environment_tf.yml 

Our Placeto source code is modified based on: https://github.com/mmitropolitsky/device-placement

Please switch to the environment installed in the first time.
To run HSDAG, provide the location of your XML file in config.py, and then run main.py with the appropriate input parameters for your experiment.

In HSDAG, we used the GPN by Yunchong Song from https://arxiv.org/pdf/2402.14393

Their license is as follow:
MIT License

Copyright (c) 2024 Yunchong Song

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.