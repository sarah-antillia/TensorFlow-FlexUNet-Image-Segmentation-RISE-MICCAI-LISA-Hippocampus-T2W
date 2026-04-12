<h2>TensorFlow-FlexUNet-Image-Segmentation-RISE-MICCAI-LISA-Hippocampus-T2W (2026/04/11)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>RISE MICCAI LISA </b>
(<b>L</b>ow field pediatric brain magnetic resonance <b>I</b>mage <b>S</b>egmentation and quality <b>A</b>ssurance)
) <b>Hippocampus T2W</b>
 based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>), and a 394x466 pixels upscaled PNG
 <a href="https://drive.google.com/file/d/1-pFytpRqwgKQxPbTnvyJpNAPr8l8b95G/view?usp=sharing">
RISE-MICCAI-LISA-Hippocampus-T2W-ImageMask-Dataset.zip</a>, which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/tjahan/subtask-2a-hippocampus-segmentations">
<b>Subtask 2a - Hippocampus Segmentations</b>
</a> on the kaggle.com
<br><br>
<hr>
<b>Actual Image Segmentation for LISA-Hippocampus-T2W Images of 394x466 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the 
ground truth masks.
<br><br>
<b> class_color_map = {Right hippocampus:red,  Left hippocampus:green }</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10001_54.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10001_54.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10001_54.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10011_59.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10011_59.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10011_59.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10012_59.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10012_59.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10012_59.png" width="320" height="auto"></td>
</tr>
</table>
<hr>

<br>
<h3>1. Dataset Citation</h3>
The dataset used here was taken from <br><br>
<a href="https://www.kaggle.com/datasets/tjahan/subtask-2a-hippocampus-segmentations">
<b>Subtask 2a - Hippocampus Segmentations</b><br>
<b>RISE MICCAI Dataset ch</b>
</a> on the kaggle.com
<br><br>
For more information on <b>RISE MICCAL</b> and <b>LISA</b>, 
please refer to the following web site:<br>
<ul>
<li><a href="https://miccai.org/index.php/about-miccai/rise-miccai/">RISE-MICCAI</a>
</li>
<li> 
<a href="https://www.synapse.org/Synapse:syn55249552/wiki/627178">LISA 2024/2.Segmentation</a>
</li>
<li>
<a href="https://zenodo.org/records/15081583">Low field pediatric brain magnetic resonance Image Segmentation and quality Assurance (LISA)</a>
</li>
</ul>

The following explanation was taken from the above <a href="https://www.synapse.org/Synapse:syn55249552/wiki/627178">LISA 2024/2.Segmentation</a>
 web site.<br><br>
The challenge of automatic segmentation in low-field MRI environments poses a significant hurdle, 
particularly in regions with limited resources where high-field MRI systems are scarce. <br>
Structural delineation becomes even more challenging due to the lower resolution of accessible systems like 
the 0.064T Hyperfine scanner. Despite these obstacles, the benefits of utilizing low-field MRI, 
such as portability and reduced clinical costs, are undeniable, especially in settings where sedation 
for young patients is impractical.
<br><br>
Addressing this segmentation challenge head-on, the LISA challenge introduces its second task.
Participants are called upon to pioneer deep learning methods tailored for automatically segmenting the bilateral 
hippocampi in ultra low-field (0.064T) T2-weighted MRI images of early childhood brains. <br>
The bilateral hippocampi play a critical role in cognitive and memory functions, 
making their accurate segmentation essential for understanding abnormal neurodevelopment.
<br><br>
<b>License</b><br>
Unknown (on the kaggle web site)
<br>
<br>
<h3>
<a id="2">
2 LISA-Hippocampus-T2W ImageMask Dataset
</a>
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this LISA-Hippocampus-T2W Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1-pFytpRqwgKQxPbTnvyJpNAPr8l8b95G/view?usp=sharing">
RISE-MICCAI-LISA-Hippocampus-T2W-ImageMask-Dataset.zip</a>, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<br>
<pre>
./dataset
└─LISA-Hippocampus-T2W
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>LISA-Hippocampus-T2W Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/LISA-Hippocampus-T2W_Statistics.png" width="512" height="auto"><br>
<br><br>
As shown above, the number of images of train and valid datasets is not so large to use for the
 training set of our segmentation model.
<br><br>
<h3>2.2 Derivation PNG dataset</h3>
The folder structure of the original dataset is the following.<br>
<pre>
./archive
├─Low Field Images
│  ├─LISA_0001_ciso.nii
...
│  └─LISA_1016_ciso.nii
├─Subtask 2a - Hippocampus Segmentations
│  ├─LISA_0001_HF_hipp.nii
...
│  └─LISA_1016_HF_hipp.nii
└─Task 2 - Segmentation Validation
    ├─LISA_VALIDATION_0001_ciso.nii
...
    └─LISA_VALIDATION_0012_ciso.nii

</pre>

We used a simple Python script to generate the upscaled PNG  dataset from the following Image and Segmentation NIfTI files
of 79 subjects:<br>
<pre>
<b>Low Field Images/LISA_*_ciso.nii</b>
<b>Subtask 2a - Hippocampus Segmentations/LISA_*_HF_hipp.nii</b>
</pre>
<br>
We excluded all empty black masks and their corresponding images to generate the PNG dataset, 
which were irrelevant to train our segmentation model, and upscaled all images and masks to 394x466 pixels.
<br>
<h3>2.3 Train Image and Mask Saｍples</h3>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained LISA-Hippocampus-T2W TensorFlowFlexUNet Model by using the 
<a href="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 5
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specifed rgb color map dict for LISA-Hippocampus-T2W 1+2 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;LISA-Hippocampus-T2W rgb color map dict for 1+2 classes.
rgb_map = {(0,0,0):0, (255,0,0):1,(0,255,0):2, }
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was terminated at epoch 40.<br><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/train_console_output_at_epoch40.png" width="1024" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W</b> folder, 
and run the following bat file to evaluate TensorFlowUNet model for LISA-Hippocampus-T2W.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/evaluate_console_output_at_epoch40.png" width="1024" height="auto">
<br><br>Image-Segmentation-LISA-Hippocampus-T2W

<a href="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this <b>LISA-Hippocampus-T2W/test</b> was low, and dice_coef_multiclass was high as shown below.
<br>
<pre>
categorical_crossentropy,0.0014
dice_coef_multiclass,0.9994
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W</b> folder, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for LISA-Hippocampus-T2W.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of LISA-Hippocampus-T2W Images of 394x466 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ,
ground truth masks.
<br><br>
<b> class_color_map = {Right hippocampus:red,  Left hippocampus:green }</b>
<br><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10002_63.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10002_63.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10002_63.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10005_63.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10005_63.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10005_63.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10006_52.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10006_52.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10006_52.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10008_66.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10008_66.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10008_66.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10011_75.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10011_75.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10011_75.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/images/10016_58.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test/masks/10016_58.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/LISA-Hippocampus-T2W/mini_test_output/10016_58.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Quality Assurance and Hippocampal Segmentation on Low-Field Pediatric Magnetic Resonance Images</b><br>
Austin Tapp, Rahimeh Rouhi, Jeffrey Tanedo, Shreyash Zanjal, Sean Deoni, Marius George Linguraru & Natasha Lepore<br>
<a href="https://link.springer.com/chapter/10.1007/978-3-031-83008-2_6">
https://link.springer.com/chapter/10.1007/978-3-031-83008-2_6
</a>
<br><br>
<b>2.  Hippocampus Segmentation Using U-Net Convolutional Network from Brain Magnetic Resonance Imaging (MRI)</b><br>
Ruhul Amin Hazarika, Arnab Kumar Maji, Raplang Syiem, Samarendra Nath Sur, Debdatta Kandar <br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9485390/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC9485390/</a>
<br><br>
<b>3. Fully Automated Hippocampus Segmentation using T2-informed Deep Convolutional Neural Networks</b><br>
Maximilian Sackl, Christian Tinauer, Martin Urschler, Christian Enzinger, Rudolf Stollberger, Stefan Ropele <br>
<a href="https://www.sciencedirect.com/science/article/pii/S1053811924002647">
https://www.sciencedirect.com/science/article/pii/S1053811924002647</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Hippocampus-T1W</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Hippocampus-T1W">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Hippocampus-T1W
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
