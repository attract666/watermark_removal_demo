# Dependencies and Installaion
1.Clone Repo
```
# clone the repository locally and install with
git clone https://github.com/attract666/watermark_removal_demo.git

# install Segment-anything
cd segment-anything-main
python setup.py install
```
2.Create Conda Environment and Install Dependencies
```
# create new anaconda env
conda create -n watermark python=3.8 -y
conda activate watermark

# install python dependencies
pip3 install -r requirements.txt
```
- CUDA >= 9.2
- PyTorch >= 1.7.1
- Torchvision >= 0.8.2
- Other required packages in ```requirements.txt```

# Get Started
## Prepare pretrained models
Ensure that the weights of the pre-trained model have been downloaded locally.

The directory structure will be arranged as:
```
ProPainter-main/
├── weights/
|   |- ProPainter.pth
|   |- recurrent_flow_completion.pth
|   |- raft-things.pth
|   |- i3d_rgb_imagenet.pt (for evaluating VFID metric)
|   |- README.md

segment-anything-main/
├── checkpoint/
|   |- sam_vit_h_4b8939.pth
```

## Test
To test your own videos, you can place the video files and mask images you need to process in ```ProPainter-main/inputs/video_completion/``` folder.

Run the following commands to try it out(Run it in ```ProPainter``` folder):
```
python3 inference_propainter.py --video inputs/video_completion/{your_video.mp4} --mask inputs/video_complection/{your_mask.png} --height 320 --width 240
```

The results will be saved in the ```results``` folder.

## Memory-efficient inference
You can use the following options to reduce memory usage further:
- Reduce the number of local neighbors through decreasing the ```--neighbor_length``` (default 10).
- Reduce the number of global references by increasing the ```--ref_stride``` (default 10).
- Set the ```--resize_ratio``` (default 1.0) to resize the processing video.
- Set a smaller video size via specifying the ```--width``` and ```--height```.
- Set ```--fp16``` to use fp16 (half precision) during inference.
- Reduce the frames of sub-videos ```--subvideo_length``` (default 80), which effectively decouples GPU memory costs and video length.

## Generate mask
If you want to generate a mask image or use the ```object removal``` function.

You can generate video frames using ```video_capture```.You only need to change variable ```video_path``` and variable ```output_path``` in ```video_capture```.

If you need to generate a mask image, you can run program ```generate_mask```.This is just a demo for generating mask images for TikTok videos.If you need to generate other image masks you need to modify ```generate_mask```.

# Acknowledgement
This code is based on [ProPainter](https://github.com/sczhou/ProPainter) and [segment-anything](https://github.com/facebookresearch/segment-anything).Thanks for their awesome works.

# Disclaimer
This project does not sell, share, encrypt, upload, or research any personal information. This project and its associated code are for study and research purposes only and do not constitute any express or implied warranty. The author is not responsible for any kind of damages that users may incur as a result of using this project and its code.