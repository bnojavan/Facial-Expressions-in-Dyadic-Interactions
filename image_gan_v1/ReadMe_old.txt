model_new and main_new are actually not the lastest, they are for the L1 loss with layer features only.

Use model.py and main.py are actually the lastest, containing both choices: the layer_feature choice or not.

For training:

Each image_shape%02d.mat file contains a 7000 X 256 X 256 X 4 matrix 'image_shape', where image_shape[:,:,:,0:3] are 7000 color images, image_shape[:,:,:,3] are sketches.

For testing:

The folder 'gen_openbw' contains 7000 folders of all the sketch results from the first stage. 'openbw' means these reuslts have been processed by a matlab script that get rid of little noise.

During testing, the results are written to 'gen_openbw/im'

Different testing function, test, test2 and test3 are used for different purpose.

test: generate image frames with large steps, 100:5:400
test2: generate image frames with step=1, 100:1:400 (generating every frame)
test3: for comparison between the model with layer_feature choice and the model without layer_feature choice

