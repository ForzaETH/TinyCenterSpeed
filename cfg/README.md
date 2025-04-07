# TinyCenterSpeed Configuration

These files are used to configure parameters for TinyCenterSpeed. 
They do not configure the training parameters, instead they specify parameters used during inference.

There are two files: 

- __TinyCenterSpeed.yaml__: Specifies parameters for inference with TinyCenterSpeed.
- __opponent_tracker.yaml__: Specifies parameters for tracking using a Kalman Filter. 

## Parameters

The parameters are the following:

```yaml
TinyCenterSpeed:
  using_centerspeed: True # For testing purposes only
  rate: 40 # Rate in hz 
  image_size: 64 # Image Dimension
  pixel_size: 0.1 # Quantization of space in pixels
  boundary_inflation: 0.1 # Area to ignore around track boundary
  feature_size: 3 # Number of features: default 3
  num_opponents: 1 # Number of opponents to extract from the output
  using_raceline: True # Whether raceline information is used 
  visualize: False # Whether a live visualization is shown
  using_coral: False # Inference on Coral TPU
  dense: True # Dense preditions / Sparse predictions for a single opponent
  quantize: False # Quantize the model
  publish_cartesian: False # Publish the opponents in cartesian coords. Default: Frenet
  publish_foxglove: False # Publish an image visualizable in foxglove (image msg)
```


