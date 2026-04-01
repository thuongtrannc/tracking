This repo develops the algorithm for 3D multiple object tracking (3D MOT). For tracking, I use Kalman filter to track and estimate x, y, yaw, w, h, vx, vy of objects. The Hungarian algorithm is utilized for matching.

The order of developing this repo is as follows:

- The repo is started by generating simulating data for one object with init position, yaw and moving in linear constant
- Visualize the motion of one object
- Kalman filter for one object tracking
- Data generation for matching algorithm testing
- Matching with the Hungarian algorithm
- Data generation for multiple object tracking
- Finally, I develop the algorithm for 3D object tracking with magaging dead and alive objects
- Visualization of the algorithm is added for better institutional understanding.

** Basic running commands **

```python tools/data_single.py```

```python test/test_kf.py```

```python vis/data_vis.py```

```python vis/filter_vis.py```

```python tools/data_hungarian.py```

```python test/test_hungarian.py```

```python vis/hungarian_vis.py```

```python tools/data_multi.py```

```python vis/multi_vis.py```

This code is the not official implementation of the paper `3D Multi-Object Tracking: A Baseline and New Evaluation Metrics`: https://arxiv.org/pdf/1907.03961.pdf 



