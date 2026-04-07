# Bug Report Datasets for FairBug

This directory contains bug report datasets from 5 popular deep learning frameworks.

## Dataset Description

Each CSV file contains bug reports collected from GitHub issues. The reports are labeled as performance-related (1) or not performance-related (0).

### File Format
Each CSV has the following columns:
- `report_id`: Unique identifier for the bug report
- `title`: Bug report title
- `description`: Full bug report description
- `label`: Binary label (1 = performance bug, 0 = not performance bug)
- `created_at`: Creation date (YYYY-MM-DD)
- `status`: Issue status (open/closed)
- `comments_count`: Number of comments

### Statistics

| Project | Total Reports | Performance Bugs | Non-Performance Bugs | Performance % |
|---------|--------------|------------------|---------------------|---------------|
| TensorFlow | 1490 | 279 | 1211 | 18.7% |
| PyTorch | 752 | 95 | 657 | 12.6% |
| Keras | 668 | 135 | 533 | 20.2% |
| MXNet | 516 | 65 | 451 | 12.6% |
| Caffe | 286 | 33 | 253 | 11.5% |
| **TOTAL** | **3712** | **607** | **3105** | **16.4%** |

### Data Source
These datasets are derived from the GitHub repositories:
- TensorFlow: https://github.com/tensorflow/tensorflow/issues
- PyTorch: https://github.com/pytorch/pytorch/issues
- Keras: https://github.com/keras-team/keras/issues
- MXNet: https://github.com/apache/mxnet/issues
- Caffe: https://github.com/BVLC/caffe/issues

### Sample Data

```csv
report_id,title,description,label,created_at,status,comments_count
1,"Model training very slow","When training on GPU, the model runs extremely slowly. Memory usage is high.",1,2023-01-15,closed,5
2,"Documentation update needed","The installation guide is outdated for Windows 11",0,2023-01-16,open,2
3,"Memory leak in training loop","Memory usage keeps increasing during training, eventually crashes",1,2023-01-17,closed,8
4,"Add new feature request","Please add support for custom loss functions",0,2023-01-18,open,3
5,"GPU utilization drops","GPU usage drops to 0% after first epoch",1,2023-01-19,closed,6