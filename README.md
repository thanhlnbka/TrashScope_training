# TrashScope_training

Using the Dockerfile to Build the Training and Deployment Environment
To build the environment for training and deploying the model, use the following command:

```bash
git clone https://github.com/thanhlnbka/Trashscope_training.git
docker build -t ultralytics-docker .
```


## Navigate to Each Task Folder to Perform Training

### Run docker for training 

```bash
cd Trashscope
bash run_docker.sh
```

### Detection Task

* Make data training 

    Download label image from CVAT and run command for preprocess data


    ```bash
    # Convert CVAT format to YOLO format
    python cvat2yolo.py

    # If there are multiple folders that need to be converted, you can run this command to merge them into a single folder. 
    # Otherwise, skip to the next step.
    python3 merge_data.py

    # Convert YOLO format with 12 classes to YOLO format with 4 classes
    python3 relabel_new_format.py

    # Split data into training and validation sets
    python3 split_train_val.py


    ```

* Training

    ```bash
    cd task_det
    python3 training-det.py
    ```

### Classification  Task

*   Make data training

    Download label yml with format CVAT and run command for preprocess data
    ```bash
    python3 crop_image_from_label_cvat.py

    ```


* Training 

    ```bash
    cd task_cls
    python3 training-kFold-focal-loss.py
    ```


#### OTHER TASK FOR TUNING LABEL

* Training with 12 Classes (12cls)

    Before training, create the dataset for 12 classes similar to the detection task mentioned above. Then, proceed with training:

    ```bash
    python3 training-12cls.py

    ```


* Tuning with Pretrained 12 Classes Model

    After training with the 12-classes pretrained model, upload the new data to CVAT and download the unannotated data in YAML format.
    Next, run the following command to generate a new YAML file based on the pretrained model:

    ```bash
    python3 make_data_to_tunning_label.py
    ```
    Next, upload the new YAML file to CVAT for relabeling.