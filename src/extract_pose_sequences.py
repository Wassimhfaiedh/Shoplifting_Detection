import cv2

import csv

import numpy as np



from pathlib import Path

from ultralytics import YOLO





# ============================================================

# CONFIGURATION

# ============================================================



MODEL_PATH = "yolo11s-pose.pt"



DATASET_DIR = Path("dataset")



NORMAL_DIR = DATASET_DIR / "Normal"

SHOPLIFTING_DIR = DATASET_DIR / "Shoplifting"



OUTPUT_DIR = Path("pose_dataset")

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)





# ============================================================

# TEMPORAL SETTINGS

# ============================================================



SEQUENCE_LENGTH = 10

SEQUENCE_STRIDE = 5





# ============================================================

# POSE SETTINGS

# ============================================================



POSE_CONF_THRESHOLD = 0.5

KEYPOINT_CONF_THRESHOLD = 0.5



NUM_KEYPOINTS = 17

FEATURE_SIZE = NUM_KEYPOINTS * 2





# ============================================================

# VIDEO EXTENSIONS

# ============================================================



VIDEO_EXTENSIONS = {

    ".mp4",

    ".avi",

    ".mov",

    ".mkv",

    ".m4v"

}





# ============================================================

# LABELS

# ============================================================



LABEL_MAP = {

    "normal": 0,

    "shoplifting": 1

}





# ============================================================

# LOAD YOLO POSE

# ============================================================



print("=" * 70)

print("Loading YOLO Pose")

print("=" * 70)



model = YOLO(

    MODEL_PATH

)



print(

    f"Model: {MODEL_PATH}"

)



print()





# ============================================================

# NORMALIZE LABEL

# ============================================================



def normalize_person_label(label):



    """

    New behavior:



    Normal       -> Normal

    Shoplifting  -> Shoplifting

    Unlabeled    -> Normal

    Empty        -> Normal

    Missing      -> Normal



    This matches the new Tkinter application:

    every person is Normal by default.

    """



    label = str(

        label or ""

    ).strip().lower()



    if label == "shoplifting":

        return "Shoplifting"



    # EVERYTHING ELSE DEFAULTS TO NORMAL

    return "Normal"





# ============================================================

# LOAD TRACKING CSV

# ============================================================



def load_tracking_csv(

    csv_path

):



    rows = []



    with open(

        csv_path,

        "r",

        encoding="utf-8-sig",

        newline=""

    ) as file:



        reader = csv.DictReader(

            file

        )



        required_columns = {

            "frame_index",

            "track_id",

            "x1",

            "y1",

            "x2",

            "y2"

        }



        columns = set(

            reader.fieldnames or []

        )



        missing = (

            required_columns

            -

            columns

        )



        if missing:



            raise RuntimeError(

                f"{csv_path} missing columns: "

                f"{missing}"

            )



        # ====================================================

        # READ ROWS

        # ====================================================



        for row in reader:



            try:



                source_frame = int(

                    float(

                        row["frame_index"]

                    )

                )



                track_id = int(

                    float(

                        row["track_id"]

                    )

                )



                x1 = float(

                    row["x1"]

                )



                y1 = float(

                    row["y1"]

                )



                x2 = float(

                    row["x2"]

                )



                y2 = float(

                    row["y2"]

                )



            except Exception:

                continue



            # ================================================

            # PERSON LABEL

            #

            # NEW:

            # Missing / Unlabeled automatically becomes Normal

            # ================================================



            person_label = normalize_person_label(

                row.get(

                    "person_label",

                    "Normal"

                )

            )



            rows.append({



                "source_frame":

                    source_frame,



                "track_id":

                    track_id,



                "label":

                    person_label,



                "bbox": [

                    x1,

                    y1,

                    x2,

                    y2

                ]



            })



    if len(rows) == 0:



        print(

            f"[WARNING] Empty CSV: "

            f"{csv_path}"

        )



        return {}



    # ========================================================

    # ORIGINAL SOURCE FRAME RANGE

    # ========================================================



    first_source_frame = min(



        row["source_frame"]



        for row in rows

    )



    last_source_frame = max(



        row["source_frame"]



        for row in rows

    )



    print(

        f"CSV source frames: "

        f"{first_source_frame} -> "

        f"{last_source_frame}"

    )



    # ========================================================

    # CONVERT SOURCE FRAME -> SEGMENT FRAME

    # ========================================================



    annotations = {}



    for row in rows:



        segment_frame = (



            row["source_frame"]



            -



            first_source_frame



            +



            1

        )



        if (

            segment_frame

            not in

            annotations

        ):



            annotations[

                segment_frame

            ] = []



        annotations[

            segment_frame

        ].append({



            "track_id":

                row["track_id"],



            "label":

                row["label"],



            "bbox":

                row["bbox"],



            "source_frame":

                row["source_frame"]



        })



    print(

        f"Converted segment frames: "

        f"{min(annotations.keys())} -> "

        f"{max(annotations.keys())}"

    )



    return annotations





# ============================================================

# NORMALIZE POSE

# ============================================================



def normalize_pose(

    keypoints_xy,

    keypoints_conf,

    crop_width,

    crop_height

):



    if (

        crop_width <= 1

        or

        crop_height <= 1

    ):



        return None



    features = []



    # ========================================================

    # 17 COCO KEYPOINTS

    # ========================================================



    for index in range(

        NUM_KEYPOINTS

    ):



        x = float(

            keypoints_xy[

                index

            ][0]

        )



        y = float(

            keypoints_xy[

                index

            ][1]

        )



        confidence = 1.0



        if (

            keypoints_conf

            is not None

        ):



            confidence = float(

                keypoints_conf[

                    index

                ]

            )



        # ====================================================

        # INVALID / LOW CONFIDENCE KEYPOINT

        # ====================================================



        if (

            confidence

            <

            KEYPOINT_CONF_THRESHOLD



            or



            x <= 0



            or



            y <= 0

        ):



            features.extend([

                0.0,

                0.0

            ])



            continue



        # ====================================================

        # NORMALIZE TO PERSON CROP

        # ====================================================



        x_norm = (

            x

            /

            crop_width

        )



        y_norm = (

            y

            /

            crop_height

        )



        x_norm = np.clip(

            x_norm,

            0.0,

            1.0

        )



        y_norm = np.clip(

            y_norm,

            0.0,

            1.0

        )



        features.extend([



            float(

                x_norm

            ),



            float(

                y_norm

            )



        ])



    return np.asarray(

        features,

        dtype=np.float32

    )





# ============================================================

# EXTRACT PERSON POSE

# ============================================================



def extract_person_pose(

    frame,

    bbox

):



    frame_height, frame_width = (

        frame.shape[:2]

    )



    x1, y1, x2, y2 = bbox



    # ========================================================

    # FLOAT -> INTEGER

    # ========================================================



    x1 = int(

        round(

            x1

        )

    )



    y1 = int(

        round(

            y1

        )

    )



    x2 = int(

        round(

            x2

        )

    )



    y2 = int(

        round(

            y2

        )

    )



    # ========================================================

    # CLIP BBOX

    # ========================================================



    x1 = max(

        0,

        min(

            x1,

            frame_width - 1

        )

    )



    y1 = max(

        0,

        min(

            y1,

            frame_height - 1

        )

    )



    x2 = max(

        0,

        min(

            x2,

            frame_width

        )

    )



    y2 = max(

        0,

        min(

            y2,

            frame_height

        )

    )



    # ========================================================

    # INVALID BBOX

    # ========================================================



    if (

        x2 <= x1

        or

        y2 <= y1

    ):



        return None



    # ========================================================

    # 8% PADDING

    # ========================================================



    bbox_width = (

        x2 - x1

    )



    bbox_height = (

        y2 - y1

    )



    padding_x = int(

        bbox_width

        *

        0.08

    )



    padding_y = int(

        bbox_height

        *

        0.08

    )



    crop_x1 = max(

        0,

        x1 - padding_x

    )



    crop_y1 = max(

        0,

        y1 - padding_y

    )



    crop_x2 = min(

        frame_width,

        x2 + padding_x

    )



    crop_y2 = min(

        frame_height,

        y2 + padding_y

    )



    # ========================================================

    # PERSON CROP

    # ========================================================



    person_crop = frame[

        crop_y1:crop_y2,

        crop_x1:crop_x2

    ]



    if (

        person_crop is None

        or

        person_crop.size == 0

    ):



        return None



    crop_height, crop_width = (

        person_crop.shape[:2]

    )



    # ========================================================

    # YOLO POSE ON PERSON CROP

    # ========================================================



    try:



        result = model.predict(



            source=person_crop,



            classes=[0],



            conf=POSE_CONF_THRESHOLD,



            verbose=False



        )[0]



    except Exception as e:



        print(

            f"Pose inference error: "

            f"{e}"

        )



        return None



    # ========================================================

    # CHECK DETECTION

    # ========================================================



    if (

        result.boxes is None

        or

        len(result.boxes) == 0

        or

        result.keypoints is None

    ):



        return None



    # ========================================================

    # CHOOSE BEST DETECTED PERSON

    # ========================================================



    confidences = (



        result.boxes.conf



        .cpu()



        .numpy()



    )



    best_index = int(

        np.argmax(

            confidences

        )

    )



    # ========================================================

    # KEYPOINT XY

    # ========================================================



    keypoints_xy = (



        result.keypoints.xy[

            best_index

        ]



        .cpu()



        .numpy()



    )



    # ========================================================

    # KEYPOINT CONFIDENCE

    # ========================================================



    keypoints_conf = None



    if (

        result.keypoints.conf

        is not None

    ):



        keypoints_conf = (



            result.keypoints.conf[

                best_index

            ]



            .cpu()



            .numpy()



        )



    if (

        len(

            keypoints_xy

        )

        !=

        NUM_KEYPOINTS

    ):



        return None



    # ========================================================

    # NORMALIZE

    # ========================================================



    return normalize_pose(



        keypoints_xy,



        keypoints_conf,



        crop_width,



        crop_height



    )





# ============================================================

# GET PERSON / TRACK LABEL

# ============================================================



def get_track_label(

    track_samples

):



    """

    New logic:



    Every track is Normal by default.



    Only a track with a majority of Shoplifting labels

    becomes Shoplifting.



    Returns:

        0 -> Normal

        1 -> Shoplifting

    """



    normal_count = 0

    shoplifting_count = 0



    for sample in track_samples:



        label = normalize_person_label(



            sample.get(

                "label",

                "Normal"

            )



        )



        if label == "Shoplifting":



            shoplifting_count += 1



        else:



            normal_count += 1



    # ========================================================

    # ONLY CHANGE TO SHOPLIFTING IF MAJORITY SAYS SHOPLIFTING

    # ========================================================



    if (

        shoplifting_count

        >

        normal_count

    ):



        return 1



    # ========================================================

    # DEFAULT

    # ========================================================



    return 0





# ============================================================

# PROCESS VIDEO

# ============================================================



def process_video(

    video_path,

    csv_path

):



    print()

    print("=" * 70)



    print(

        f"VIDEO : "

        f"{video_path.name}"

    )



    print(

        f"CSV   : "

        f"{csv_path.name}"

    )



    print("=" * 70)



    # ========================================================

    # LOAD CSV

    # ========================================================



    annotations = load_tracking_csv(

        csv_path

    )



    if len(

        annotations

    ) == 0:



        print(

            "[WARNING] "

            "No annotations."

        )



        return {}



    print(

        f"Annotated segment frames: "

        f"{len(annotations)}"

    )



    print(

        f"First annotation frame: "

        f"{min(annotations.keys())}"

    )



    print(

        f"Last annotation frame: "

        f"{max(annotations.keys())}"

    )



    # ========================================================

    # OPEN VIDEO

    # ========================================================



    cap = cv2.VideoCapture(

        str(

            video_path

        )

    )



    if not cap.isOpened():



        print(

            f"[ERROR] "

            f"Cannot open video: "

            f"{video_path}"

        )



        return {}



    total_frames = int(

        cap.get(

            cv2.CAP_PROP_FRAME_COUNT

        )

    )



    fps = cap.get(

        cv2.CAP_PROP_FPS

    )



    if fps <= 0:



        fps = 25.0



    print(

        f"Video frames: "

        f"{total_frames}"

    )



    print(

        f"FPS: "

        f"{fps:.2f}"

    )



    # ========================================================

    # TRACK STORAGE

    # ========================================================



    tracks = {}



    frame_index = 0



    pose_success = 0

    pose_failed = 0



    # ========================================================

    # FRAME LOOP

    # ========================================================



    while True:



        ret, frame = cap.read()



        if not ret:



            break



        frame_index += 1



        # ----------------------------------------------------

        # NO CSV ANNOTATION FOR THIS FRAME

        # ----------------------------------------------------



        if (

            frame_index

            not in

            annotations

        ):



            continue



        persons = annotations[

            frame_index

        ]



        # ====================================================

        # EACH TRACKED PERSON

        # ====================================================



        for person in persons:



            track_id = person[

                "track_id"

            ]



            bbox = person[

                "bbox"

            ]



            pose = extract_person_pose(



                frame,



                bbox



            )



            if pose is None:



                pose_failed += 1



                continue



            pose_success += 1



            if (

                track_id

                not in

                tracks

            ):



                tracks[

                    track_id

                ] = []



            tracks[

                track_id

            ].append({



                "frame":

                    frame_index,



                "source_frame":

                    person[

                        "source_frame"

                    ],



                "pose":

                    pose,



                "label":

                    person[

                        "label"

                    ]



            })



        # ====================================================

        # PROGRESS

        # ====================================================



        if (

            frame_index % 100 == 0



            or



            frame_index

            ==

            total_frames

        ):



            print(



                f"\rFrame "

                f"{frame_index}/"

                f"{total_frames}",



                end=""



            )



    cap.release()



    print()



    print(

        f"Pose success: "

        f"{pose_success}"

    )



    print(

        f"Pose failed : "

        f"{pose_failed}"

    )



    # ========================================================

    # TRACK SUMMARY

    # ========================================================



    for (

        track_id,

        samples

    ) in tracks.items():



        print(

            f"Track ID "

            f"{track_id}: "

            f"{len(samples)} "

            f"pose frames"

        )



    return tracks





# ============================================================

# CREATE TEMPORAL SEQUENCES

# ============================================================



def create_sequences(

    track_samples

):



    sequences = []



    # ========================================================

    # TOO SHORT

    # ========================================================



    if (

        len(

            track_samples

        )

        <

        SEQUENCE_LENGTH

    ):



        return sequences



    # ========================================================

    # SORT BY FRAME

    # ========================================================



    track_samples = sorted(



        track_samples,



        key=lambda x:

            x["frame"]



    )



    # ========================================================

    # SLIDING WINDOW

    # ========================================================



    for start_index in range(



        0,



        len(

            track_samples

        )

        -

        SEQUENCE_LENGTH

        +

        1,



        SEQUENCE_STRIDE



    ):



        window = track_samples[



            start_index:



            start_index

            +

            SEQUENCE_LENGTH



        ]



        # ====================================================

        # FRAME CONTINUITY

        # ====================================================



        frame_numbers = np.asarray([



            item[

                "frame"

            ]



            for item in window



        ])



        frame_gaps = np.diff(

            frame_numbers

        )



        # Allow small gaps

        if (

            len(

                frame_gaps

            ) > 0



            and



            np.max(

                frame_gaps

            ) > 3

        ):



            continue



        # ====================================================

        # POSE ARRAY

        # ====================================================



        sequence = np.stack([



            item[

                "pose"

            ]



            for item in window



        ])



        if (

            sequence.shape

            !=

            (

                SEQUENCE_LENGTH,

                FEATURE_SIZE

            )

        ):



            continue



        sequences.append({



            "sequence":

                sequence,



            "start_frame":

                int(

                    window[

                        0

                    ][

                        "frame"

                    ]

                ),



            "end_frame":

                int(

                    window[

                        -1

                    ][

                        "frame"

                    ]

                ),



            "source_start":

                int(

                    window[

                        0

                    ][

                        "source_frame"

                    ]

                ),



            "source_end":

                int(

                    window[

                        -1

                    ][

                        "source_frame"

                    ]

                )



        })



    return sequences





# ============================================================

# GLOBAL DATASET

# ============================================================



X = []



y = []



metadata = []





# ============================================================

# PROCESS FOLDER

# ============================================================



def process_folder(

    folder,

    folder_name

):



    print()



    print(

        "#" * 70

    )



    print(

        f"Processing folder: "

        f"{folder_name}"

    )



    print(

        "#" * 70

    )



    if not folder.exists():



        print(

            f"[WARNING] "

            f"Folder not found: "

            f"{folder}"

        )



        return



    # ========================================================

    # FIND VIDEOS

    # ========================================================



    videos = sorted([



        path



        for path in folder.iterdir()



        if (

            path.is_file()



            and



            path.suffix.lower()

            in

            VIDEO_EXTENSIONS

        )



    ])



    print(

        f"Found "

        f"{len(videos)} "

        f"videos"

    )



    # ========================================================

    # EACH VIDEO

    # ========================================================



    for video_path in videos:



        csv_path = (

            video_path.with_suffix(

                ".csv"

            )

        )



        if not csv_path.exists():



            print(

                f"[WARNING] "

                f"Missing CSV: "

                f"{csv_path.name}"

            )



            continue



        # ====================================================

        # PROCESS VIDEO

        # ====================================================



        tracks = process_video(



            video_path,



            csv_path



        )



        # ====================================================

        # EACH TRACK ID

        # ====================================================



        for (

            track_id,

            track_samples

        ) in tracks.items():



            # ================================================

            # GET PERSON LABEL

            #

            # NEW:

            # no Unlabeled case

            # ================================================



            track_label = get_track_label(

                track_samples

            )



            if (

                track_label

                ==

                1

            ):



                class_name = (

                    "Shoplifting"

                )



            else:



                class_name = (

                    "Normal"

                )



            # ================================================

            # CREATE SEQUENCES

            # ================================================



            sequences = create_sequences(

                track_samples

            )



            print(



                f"  ID "

                f"{track_id} "



                f"-> "

                f"{class_name} "



                f"-> "

                f"{len(sequences)} "

                f"sequences"



            )



            # ================================================

            # SAVE SEQUENCES

            # ================================================



            for (

                sequence_index,

                sequence_info

            ) in enumerate(

                sequences

            ):



                X.append(

                    sequence_info[

                        "sequence"

                    ]

                )



                y.append(

                    track_label

                )



                metadata.append({



                    "video":

                        video_path.name,



                    "csv":

                        csv_path.name,



                    "folder":

                        folder_name,



                    "person_class":

                        class_name,



                    "label":

                        track_label,



                    "track_id":

                        track_id,



                    "sequence":

                        sequence_index,



                    "segment_start":

                        sequence_info[

                            "start_frame"

                        ],



                    "segment_end":

                        sequence_info[

                            "end_frame"

                        ],



                    "source_start":

                        sequence_info[

                            "source_start"

                        ],



                    "source_end":

                        sequence_info[

                            "source_end"

                        ]



                })





# ============================================================

# PROCESS NORMAL FOLDER

# ============================================================



process_folder(

    NORMAL_DIR,

    "Normal"

)





# ============================================================

# PROCESS SHOPLIFTING FOLDER

# ============================================================



process_folder(

    SHOPLIFTING_DIR,

    "Shoplifting"

)





# ============================================================

# CHECK DATASET

# ============================================================



if len(X) == 0:



    raise RuntimeError(

        "No pose sequences generated."

    )





# ============================================================

# CONVERT TO NUMPY

# ============================================================



X = np.asarray(

    X,

    dtype=np.float32

)



y = np.asarray(

    y,

    dtype=np.int64

)





# ============================================================

# OUTPUT PATHS

# ============================================================



X_path = (

    OUTPUT_DIR

    /

    "X.npy"

)



y_path = (

    OUTPUT_DIR

    /

    "y.npy"

)



metadata_path = (

    OUTPUT_DIR

    /

    "metadata.csv"

)





# ============================================================

# SAVE NUMPY ARRAYS

# ============================================================



np.save(

    X_path,

    X

)



np.save(

    y_path,

    y

)





# ============================================================

# SAVE METADATA

# ============================================================



with open(



    metadata_path,



    "w",



    newline="",



    encoding="utf-8"



) as file:



    writer = csv.writer(

        file

    )



    writer.writerow([



        "sample",



        "video",



        "csv",



        "folder",



        "person_class",



        "label",



        "track_id",



        "sequence",



        "segment_start",



        "segment_end",



        "source_start",



        "source_end"



    ])



    for (

        sample_index,

        item

    ) in enumerate(

        metadata

    ):



        writer.writerow([



            sample_index,



            item[

                "video"

            ],



            item[

                "csv"

            ],



            item[

                "folder"

            ],



            item[

                "person_class"

            ],



            item[

                "label"

            ],



            item[

                "track_id"

            ],



            item[

                "sequence"

            ],



            item[

                "segment_start"

            ],



            item[

                "segment_end"

            ],



            item[

                "source_start"

            ],



            item[

                "source_end"

            ]



        ])





# ============================================================

# STATISTICS

# ============================================================



normal_count = int(

    np.sum(

        y == 0

    )

)



shoplifting_count = int(

    np.sum(

        y == 1

    )

)





# ============================================================

# FINAL OUTPUT

# ============================================================



print()



print(

    "=" * 70

)



print(

    "POSE DATASET CREATED"

)



print(

    "=" * 70

)



print()



print(

    f"X shape: "

    f"{X.shape}"

)



print(

    f"y shape: "

    f"{y.shape}"

)



print()



print(

    f"Normal sequences      : "

    f"{normal_count}"

)



print(

    f"Shoplifting sequences : "

    f"{shoplifting_count}"

)



print(

    f"Total sequences       : "

    f"{len(y)}"

)



print()



print(

    f"Sequence length: "

    f"{SEQUENCE_LENGTH}"

)



print(

    f"Features/frame: "

    f"{FEATURE_SIZE}"

)



print()



print(

    "Saved:"

)



print(

    X_path

)



print(

    y_path

)



print(

    metadata_path

)



print(

    "=" * 70

)





# ============================================================

# CLASS IMBALANCE INFORMATION

# ============================================================



if (

    normal_count > 0



    and



    shoplifting_count > 0

):



    ratio = (



        max(

            normal_count,

            shoplifting_count

        )



        /



        min(

            normal_count,

            shoplifting_count

        )



    )



    print(

        f"Class imbalance ratio: "

        f"{ratio:.2f}:1"

    )