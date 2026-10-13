import cv2

import csv

import re

import tkinter as tk



from tkinter import ttk, filedialog, messagebox

from pathlib import Path

from PIL import Image, ImageTk

from ultralytics import YOLO





# ============================================================

# CONFIGURATION

# ============================================================



MODEL_PATH = "yolo11s-pose.pt"



OUTPUT_DIR = Path("dataset")



NORMAL_DIR = OUTPUT_DIR / "Normal"

SHOPLIFTING_DIR = OUTPUT_DIR / "Shoplifting"



NORMAL_DIR.mkdir(

    parents=True,

    exist_ok=True

)



SHOPLIFTING_DIR.mkdir(

    parents=True,

    exist_ok=True

)





DISPLAY_WIDTH = 640

DISPLAY_HEIGHT = 420



CONF_THRESHOLD = 0.5



TRACKER = "bytetrack.yaml"





# ============================================================

# APPLICATION

# ============================================================



class ShopliftingApp:



    def __init__(

        self,

        root

    ):



        self.root = root



        self.root.title(

            "Shoplifting Dataset Labeling - YOLO Pose"

        )



        self.root.geometry(

            "1450x820"

        )



        self.root.minsize(

            1100,

            650

        )



        # ====================================================

        # YOLO MODEL

        # ====================================================



        print(

            f"Loading YOLO model: {MODEL_PATH}"

        )



        try:



            self.model = YOLO(

                MODEL_PATH

            )



        except Exception as e:



            messagebox.showerror(

                "Model Error",

                f"Could not load YOLO model:\n\n{e}"

            )



            raise



        # ====================================================

        # VIDEO STATE

        # ====================================================



        self.cap = None



        self.video_path = None



        self.source_id = None



        self.running = False



        self.after_id = None



        self.fps = 25.0



        self.video_width = 0



        self.video_height = 0



        self.total_frames = 0



        self.frame_index = 0



        # ====================================================

        # RECORDING STATE

        # ====================================================



        self.writer = None



        self.csv_file = None



        self.csv_writer = None



        self.active_label = None



        self.output_path = None



        # ====================================================

        # PERSON LABEL STATE

        #

        # Every new person is NORMAL by default.

        #

        # Only selected IDs are changed to Shoplifting.

        # ====================================================



        self.person_labels = {}



        self.current_track_ids = []



        # ====================================================

        # TKINTER IMAGE REFERENCES

        # ====================================================



        self.original_photo = None



        self.pose_photo = None



        # ====================================================

        # BUILD UI

        # ====================================================



        self.build_ui()



        # ====================================================

        # CLOSE

        # ====================================================



        self.root.protocol(

            "WM_DELETE_WINDOW",

            self.close_app

        )





    # ========================================================

    # USER INTERFACE

    # ========================================================



    def build_ui(self):



        # ====================================================

        # TITLE

        # ====================================================



        ttk.Label(

            self.root,

            text=(

                "YOLO Pose + ByteTrack | "

                "Shoplifting Dataset Creator"

            ),

            font=(

                "Arial",

                17,

                "bold"

            )

        ).pack(

            pady=(8, 5)

        )



        # ====================================================

        # MAIN FRAME

        # ====================================================



        main_frame = ttk.Frame(

            self.root,

            padding=5

        )



        main_frame.pack(

            fill="both",

            expand=True,

            padx=5,

            pady=2

        )



        # ====================================================

        # VIDEO AREA

        # ====================================================



        video_frame = ttk.Frame(

            main_frame

        )



        video_frame.pack(

            side="left",

            fill="both",

            expand=True

        )



        # ====================================================

        # ORIGINAL VIDEO

        # ====================================================



        original_box = ttk.LabelFrame(

            video_frame,

            text="Original Video"

        )



        original_box.pack(

            side="left",

            fill="both",

            expand=True,

            padx=4

        )



        self.original_panel = ttk.Label(

            original_box,

            anchor="center"

        )



        self.original_panel.pack(

            fill="both",

            expand=True,

            padx=5,

            pady=5

        )



        # ====================================================

        # YOLO POSE VIDEO

        # ====================================================



        pose_box = ttk.LabelFrame(

            video_frame,

            text="YOLO Pose + ByteTrack"

        )



        pose_box.pack(

            side="left",

            fill="both",

            expand=True,

            padx=4

        )



        self.pose_panel = ttk.Label(

            pose_box,

            anchor="center"

        )



        self.pose_panel.pack(

            fill="both",

            expand=True,

            padx=5,

            pady=5

        )



        # ====================================================

        # PERSON PANEL

        # ====================================================



        person_box = ttk.LabelFrame(

            main_frame,

            text="Detected Persons",

            padding=10

        )



        person_box.pack(

            side="right",

            fill="y",

            padx=(8, 4)

        )



        # ====================================================

        # PERSON INFORMATION

        # ====================================================



        ttk.Label(

            person_box,

            text=(

                "All detected persons are\n"

                "Normal by default."

            ),

            font=(

                "Arial",

                11,

                "bold"

            ),

            justify="left"

        ).pack(

            anchor="w",

            pady=(0, 3)

        )



        ttk.Label(

            person_box,

            text=(

                "Select only the shoplifting person:"

            ),

            justify="left"

        ).pack(

            anchor="w",

            pady=(0, 8)

        )



        # ====================================================

        # PERSON LIST

        # ====================================================



        self.person_list = tk.Listbox(

            person_box,

            width=32,

            height=15,

            exportselection=False,

            font=(

                "Arial",

                11

            )

        )



        self.person_list.pack(

            fill="both",

            expand=False,

            pady=(0, 10)

        )



        # Double click also marks as Shoplifting



        self.person_list.bind(

            "<Double-Button-1>",

            self.mark_selected_shoplifting

        )



        # ====================================================

        # MARK SHOPLIFTING

        # ====================================================



        self.shoplifting_person_button = ttk.Button(

            person_box,

            text="MARK AS SHOPLIFTING",

            command=self.mark_selected_shoplifting,

            width=28

        )



        self.shoplifting_person_button.pack(

            fill="x",

            ipady=7,

            pady=(3, 7)

        )



        # ====================================================

        # SET BACK NORMAL

        # ====================================================



        self.normal_person_button = ttk.Button(

            person_box,

            text="SET BACK TO NORMAL",

            command=self.mark_selected_normal,

            width=28

        )



        self.normal_person_button.pack(

            fill="x",

            ipady=7,

            pady=(0, 8)

        )



        # ====================================================

        # SMALL LABEL INFO

        # ====================================================



        ttk.Separator(

            person_box

        ).pack(

            fill="x",

            pady=8

        )



        ttk.Label(

            person_box,

            text=(

                "Normal = default\n"

                "Shoplifting = manually selected"

            ),

            justify="left"

        ).pack(

            anchor="w",

            pady=(2, 5)

        )



        # ====================================================

        # VIDEO INFORMATION

        # ====================================================



        info_frame = ttk.Frame(

            self.root,

            padding=(12, 5)

        )



        info_frame.pack(

            fill="x"

        )



        # ====================================================

        # OPEN VIDEO

        # ====================================================



        ttk.Button(

            info_frame,

            text="OPEN VIDEO",

            command=self.open_video,

            width=18

        ).pack(

            side="left",

            padx=(0, 10),

            ipady=4

        )



        # ====================================================

        # TIME

        # ====================================================



        self.time_label = ttk.Label(

            info_frame,

            text="Time: 00:00.00",

            font=(

                "Arial",

                10,

                "bold"

            )

        )



        self.time_label.pack(

            side="left",

            padx=10

        )



        # ====================================================

        # FRAME

        # ====================================================



        self.frame_label = ttk.Label(

            info_frame,

            text="Frame: 0",

            font=(

                "Arial",

                10,

                "bold"

            )

        )



        self.frame_label.pack(

            side="left",

            padx=10

        )



        # ====================================================

        # SOURCE

        # ====================================================



        self.source_label = ttk.Label(

            info_frame,

            text="Source: None"

        )



        self.source_label.pack(

            side="left",

            padx=10

        )



        # ====================================================

        # RECORDING CONTROLS

        # ====================================================



        control_box = ttk.LabelFrame(

            self.root,

            text="Dataset Recording",

            padding=10

        )



        control_box.pack(

            fill="x",

            padx=12,

            pady=(3, 5)

        )



        # ====================================================

        # GRID COLUMNS

        # ====================================================



        for column in range(5):



            control_box.columnconfigure(

                column,

                weight=1

            )



        # ====================================================

        # START NORMAL

        # ====================================================



        self.start_normal_button = ttk.Button(

            control_box,

            text="START NORMAL",

            command=lambda:

                self.start_segment(

                    "Normal"

                )

        )



        self.start_normal_button.grid(

            row=0,

            column=0,

            sticky="ew",

            padx=5,

            ipady=8

        )



        # ====================================================

        # START SHOPLIFTING

        # ====================================================



        self.start_shoplifting_button = ttk.Button(

            control_box,

            text="START SHOPLIFTING",

            command=lambda:

                self.start_segment(

                    "Shoplifting"

                )

        )



        self.start_shoplifting_button.grid(

            row=0,

            column=1,

            sticky="ew",

            padx=5,

            ipady=8

        )



        # ====================================================

        # STOP / PAUSE

        # ====================================================



        self.stop_button = ttk.Button(

            control_box,

            text="STOP / PAUSE",

            command=self.pause_segment

        )



        self.stop_button.grid(

            row=0,

            column=2,

            sticky="ew",

            padx=5,

            ipady=8

        )



        # ====================================================

        # CONTINUE

        # ====================================================



        self.continue_button = ttk.Button(

            control_box,

            text="CONTINUE",

            command=self.continue_segment

        )



        self.continue_button.grid(

            row=0,

            column=3,

            sticky="ew",

            padx=5,

            ipady=8

        )



        # ====================================================

        # COMPLETE

        # ====================================================



        self.complete_button = ttk.Button(

            control_box,

            text="COMPLETE & SAVE",

            command=self.complete_segment

        )



        self.complete_button.grid(

            row=0,

            column=4,

            sticky="ew",

            padx=5,

            ipady=8

        )



        # ====================================================

        # STATUS

        # ====================================================



        self.status_label = ttk.Label(

            self.root,

            text="Open a video to start.",

            anchor="w",

            font=(

                "Arial",

                10,

                "bold"

            )

        )



        self.status_label.pack(

            fill="x",

            padx=15,

            pady=(3, 8)

        )





    # ========================================================

    # OPEN VIDEO

    # ========================================================



    def open_video(self):



        path = filedialog.askopenfilename(



            title="Select Video",



            filetypes=[



                (

                    "Video files",

                    "*.mp4 *.avi *.mov *.mkv *.m4v"

                ),



                (

                    "All files",

                    "*.*"

                )



            ]

        )



        if not path:

            return



        # ----------------------------------------------------

        # STOP CURRENT VIDEO LOOP

        # ----------------------------------------------------



        self.stop_playback()



        # ----------------------------------------------------

        # CLOSE ACTIVE RECORDING

        # ----------------------------------------------------



        self.force_stop_segment()



        # ----------------------------------------------------

        # RELEASE PREVIOUS VIDEO

        # ----------------------------------------------------



        if self.cap is not None:



            try:



                self.cap.release()



            except Exception:



                pass



            self.cap = None



        # ----------------------------------------------------

        # OPEN VIDEO

        # ----------------------------------------------------



        new_cap = cv2.VideoCapture(

            path

        )



        if not new_cap.isOpened():



            new_cap.release()



            messagebox.showerror(

                "Video Error",

                "Could not open this video."

            )



            return



        self.cap = new_cap



        self.video_path = Path(

            path

        )



        self.source_id = re.sub(

            r"[^A-Za-z0-9_-]+",

            "_",

            self.video_path.stem

        )



        # ----------------------------------------------------

        # FPS

        # ----------------------------------------------------



        source_fps = self.cap.get(

            cv2.CAP_PROP_FPS

        )



        if source_fps > 0:



            self.fps = source_fps



        else:



            self.fps = 25.0



        # ----------------------------------------------------

        # RESOLUTION

        # ----------------------------------------------------



        self.video_width = int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_WIDTH

            )

        )



        self.video_height = int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_HEIGHT

            )

        )



        # ----------------------------------------------------

        # TOTAL FRAMES

        # ----------------------------------------------------



        self.total_frames = int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_COUNT

            )

        )



        self.frame_index = 0



        # ----------------------------------------------------

        # RESET LABELS

        # ----------------------------------------------------



        self.person_labels.clear()



        self.current_track_ids.clear()



        self.refresh_person_list()



        # ----------------------------------------------------

        # RESET YOLO TRACKING

        # ----------------------------------------------------



        try:



            self.model.predictor = None



        except Exception:



            pass



        # ----------------------------------------------------

        # UI

        # ----------------------------------------------------



        self.source_label.config(

            text=(

                f"Source: "

                f"{self.video_path.name}"

            )

        )



        self.status_label.config(

            text=(

                f"Loaded {self.video_path.name}. "

                f"All persons default to Normal."

            )

        )



        # ----------------------------------------------------

        # SHOW FIRST FRAME

        # ----------------------------------------------------



        self.show_preview_frame()





    # ========================================================

    # PREVIEW FRAME

    # ========================================================



    def show_preview_frame(self):



        if self.cap is None:

            return



        current_position = self.cap.get(

            cv2.CAP_PROP_POS_FRAMES

        )



        ret, frame = self.cap.read()



        if not ret:

            return



        # ----------------------------------------------------

        # ORIGINAL

        # ----------------------------------------------------



        self.show_frame(

            frame,

            self.original_panel,

            "original"

        )



        # ----------------------------------------------------

        # POSE

        # ----------------------------------------------------



        try:



            result = self.model.predict(



                source=frame,



                classes=[0],



                conf=CONF_THRESHOLD,



                verbose=False



            )[0]



            annotated = result.plot()



        except Exception as e:



            print(

                f"Preview error: {e}"

            )



            annotated = frame.copy()



        self.show_frame(

            annotated,

            self.pose_panel,

            "pose"

        )



        # ----------------------------------------------------

        # RETURN POSITION

        # ----------------------------------------------------



        self.cap.set(

            cv2.CAP_PROP_POS_FRAMES,

            current_position

        )



        self.update_time_label()





    # ========================================================

    # START SEGMENT

    # ========================================================



    def start_segment(

        self,

        label

    ):



        # ----------------------------------------------------

        # CHECK VIDEO

        # ----------------------------------------------------



        if self.cap is None:



            messagebox.showwarning(

                "No Video",

                "Open a video first."

            )



            return



        # ----------------------------------------------------

        # ACTIVE SEGMENT EXISTS

        # ----------------------------------------------------



        if self.writer is not None:



            messagebox.showwarning(

                "Segment Already Active",

                (

                    f"A {self.active_label} segment "

                    f"is already active.\n\n"

                    f"Use Continue or Complete & Save first."

                )

            )



            return



        # ----------------------------------------------------

        # CHECK VIDEO END

        # ----------------------------------------------------



        current_frame = int(

            self.cap.get(

                cv2.CAP_PROP_POS_FRAMES

            )

        )



        if (

            self.total_frames > 0

            and

            current_frame >= self.total_frames

        ):



            messagebox.showinfo(

                "Video Finished",

                "The video has reached the end."

            )



            return



        # ----------------------------------------------------

        # OUTPUT PATH

        # ----------------------------------------------------



        output_path = self.get_next_output_path(

            label

        )



        # ----------------------------------------------------

        # VIDEO WRITER

        # ----------------------------------------------------



        writer = cv2.VideoWriter(



            str(

                output_path

            ),



            cv2.VideoWriter_fourcc(

                *"mp4v"

            ),



            self.fps,



            (

                self.video_width,

                self.video_height

            )

        )



        if not writer.isOpened():



            writer.release()



            messagebox.showerror(

                "Output Error",

                "Could not create output video."

            )



            return



        # ----------------------------------------------------

        # CSV PATH

        # ----------------------------------------------------



        csv_path = (

            output_path.with_suffix(

                ".csv"

            )

        )



        try:



            csv_file = open(

                csv_path,

                "w",

                newline="",

                encoding="utf-8"

            )



        except Exception as e:



            writer.release()



            messagebox.showerror(

                "CSV Error",

                f"Could not create CSV:\n\n{e}"

            )



            return



        csv_writer = csv.writer(

            csv_file

        )



        csv_writer.writerow([



            "frame_index",



            "timestamp_seconds",



            "track_id",



            "person_label",



            "confidence",



            "x1",



            "y1",



            "x2",



            "y2"



        ])



        # ----------------------------------------------------

        # SAVE STATE

        # ----------------------------------------------------



        self.writer = writer



        self.csv_file = csv_file



        self.csv_writer = csv_writer



        self.active_label = label



        self.output_path = output_path



        # ----------------------------------------------------

        # START VIDEO

        # ----------------------------------------------------



        self.running = True



        self.status_label.config(

            text=(

                f"● RECORDING {label} | "

                f"{output_path.name}"

            )

        )



        self.read_next_frame()





    # ========================================================

    # PAUSE SEGMENT

    # ========================================================



    def pause_segment(self):



        if self.writer is None:



            messagebox.showinfo(

                "No Active Segment",

                "There is no active segment."

            )



            return



        if not self.running:



            messagebox.showinfo(

                "Already Paused",

                (

                    f"{self.active_label} segment "

                    f"is already paused."

                )

            )



            return



        self.stop_playback()



        self.status_label.config(

            text=(

                f"⏸ PAUSED {self.active_label} "

                f"at frame {self.frame_index}. "

                f"Click Continue or Complete & Save."

            )

        )





    # ========================================================

    # CONTINUE SEGMENT

    # ========================================================



    def continue_segment(self):



        if self.cap is None:



            messagebox.showwarning(

                "No Video",

                "Open a video first."

            )



            return



        if self.writer is None:



            messagebox.showwarning(

                "No Active Segment",

                (

                    "No active segment exists.\n\n"

                    "Start a segment first."

                )

            )



            return



        if self.running:



            messagebox.showinfo(

                "Already Recording",

                (

                    f"{self.active_label} segment "

                    f"is already recording."

                )

            )



            return



        current_position = int(

            self.cap.get(

                cv2.CAP_PROP_POS_FRAMES

            )

        )



        if (

            self.total_frames > 0

            and

            current_position >= self.total_frames

        ):



            messagebox.showinfo(

                "Video Finished",

                (

                    "The video has reached the end. "

                    "Click Complete & Save."

                )

            )



            return



        self.running = True



        self.status_label.config(

            text=(

                f"● RECORDING {self.active_label} "

                f"| continued from frame "

                f"{self.frame_index}"

            )

        )



        self.read_next_frame()





    # ========================================================

    # COMPLETE SEGMENT

    # ========================================================



    def complete_segment(self):



        if self.writer is None:



            messagebox.showinfo(

                "No Active Segment",

                "There is no active segment to save."

            )



            return



        self.stop_playback()



        saved_path = self.output_path



        saved_label = self.active_label



        # ----------------------------------------------------

        # RELEASE WRITER

        # ----------------------------------------------------



        try:



            self.writer.release()



        except Exception as e:



            print(

                f"Video writer release error: {e}"

            )



        self.writer = None



        # ----------------------------------------------------

        # CLOSE CSV

        # ----------------------------------------------------



        if self.csv_file is not None:



            try:



                self.csv_file.flush()



                self.csv_file.close()



            except Exception as e:



                print(

                    f"CSV close error: {e}"

                )



        self.csv_file = None



        self.csv_writer = None



        self.active_label = None



        self.output_path = None



        if saved_path is not None:



            self.status_label.config(

                text=(

                    f"✓ SAVED {saved_label}: "

                    f"{saved_path.name} | "

                    f"Current frame {self.frame_index}."

                )

            )





    # ========================================================

    # FORCE STOP SEGMENT

    # ========================================================



    def force_stop_segment(self):



        if self.writer is not None:



            try:



                self.writer.release()



            except Exception:



                pass



        self.writer = None



        if self.csv_file is not None:



            try:



                self.csv_file.flush()



                self.csv_file.close()



            except Exception:



                pass



        self.csv_file = None



        self.csv_writer = None



        self.active_label = None



        self.output_path = None





    # ========================================================

    # READ NEXT FRAME

    # ========================================================



    def read_next_frame(self):



        if not self.running:

            return



        if self.cap is None:

            return



        # ----------------------------------------------------

        # READ FRAME

        # ----------------------------------------------------



        ret, frame = self.cap.read()



        if not ret:



            self.stop_playback()



            if self.writer is not None:



                self.status_label.config(

                    text=(

                        f"Video finished. "

                        f"{self.active_label} segment "

                        f"is still open. "

                        f"Click Complete & Save."

                    )

                )



            else:



                self.status_label.config(

                    text="Video finished."

                )



            return



        self.frame_index += 1



        # ====================================================

        # YOLO POSE + BYTETRACK

        # ====================================================



        try:



            result = self.model.track(



                source=frame,



                persist=True,



                tracker=TRACKER,



                classes=[0],



                conf=CONF_THRESHOLD,



                verbose=False



            )[0]



        except Exception as e:



            print(

                f"YOLO error at frame "

                f"{self.frame_index}: {e}"

            )



            self.show_frame(

                frame,

                self.original_panel,

                "original"

            )



            self.show_frame(

                frame,

                self.pose_panel,

                "pose"

            )



            if self.writer is not None:



                self.writer.write(

                    frame

                )



            self.update_time_label()



            self.schedule_next_frame()



            return



        # ====================================================

        # GET TRACK IDS

        # ====================================================



        ids = []



        if (

            result.boxes is not None

            and

            result.boxes.id is not None

        ):



            ids = [



                int(

                    track_id

                )



                for track_id

                in result.boxes.id.cpu().tolist()



            ]



        # ====================================================

        # NEW IDS DEFAULT TO NORMAL

        # ====================================================



        for tid in ids:



            if (

                tid

                not in

                self.person_labels

            ):



                self.person_labels[

                    tid

                ] = "Normal"



        self.current_track_ids = sorted(

            set(

                ids

            )

        )



        self.refresh_person_list()



        # ====================================================

        # YOLO ANNOTATED FRAME

        # ====================================================



        annotated = result.plot()



        # ====================================================

        # PERSON LABEL OVERLAY

        # ====================================================



        if (

            result.boxes is not None

            and

            result.boxes.id is not None

        ):



            boxes = (

                result.boxes.xyxy

                .cpu()

                .tolist()

            )



            track_ids = (

                result.boxes.id

                .cpu()

                .tolist()

            )



            for (

                box,

                track_id

            ) in zip(

                boxes,

                track_ids

            ):



                tid = int(

                    track_id

                )



                x1, y1, x2, y2 = map(

                    int,

                    box

                )



                label = (

                    self.person_labels.get(

                        tid,

                        "Normal"

                    )

                )



                text = (

                    f"ID {tid} | {label}"

                )



                if label == "Shoplifting":



                    color = (

                        0,

                        0,

                        255

                    )



                else:



                    color = (

                        0,

                        200,

                        0

                    )



                cv2.putText(



                    annotated,



                    text,



                    (

                        x1,

                        max(

                            25,

                            y1 - 10

                        )

                    ),



                    cv2.FONT_HERSHEY_SIMPLEX,



                    0.65,



                    color,



                    2,



                    cv2.LINE_AA



                )



        # ====================================================

        # SAVE VIDEO / CSV

        # ====================================================



        if self.writer is not None:



            # ------------------------------------------------

            # SAVE RAW FRAME

            # ------------------------------------------------



            self.writer.write(

                frame

            )



            timestamp_seconds = (



                self.frame_index



                /



                max(

                    self.fps,

                    1

                )



            )



            # ------------------------------------------------

            # CSV

            # ------------------------------------------------



            if (

                self.csv_writer is not None

                and

                result.boxes is not None

                and

                result.boxes.id is not None

            ):



                boxes = (

                    result.boxes.xyxy

                    .cpu()

                    .tolist()

                )



                track_ids = (

                    result.boxes.id

                    .cpu()

                    .tolist()

                )



                confidences = (

                    result.boxes.conf

                    .cpu()

                    .tolist()

                )



                for (

                    box,

                    track_id,

                    confidence

                ) in zip(

                    boxes,

                    track_ids,

                    confidences

                ):



                    tid = int(

                        track_id

                    )



                    x1, y1, x2, y2 = map(

                        float,

                        box

                    )



                    person_label = (

                        self.person_labels.get(

                            tid,

                            "Normal"

                        )

                    )



                    self.csv_writer.writerow([



                        self.frame_index,



                        round(

                            timestamp_seconds,

                            3

                        ),



                        tid,



                        person_label,



                        round(

                            float(

                                confidence

                            ),

                            4

                        ),



                        round(

                            x1,

                            2

                        ),



                        round(

                            y1,

                            2

                        ),



                        round(

                            x2,

                            2

                        ),



                        round(

                            y2,

                            2

                        )



                    ])



        # ====================================================

        # DISPLAY

        # ====================================================



        self.show_frame(

            frame,

            self.original_panel,

            "original"

        )



        self.show_frame(

            annotated,

            self.pose_panel,

            "pose"

        )



        self.update_time_label()



        self.schedule_next_frame()





    # ========================================================

    # SCHEDULE NEXT FRAME

    # ========================================================



    def schedule_next_frame(self):



        if not self.running:

            return



        delay_ms = max(



            1,



            int(

                1000

                /

                max(

                    self.fps,

                    1

                )

            )



        )



        self.after_id = self.root.after(

            delay_ms,

            self.read_next_frame

        )





    # ========================================================

    # STOP VIDEO LOOP

    # ========================================================



    def stop_playback(self):



        self.running = False



        if self.after_id is not None:



            try:



                self.root.after_cancel(

                    self.after_id

                )



            except tk.TclError:



                pass



            self.after_id = None





    # ========================================================

    # SHOW FRAME

    # ========================================================



    def show_frame(

        self,

        frame,

        panel,

        panel_type="original"

    ):



        if frame is None:

            return



        if frame.size == 0:

            return



        # ----------------------------------------------------

        # BGR -> RGB

        # ----------------------------------------------------



        try:



            frame_rgb = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB

            )



        except cv2.error:



            return



        height, width = (

            frame_rgb.shape[:2]

        )



        if (

            width <= 0

            or

            height <= 0

        ):



            return



        # ----------------------------------------------------

        # KEEP ASPECT RATIO

        # ----------------------------------------------------



        scale = min(



            DISPLAY_WIDTH

            /

            width,



            DISPLAY_HEIGHT

            /

            height



        )



        new_width = max(

            1,

            int(

                width * scale

            )

        )



        new_height = max(

            1,

            int(

                height * scale

            )

        )



        frame_rgb = cv2.resize(



            frame_rgb,



            (

                new_width,

                new_height

            ),



            interpolation=cv2.INTER_AREA



        )



        image = Image.fromarray(

            frame_rgb

        )



        photo = ImageTk.PhotoImage(

            image=image

        )



        if panel_type == "original":



            self.original_photo = photo



        else:



            self.pose_photo = photo



        panel.configure(

            image=photo

        )





    # ========================================================

    # UPDATE TIME

    # ========================================================



    def update_time_label(self):



        seconds = (



            self.frame_index



            /



            max(

                self.fps,

                1

            )



        )



        minutes = int(

            seconds // 60

        )



        remaining_seconds = (

            seconds % 60

        )



        self.time_label.config(

            text=(

                f"Time: "

                f"{minutes:02d}:"

                f"{remaining_seconds:05.2f}"

            )

        )



        if self.total_frames > 0:



            self.frame_label.config(

                text=(

                    f"Frame: "

                    f"{self.frame_index}"

                    f"/"

                    f"{self.total_frames}"

                )

            )



        else:



            self.frame_label.config(

                text=(

                    f"Frame: "

                    f"{self.frame_index}"

                )

            )





    # ========================================================

    # REFRESH PERSON LIST

    # ========================================================



    def refresh_person_list(self):



        selected_id = (

            self.get_selected_track_id()

        )



        self.person_list.delete(

            0,

            tk.END

        )



        for tid in self.current_track_ids:



            label = (

                self.person_labels.get(

                    tid,

                    "Normal"

                )

            )



            self.person_list.insert(

                tk.END,

                f"Person ID {tid} | {label}"

            )



        # ----------------------------------------------------

        # KEEP SELECTION

        # ----------------------------------------------------



        if (

            selected_id

            in

            self.current_track_ids

        ):



            index = (

                self.current_track_ids.index(

                    selected_id

                )

            )



            self.person_list.selection_set(

                index

            )



            self.person_list.activate(

                index

            )





    # ========================================================

    # GET SELECTED TRACK ID

    # ========================================================



    def get_selected_track_id(self):



        selection = (

            self.person_list.curselection()

        )



        if not selection:



            return None



        index = selection[0]



        displayed = (

            self.person_list.get(

                index

            )

        )



        match = re.search(

            r"Person ID (\d+)",

            displayed

        )



        if match:



            return int(

                match.group(1)

            )



        return None





    # ========================================================

    # MARK SELECTED AS SHOPLIFTING

    # ========================================================



    def mark_selected_shoplifting(

        self,

        event=None

    ):



        tid = (

            self.get_selected_track_id()

        )



        if tid is None:



            if event is None:



                messagebox.showwarning(

                    "Select Person",

                    "Select a detected person first."

                )



            return



        self.person_labels[

            tid

        ] = "Shoplifting"



        self.refresh_person_list()



        self.status_label.config(

            text=(

                f"Person ID {tid} "

                f"marked as Shoplifting."

            )

        )





    # ========================================================

    # SET SELECTED BACK TO NORMAL

    # ========================================================



    def mark_selected_normal(self):



        tid = (

            self.get_selected_track_id()

        )



        if tid is None:



            messagebox.showwarning(

                "Select Person",

                "Select a detected person first."

            )



            return



        self.person_labels[

            tid

        ] = "Normal"



        self.refresh_person_list()



        self.status_label.config(

            text=(

                f"Person ID {tid} "

                f"set to Normal."

            )

        )





    # ========================================================

    # CREATE OUTPUT PATH

    # ========================================================



    def get_next_output_path(

        self,

        label

    ):



        if label == "Normal":



            folder = NORMAL_DIR



        else:



            folder = SHOPLIFTING_DIR



        number = 1



        while True:



            filename = (



                f"{self.source_id}_"



                f"{label.lower()}_"



                f"{number:03d}.mp4"



            )



            candidate = (

                folder

                /

                filename

            )



            if not candidate.exists():



                return candidate



            number += 1





    # ========================================================

    # CLOSE APP

    # ========================================================



    def close_app(self):



        self.stop_playback()



        self.force_stop_segment()



        if self.cap is not None:



            try:



                self.cap.release()



            except Exception:



                pass



            self.cap = None



        self.root.destroy()





# ============================================================

# MAIN

# ============================================================



if __name__ == "__main__":



    root = tk.Tk()



    app = ShopliftingApp(

        root

    )



    root.mainloop()