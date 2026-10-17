# UrbanPulse — Implementation Plan for Claude / Antigravity

**Project:** UrbanPulse — AI-Based Crowd Safety and Abnormal Movement Detection  
**Purpose of this document:** A build specification for an AI coding agent. Read the existing notebook first, preserve valid work, correct the identified methodology issues, then build a working end-to-end project.

---

## 1. Instructions to the coding agent

You are continuing an existing student project, not starting from zero.

1. Inspect the uploaded/current notebook `Untitled0.ipynb` and any repository files before writing code. Map existing work to the status table in this document.
2. Reuse working code and artifacts. Do not duplicate the notebook pipeline into a second, conflicting implementation.
3. Use **Ponytail** for the coding workflow: follow its minimal-change/reuse-first principles. In Antigravity, ensure its rules are loaded through the supported Ponytail/`AGENTS.md` integration. Official repository: https://github.com/DietrichGebert/ponytail
4. Before building new abstractions or adding dependencies, check whether the current code, Python standard library, existing packages, or native platform features already solve the problem.
5. Keep data processing reproducible, tests meaningful, and code understandable to a student. Do not remove safety, validation, error handling, or evaluation merely to reduce code.
6. Do not claim the system detects a person's mental state or “panic.” The project detects **observable abnormal crowd movement** and flags a **potential crowd-safety event**.
7. Clearly label provisional research results as provisional. Do not report impressive metrics as final until labels, split strategy, and evaluation have been revalidated.
8. Use Google Colab (free GPU when available) for video/model processing if needed. Keep the final app runnable on the user's Windows laptop without a dedicated GPU by allowing CPU inference or precomputed demo results.

### Ponytail setup / use

Ponytail is an AI coding-agent ruleset, not a Python runtime dependency of UrbanPulse. Do not add it to `requirements.txt`.

- Official source: https://github.com/DietrichGebert/ponytail
- For Antigravity, use the repository's supported `AGENTS.md` / `.agents/rules/` instructions or its documented Antigravity integration. Confirm the instructions are actually loaded before implementing.
- If the environment supports Ponytail commands/skills, use `/ponytail` in `full` mode while implementing and `/ponytail-review` before finalizing. If these commands are unavailable in the current Antigravity environment, use the official repository's Antigravity-compatible rules and state that limitation rather than pretending the command ran.
- Use Ponytail's reuse-first/YAGNI approach: inspect before changing, reuse dependencies, avoid unnecessary frameworks and abstraction, and report what was omitted and what was not tested.

---

## 2. Ultimate goal

Build a credible, demonstrable college-level deep-learning system that takes a crowd video, detects and tracks people, estimates crowd-level movement, classifies temporal crowd behavior as **Normal** or **Abnormal**, and presents an understandable alert in a dashboard.

The finished demo should:
- accept a video upload or select a bundled sample video;
- display the video with person boxes and tracking IDs;
- show current/estimated person count and crowd movement features;
- show a time-series anomaly score and the predicted status;
- raise and clear a debounced warning when abnormal movement persists;
- show event time/frame and explain that the result is a model-generated warning, not a confirmed emergency;
- include a model evaluation page/section with honest test metrics and known limitations;
- run reliably from a documented setup command.

**Target is a research prototype, not a certified life-safety product.** Do not advertise guaranteed early detection, exact physical speed, real-world deployment readiness, or universal generalization.

---

## 3. Agreed architecture

```text
Video / CCTV recording
        |
        v
YOLO11n person detection (class 0)
        |
        v
ByteTrack tracking (temporary track IDs)
        |
        v
Per-person positions and movement
        |
        v
Per-frame crowd features
(count, robust speed statistics, direction/flow,
acceleration and count-change features)
        |
        v
Fixed-length temporal windows
        |
        v
LSTM baseline (GRU comparison optional)
        |
        v
Abnormal probability / score
        |
        v
Temporal alert smoothing (persistence / clear rule)
        |
        v
Dashboard: video + count + score + status + timeline
```

Keep **overcrowding** and **abnormal movement** as distinct concepts. A high count alone is not evidence of abnormal movement. If there is no calibrated site-specific occupancy threshold, show the count and label the rule as configurable rather than claiming an objectively safe/unsafe density.

---

## 4. Dataset plan and direct links

Use datasets according to their intended role. Do not merge image-only and video datasets as if they supply the same labels.

### Dataset A — UMN Unusual Crowd Activity (primary dataset)

- Official project page: https://www.crcv.ucf.edu/projects/Abnormal_Crowd/
- Current combined AVI used in the notebook: https://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi
- Role: the primary supervised proof-of-concept for temporal normal-to-abnormal crowd movement.
- Official description: 11 videos across 3 scenes; each video has a normal starting section and an abnormal ending section.
- Previously observed video metadata: 7,739 frames, 30 FPS, 320×240, about 257.97 seconds.
- Current notebook processing produced 84,692 person detections and 954 distinct tracker IDs in that combined-file run. Tracker IDs can be reassigned or fragmented, so do not interpret 954 as the true number of unique people.
- Current notebook has manually reviewed/provisional abnormal intervals and approximate clip starts. Treat these as **working annotations, not authoritative ground truth** until validated and documented.

**Important:** the combined AVI concatenates multiple sequences. Do not label the whole file with one global normal-to-abnormal transition. Each sequence must have its own boundaries and abnormal onset. A boundary or sudden drop in detected count is not sufficient proof of a clip boundary or an anomaly.

### Dataset B — ShanghaiTech Campus (secondary video dataset)

- Official dataset page: https://svip-lab.github.io/dataset/campus_dataset.html
- Role: later-stage external evaluation/generalization check on surveillance anomaly footage, if its annotations and video format can be incorporated in the available time.
- Official page describes 13 scenes, complex lighting/camera angles, 130 abnormal events, more than 270,000 training frames, and pixel-level anomaly ground truth.
- Do not assume UMN labels or frame boundaries apply to ShanghaiTech. Build a dataset-specific loader and annotation mapping.
- Start by checking download/access requirements, folder structure, frame rate, annotation format, and license/usage terms. Do not download it automatically into the repository or commit large data files.

### Dataset C — UCF-QNRF (optional crowd-counting/density dataset)

- Official dataset page: https://www.crcv.ucf.edu/research/data-sets/ucf-qnrf/
- Role: optional enhancement/validation for dense crowd counting; **not** the main temporal anomaly dataset.
- Official page describes 1,535 images (1,201 train / 334 test) and 1,251,642 person annotations.
- It consists of images and point annotations, not temporal video sequences. It can support a separate density/counting model or a future benchmark, but it cannot directly train the LSTM temporal abnormal-movement classifier.
- Optional third-party preprocessed source previously discussed: https://github.com/AdamHermes/GigaCount and https://drive.google.com/drive/folders/1X3_MApVSmkRRjHx_a5Tlnv-DUXNWOYnA?usp=sharing. Prefer the official source when feasible; inspect provenance, completeness, and license before using third-party copies.

**Recommended priority:** UMN first; finish the data/labeling/model/evaluation loop. Add ShanghaiTech only after the UMN pipeline is methodologically sound. Treat UCF-QNRF as optional because it solves a related but different task.

---

## 5. What exists now (based on the provided notebook)

### Completed / present in notebook
- Installed and used Ultralytics, OpenCV, pandas, NumPy, Matplotlib and TensorFlow/Keras.
- Loaded `yolo11n.pt`; verified person detection on a sample image.
- Downloaded and tested a short `people.mp4` clip.
- Ran YOLO person tracking with ByteTrack and rendered tracking IDs.
- Downloaded `Crowd-Activity-All.avi` and inspected metadata.
- Processed the full UMN combined video with YOLO11n + ByteTrack.
- Extracted person-level frame, track ID, center coordinates and confidence.
- Calculated displacement, pixel-per-second speed and movement direction.
- Aggregated frame-level features: `person_count`, `avg_speed`, `median_speed`, `max_speed`, `speed_std`, `avg_direction`, `direction_std`.
- Created crowd count and average-speed plots.
- Reindexed features to all 7,739 frames and created a labeled CSV using manually reviewed provisional intervals.
- Created approximate boundaries for 11 clips and windowed the features into 30-frame sequences with stride 5.
- Trained an initial LSTM classifier, selected a threshold using validation predictions, and implemented a persistence/debounce rule for alerts.
- Saved artifacts in the notebook: `urbanpulse_lstm.keras`, `urbanpulse_scaler.joblib`, `urbanpulse_config.json`.

### Current experimental results are NOT yet final
The notebook shows encouraging metrics from the provisional run, but labels and boundaries were still being adjusted manually. In particular, the notebook updates the label for a held-out interval after the model predictions were produced, then recalculates metrics using those unchanged predictions. Those post-hoc metrics are **not a valid final evaluation**. Rebuild the annotations, retrain the model, reselect the threshold on validation only, and rerun all test metrics after the annotation freeze.

### Known provisional annotation details
The notebook currently uses approximate clip starts:
`[0, 625, 1453, 1986, 2685, 3429, 4035, 4929, 5597, 6235, 6913]`

It contains provisional abnormal intervals, including:
`(510,614), (1330,1439), (1830,1985), (2610,2684), (3240,3428), (3960,4017), (4780,4805), (5430,5595), (6210,6234), (6845,6912), (7710,7719)`

These values are evidence of work already performed, **not approved ground truth**. Some later notebook cells explicitly mark some event onsets unresolved. The coding agent must reconcile the latest notebook cells, verify each boundary/onset from the source video and official dataset structure, and create a versioned annotation file with notes and confidence. Do not silently reuse the first interval list or assume every one of the 11 intervals is correct.

---

## 6. Implementation phases

### Phase 0 — Inspect and preserve the existing work
1. Open `Untitled0.ipynb` and inspect the latest cells, not only early cells.
2. Inventory current artifacts and where they are saved in Colab/Drive.
3. Create a project README describing existing work, limitations, and reproducible steps.
4. Avoid rerunning the expensive full-video detector if the feature CSV and tracking output can be recovered.
5. Keep the original notebook unchanged as a record; create a cleaned notebook or Python modules for the reproducible pipeline.

**Acceptance:** agent reports what it found, which files/artifacts exist, and which steps can be reused.

### Phase 1 — Establish a clean project structure
Use a small structure; do not overengineer.

```text
UrbanPulse/
├── AGENTS.md                  # Ponytail/agent instructions, if needed
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_umndata_inspection.ipynb
│   ├── 02_extract_tracking_features.ipynb
│   ├── 03_label_and_validate.ipynb
│   ├── 04_train_evaluate_lstm.ipynb
│   └── 05_demo_inference.ipynb
├── src/
│   ├── video_features.py
│   ├── annotations.py
│   ├── sequences.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── app/
│   └── app.py
├── models/                    # model/scaler/config; large artifacts excluded from Git if appropriate
├── data/
│   └── README.md              # dataset URLs and download/setup instructions; don't commit raw datasets
├── outputs/
│   └── README.md
└── tests/
    ├── test_annotations.py
    ├── test_sequences.py
    └── test_alert_logic.py
```

The exact app framework can be chosen after inspecting existing code. Prefer the simplest framework already installed or the one that gives a reliable demo without unnecessary complexity.

### Phase 2 — Validate the video and sequence boundaries
1. Record metadata (frame count, FPS, resolution, duration) using OpenCV.
2. Determine the 11 constituent sequence boundaries from the source file/official data structure and frame-by-frame visual inspection.
3. Save a table such as `data/umn_annotations.csv` with:
   - `clip_id`, `start_frame`, `end_frame`, `scene`,
   - `abnormal_onset_frame`,
   - `onset_source` (`official`, `visually_reviewed`, or `unresolved`),
   - `confidence` and `notes`.
4. Verify that boundaries cover the full frame range without gaps or overlaps.
5. Review each onset with contact sheets around the transition. Do not use count drops, speed spikes, or overlay detection alone as ground truth.
6. If an onset is uncertain, mark it unresolved and either exclude that clip from supervised evaluation or clearly document the uncertainty; do not invent a precise label.

**Acceptance:** annotation CSV is versioned, boundary checks pass, each onset has a source and confidence, and unresolved cases are visible.

### Phase 3 — Improve feature extraction robustly
Reuse YOLO11n and ByteTrack. Avoid reprocessing the full video unless required.

Per detection: `frame`, `clip_id`, `track_id`, `center_x`, `center_y`, `confidence`, bounding-box width/height if useful.

Per-person movement:
- calculate displacement only between observations from the same track within the same clip;
- if a track has a frame gap, do not divide its full displacement by one-frame time; use actual frame difference and optionally reject excessively long gaps;
- keep pixel speed explicitly in `pixels/second`, not m/s or km/h;
- avoid speed spikes from tracker switches using robust clipping/winsorization based only on training data or robust per-frame statistics;
- handle direction as circular data: average raw degrees can be misleading around -180°/180°. Prefer sine/cosine flow components or circular statistics;
- calculate acceleration or change in speed only when valid consecutive measurements exist.

Per-frame features:
- `person_count`
- robust central speed (median/trimmed mean)
- speed dispersion (IQR or standard deviation)
- fraction of tracks with high relative speed, if justified by training distribution
- directional/flow consistency using circular-safe features
- `count_delta` and rolling count-change features
- optional acceleration statistics
- optional number of newly appearing/disappearing tracks

Missing frames:
- reindex to every frame within each clip;
- distinguish genuine zero detections from processing failures if possible;
- use documented imputation and missingness flags rather than automatically interpreting all missing feature rows as an empty scene.

**Acceptance:** feature table has one row per frame per clip, no cross-clip track differencing, no NaNs/infinities, and feature definitions are documented.

### Phase 4 — Freeze labels before training
1. Rebuild labels from the reviewed annotation CSV, independently for each clip.
2. Label frames before onset as normal (`0`) and the abnormal interval after onset as abnormal (`1`), following the dataset's sequence semantics and the final reviewed onset.
3. Do not label based on the model's own speed thresholds or predictions.
4. Save `umn_crowd_features_labeled.csv` and a label-distribution report.
5. Plot count/movement features with annotation bands to visually audit labels.
6. Freeze a version (for example, `annotation_version: 1.0`) before final training. If labels change later, retrain and rerun the entire evaluation.

**Acceptance:** label file and annotation file agree; a label change automatically invalidates model metrics until retraining/evaluation is rerun.

### Phase 5 — Sequence windows and leakage-safe split
- Use 30 frames (1 second at 30 FPS) as the initial window length and stride 5 as a starting point; make them configurable.
- Each sample uses only frames from one clip. Never allow a window to cross a clip boundary.
- Keep the current feature-scaling principle: fit `StandardScaler` on training clips only; apply it to validation/test.
- Split by entire clips/scenes, not randomly by overlapping windows. A reasonable initial split is the current intended split (`train=[1,3,5,7,8,11]`, `validation=[4,9]`, `test=[2,6,10]`) only after boundaries are verified. Document scene representation and the limitations of 11 clips.
- Avoid tuning the threshold or features using the held-out test set.
- Store split IDs in config for reproducibility.

**Acceptance:** no clip overlap across train/validation/test; no window crosses a clip boundary; scaler is fitted on train data only.

### Phase 6 — Train and evaluate the temporal model
- Keep LSTM as the first baseline. A small GRU comparison is optional and should only be added if it offers useful evidence.
- Start with the existing compact architecture (LSTM(32) → Dense(16, ReLU) → Dropout → sigmoid output), early stopping on validation PR-AUC, fixed random seed, and class weighting.
- Save training history, scaler, feature list, window size, stride, threshold, clip split and annotation version alongside the model.
- Select the operating threshold using validation data only (for example, maximize validation F1 or choose a precision/recall trade-off relevant to early warnings).
- Recompute final test predictions only after labels are frozen and model retrained.
- Report: confusion matrix, per-class precision/recall/F1, PR-AUC, ROC-AUC where both classes exist, and alert/event-level false alarms and missed events.
- Include per-clip results and a plot of probability vs. ground truth.
- Because overlapping windows are correlated and there are few source clips, avoid presenting window-level metrics as if every window were an independent real-world trial.
- Compare a simple baseline (e.g. logistic regression on window aggregates or a simple rule baseline) if time allows; this helps show whether LSTM adds value.

**Acceptance:** metrics can be regenerated by one documented command/notebook from frozen labels; test labels were never used for model or threshold selection.

### Phase 7 — Alert logic
Implement a simple, configurable persistence rule:
- raise alert only after a configurable number of consecutive positive windows;
- clear alert after a configurable number of consecutive negative windows;
- avoid resetting the state at every video frame; operate on prediction windows and preserve timestamps;
- emit event start/end frame/time and peak anomaly score;
- do not create an alert for an isolated one-window spike unless the user explicitly configures that mode.
- distinguish `NORMAL`, `ABNORMAL MOVEMENT`, `HIGH COUNT` (if an occupancy threshold is configured), and `INSUFFICIENT DETECTIONS/LOW CONFIDENCE`.

**Acceptance:** unit tests cover persistent alert, clearing, single spike, clip boundary reset and empty/invalid video.

### Phase 8 — Dashboard and inference
Build a clean but modest dashboard. Use an existing framework if the repo already has one; otherwise prefer a minimal Python app (for example, Streamlit) over a custom frontend/backend stack unless there is a clear reason.

Required interface:
- title: UrbanPulse — AI-Based Crowd Safety Monitoring;
- upload/select a video;
- video playback with person boxes and ByteTrack IDs (pre-rendered or frame-by-frame inference, based on feasible runtime);
- current count and movement status;
- anomaly probability/score and chosen threshold;
- status banner and event timeline;
- crowd-count and anomaly-score plots over time;
- table of detected alert intervals with frame and timestamp;
- clear “prototype / not an emergency service” note;
- show processing progress and useful errors for unsupported videos/missing model files.

For a first demo, **offline video upload + processed playback is acceptable**. Real-time CCTV streaming is a stretch goal, not a requirement. The app should not imply it is processing live CCTV if it is actually showing precomputed results.

**Acceptance:** a user can run the app from the README, upload a sample video, see the result and understand when/why an alert was raised.

### Phase 9 — Test, package and document
- Add small tests for annotation coverage, sequence window creation, scaler/model input shape, and alert debouncing.
- Run inference on at least one held-out UMN clip and one non-UMN sample if legally/technically available.
- Save a demo video or screenshots and a concise model card describing dataset, labels, split, metrics, limitations and intended use.
- Document all commands for local CPU use and Colab GPU processing.
- Do not commit large raw datasets, temporary videos, model checkpoints, secrets, or personal file paths. Use `.gitignore` and a `data/README.md`.
- Provide setup instructions, dependency versions and known troubleshooting notes.

---

## 7. Recommended technical choices

- **Person detector/tracker:** Ultralytics YOLO11n + ByteTrack (already working).
- **Feature processing:** Python, OpenCV, NumPy, pandas.
- **Model:** TensorFlow/Keras LSTM baseline; GRU optional.
- **Scaling/metrics:** scikit-learn.
- **Dashboard:** Streamlit if no dashboard exists yet; reuse current framework if one exists.
- **Compute:** Google Colab for heavy extraction/training; local Windows laptop for code/dashboard and CPU inference.
- **Artifacts:** `.keras` model, `joblib` scaler, JSON config, annotation CSV, evaluation report.
- **Reproducibility:** fixed random seeds, config file, pinned dependencies where practical.

Do not train a crowd-count model from UCF-QNRF as part of the first milestone. YOLO person counts are adequate for the current sparse UMN prototype; dense crowd count may require a separate density-estimation approach later.

---

## 8. Current and planned changes — with reasons

| Change | Why |
|---|---|
| Verify the 11 clip boundaries | The AVI is concatenated; one global label transition would be incorrect. |
| Replace informal interval assumptions with a reviewed annotation CSV | Keeps labels auditable, versioned, and reproducible. |
| Rebuild and freeze labels before training | Prevents post-hoc label changes from invalidating evaluation. |
| Retrain after the last label change | The current notebook recalculates metrics after changing a test label without regenerating model predictions; those metrics are not a valid final result. |
| Avoid using speed spikes as labels | Tracking noise, perspective, and missed detections can create spikes. |
| Prevent cross-clip track differencing | Tracker IDs can repeat/restart; movement must not be computed across sequence boundaries. |
| Use robust speed/direction features | Raw max speed and arithmetic direction means can be unstable. |
| Keep clips separated across splits | Overlapping windows from the same clip can cause data leakage and overoptimistic results. |
| Fit scaler on training clips only | Prevents test/validation information leaking into preprocessing. |
| Add persistence/debounce to alerts | Reduces flickering alerts from isolated positive windows. |
| Keep overcrowding separate from abnormal movement | High person count and unusual movement are different signals. |
| Add ShanghaiTech only after UMN is validated | Prevents scope creep and gives a meaningful external evaluation target. |
| Keep UCF-QNRF optional | It is an image crowd-counting dataset, not a temporal anomaly dataset. |
| Build offline upload demo first | More feasible and reliable on a laptop without a dedicated GPU than true live CCTV streaming. |
| Use Ponytail/reuse-first development | Avoids unnecessary libraries, architecture and code while preserving validation and safety. |

---

## 9. Repository deliverables

By completion, the repository should contain:
- readable, reusable code for feature extraction, annotation, windowing, training, evaluation and inference;
- a verified annotation CSV with unresolved items explicitly marked;
- reproducible feature and training pipeline;
- saved model/scaler/config;
- final evaluation report and plots;
- working dashboard;
- README and data documentation;
- tests for key correctness conditions;
- demo video or screenshots.

Raw datasets should not be committed to GitHub. Store dataset files in Colab/Drive or a local data directory and provide download/setup instructions.

---

## 10. Definition of done

The project is complete when all of the following are true:

- [ ] The coding agent has inspected and preserved the existing notebook work.
- [ ] Ponytail rules were loaded or the integration limitation was clearly reported.
- [ ] UMN clip boundaries and abnormal onsets are reviewed, versioned and documented.
- [ ] Features and labels are rebuilt from the frozen annotations.
- [ ] The LSTM is retrained after the final annotation version.
- [ ] Train/validation/test splits are clip-based and leakage-safe.
- [ ] Final metrics are regenerated from the final model and final test labels.
- [ ] Alert logic has tests and does not flicker on isolated spikes.
- [ ] The dashboard runs and demonstrates a video end-to-end.
- [ ] README setup steps work on the target environment.
- [ ] Limitations are documented and no claims of guaranteed panic/emergency detection are made.

---

## 11. Dataset and tool references

- UMN abnormal crowd project: https://www.crcv.ucf.edu/projects/Abnormal_Crowd/
- UMN combined AVI currently used: https://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi
- UCF-QNRF official page: https://www.crcv.ucf.edu/research/data-sets/ucf-qnrf/
- ShanghaiTech Campus official page: https://svip-lab.github.io/dataset/campus_dataset.html
- Ponytail official repository: https://github.com/DietrichGebert/ponytail

**Final instruction:** prioritize a correct, reproducible, modest working system over adding more models or datasets. Make the existing pipeline trustworthy first; then polish the dashboard and demo.
