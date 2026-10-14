# Technical Report: Multi-Modal Person Re-ID & Face Association Pipeline

This document outlines the architecture, components, and workflow of our Multi-Modal Person Re-Identification (Re-ID) and Face Association system. It is designed to track individuals across a supermarket environment using both body features and facial recognition, resolving identities dynamically.

---

## 1. System Architecture

The system is built as a multi-modal pipeline that processes video feeds, extracts body and face features, and maintains a persistent database of identities across multiple cameras.

### 1.1 Core Modalities
*   **Body Re-ID:** Tracks individuals based on clothing, body shape, and overall appearance. It is robust to distance and angle but susceptible to similar clothing (e.g., uniforms).
*   **Face Recognition:** Highly accurate identity verification, used as the "ground truth" when a clear face is visible.

### 1.2 Storage & Infrastructure
*   **MongoDB (Atlas):** Acts as the central database storing metadata, sighting histories (visit durations, timestamps, camera IDs), and links to images.
*   **Pinecone (Vector Database):** Stores embeddings (mathematical representations) for fast similarity searches.
    *   `supermarket-reid`: Stores body prototypes (1280-dimensional).
    *   `supermarket-faces`: Stores face embeddings (512-dimensional).
*   **ImageKit (CDN):** Hosts the physical image crops (body and face chips) to offload storage from the local machine and provide fast URLs for the UI.

---

## 2. AI Models & Components

The pipeline utilizes several specialized deep learning models:

1.  **RT-DETR (Real-Time DEtection TRansformer):** Detects people in the frame with high accuracy.
2.  **YOLOv8-Pose:** Estimates the human skeleton (keypoints). We use this as a "Quality Gate" to ensure we only extract features from clear, visible bodies (e.g., rejecting crops where only an arm is visible).
3.  **BotSort:** A robust multi-object tracker that assigns a temporary `track_id` to a person as long as they remain continuously visible in the camera view.
4.  **CLIP-ReID (Market1501):** Extracts a 1280-D feature vector representing the person's overall appearance.
5.  **InsightFace (SCRFD):** A lightweight, high-performance face detection model that locates faces within the frame.
6.  **InsightFace (ArcFace):** Extracts a 512-D feature vector representing a person's facial identity. Highly accurate, even with partial occlusions.

---

## 3. The Processing Workflow (Frame-by-Frame)

When a video is processed, the system follows this logic:

### Step 1: Detection & Tracking
For every frame, RT-DETR finds people, and BotSort assigns them a temporary track ID (e.g., `tid = 5`).

### Step 2: Quality Filtering
We extract the bounding box crop of the person. YOLOv8-Pose checks how much of the skeleton is visible. If the visibility ratio is too low, the crop is ignored to prevent garbage data from entering the system.

### Step 3: Multi-Modal Feature Extraction
*   **Body:** CLIP-ReID generates a body embedding.
*   **Face:** If the camera is a designated "Face Camera", InsightFace scans the frame. If a face is found inside a person's bounding box (specifically in the top 40%), it extracts the face embedding.

### Step 4: Identity Resolution (The Core Logic)
The system must map the temporary `track_id` to a persistent `global_id` (GID).

1.  **Face Ground Truth (VIP Bypass):** If a face is detected, we query the Pinecone `supermarket-faces` index.
    *   If the face matches an existing identity (e.g., GID 29), we immediately force this track to be GID 29.
    *   If the system previously thought this track was GID 26 based on body features, an **Identity Merge** is triggered. GID 26 is folded into GID 29, and all history is updated.
2.  **Gallery Matching (Body):** If no face is recognized, the body embedding is compared against our `GalleryManager` (a self-adaptive memory of known bodies).
    *   It uses a mathematical concept called Softmax to calculate confidence.
    *   If confidence is high, the track is assigned to the matched GID.
3.  **Unknown Data Manager (UDM):** If the system is unsure (ambiguous match or low confidence), the crop is sent to the UDM.
    *   The UDM pools these unknown crops. Every 30 frames, it runs **DBSCAN clustering**.
    *   If a cluster of similar unknown crops forms, the system "mints" a brand new Global ID for them.

### Step 5: Sighting Tracking
As a person moves through the frame, the `SightingTracker` records their entry time, exit time, and keeps track of the absolute best body crop and face crop captured during that specific visit.

---

## 4. Saving & Indexing

When the video finishes (or the person leaves the camera), the system saves the data:
1.  **CDN Upload:** The best body and face crops are uploaded to ImageKit.
2.  **Face Indexing:** If a new face was found, its embedding and ImageKit URL are pushed to the Pinecone face index.
3.  **MongoDB Update:** The person's document is updated with the new sighting history (duration, camera ID, URLs).

---

## 5. The Search Engine

The system supports searching the database using an image.

### Body Search ($\alpha$-FPS)
Because a person might have hundreds of body crops, we don't store them all in Pinecone. We use an algorithm called **$\alpha$-FPS (Adaptive Farthest Point Sampling)**.
*   It analyzes all crops for a person and selects the **Top 3 Prototypes** that maximize both *image quality* (clear skeleton) and *visual diversity* (e.g., front view, back view, side view).
*   Only these 3 master prototypes are stored in Pinecone for body searches.

### Face Search
Face embeddings are highly consistent, so we can search the Pinecone face index directly.
*   *Note on preprocessing:* If a user uploads a tiny, pre-cropped face, the system pads it onto a larger gray canvas before feeding it to InsightFace. This prevents the detector from failing due to lack of background context.

---

## 6. Interactive Dashboard

We built a Gradio-based web UI that encapsulates the entire pipeline.
*   **Video Processing Tab:** Allows users to upload a video, tweak all thresholds (confidence, gallery budget, clustering strictness), and run the pipeline live with progress tracking.
*   **Search Tab:** Upload an image to search by face, body, or both simultaneously.
*   **Database Browser:** View all recognized individuals, their visit logs, and their associated body/face galleries.
