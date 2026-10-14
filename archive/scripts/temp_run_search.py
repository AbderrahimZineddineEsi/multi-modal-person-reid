def run_search(image, image_url, search_mode, is_face_crop, body_threshold, face_threshold, tau):

    if image is None and not image_url:

        return "Please upload an image or provide a URL.", None, ""

    

    if image is None and image_url:

        try:

            import urllib.request

            req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})

            resp = urllib.request.urlopen(req)

            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)

            image = cv2.imdecode(arr, -1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except Exception as e:

            return f"Failed to load image from URL: {e}", None, ""



    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    body_result, face_result = None, None

    body_text, face_text = "", ""

    display_img = None



    if search_mode in ["Body", "Both"]:

        body_result = search_body(img_bgr, th_emb=body_threshold, tau=tau)

        if body_result["status"] == "MATCH":

            body_text = f"**Body Match: GID {body_result['global_id']}** (Score: {body_result['score']:.3f})"

            doc = body_result.get("mongodb_data", {})

            for s in doc.get("sightings", []):

                if s.get("best_crop_url"):

                    display_img = fetch_image_from_url(s["best_crop_url"], as_rgb=True); break

        else:

            body_text = f"Body: {body_result['status']} — {body_result.get('reason', '')}"



    if search_mode in ["Face", "Both"]:

        face_result = search_face_query(img_bgr, is_crop=is_face_crop, th_face=face_threshold)

        if face_result["status"] == "MATCH":

            face_text = f"**Face Match: GID {face_result['global_id']}** (Score: {face_result['score']:.3f})"

            if display_img is None:

                furl = face_result.get("match_face_url", "")

                if furl: display_img = fetch_image_from_url(furl, as_rgb=True)

        else:

            face_text = f"Face: {face_result['status']} — {face_result.get('reason', '')}"



    lines = ["## Search Results\n"]

    if body_text: lines.append(body_text)

    if face_text: lines.append(face_text)



    details = ""

    best_gid = None

    if body_result and body_result["status"] == "MATCH": best_gid = body_result["global_id"]

    if face_result and face_result["status"] == "MATCH": best_gid = face_result["global_id"]

    if best_gid:

        info, _, _ = get_person_details(best_gid)

        details = info



    return "\n\n".join(lines), display_img, details





# ═══════════════════════════════════════════════════════════════

# PIPELINE HANDLER + RESULTS RENDERING

# ═══════════════════════════════════════════════════════════════

_last_run = {"gallery": None, "sighting_tracker": None, "is_face": True, "camera_id": CAMERA_ID}


