from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import cv2
import numpy as np
import base64
import requests
import os

app = FastAPI(title="ConTech Delta Engine")

# Grab your Gemini API key from Render environment variables or paste it here for testing
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

class DiffRequest(BaseModel):
    pdf_old_b64: str
    pdf_new_b64: str
    target_dpi: int = 150

@app.get("/health")
def health():
    return {"status": "online", "engine": "OpenCV-RANSAC-Gemini"}

@app.post("/webhook/drawing-delta")
def process_drawing_delta(req: DiffRequest):
    try:
        # 1. Decode PDFs
        old_bytes = base64.b64decode(req.pdf_old_b64)
        new_bytes = base64.b64decode(req.pdf_new_b64)
        
        doc_old = fitz.open(stream=old_bytes, filetype="pdf")
        doc_new = fitz.open(stream=new_bytes, filetype="pdf")
        
        page_old = doc_old[0]
        page_new = doc_new[0]
        
        zoom = req.target_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        pix_old = page_old.get_pixmap(matrix=mat, alpha=False)
        pix_new = page_new.get_pixmap(matrix=mat, alpha=False)
        
        img_old = np.frombuffer(pix_old.samples, dtype=np.uint8).reshape(pix_old.h, pix_old.w, 3)
        img_new = np.frombuffer(pix_new.samples, dtype=np.uint8).reshape(pix_new.h, pix_new.w, 3)
        
        gray_old = cv2.cvtColor(img_old, cv2.COLOR_RGB2GRAY)
        gray_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
        doc_old.close()
        
        # 2. Scale & Boundary Normalization
        if gray_old.shape != gray_new.shape:
            max_h = max(gray_old.shape[0], gray_new.shape[0])
            max_w = max(gray_old.shape[1], gray_new.shape[1])
            gray_old = cv2.copyMakeBorder(gray_old, 0, max_h - gray_old.shape[0], 0, max_w - gray_old.shape[1], cv2.BORDER_CONSTANT, value=255)
            gray_new = cv2.copyMakeBorder(gray_new, 0, max_h - gray_new.shape[0], 0, max_w - gray_new.shape[1], cv2.BORDER_CONSTANT, value=255)
            img_new = cv2.copyMakeBorder(img_new, 0, max_h - img_new.shape[0], 0, max_w - img_new.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])

        h, w = gray_old.shape
        alignment_success = True

        # 3. ORB + RANSAC Homography Alignment
        orb = cv2.ORB_create(nfeatures=2500)
        kp_old, des_old = orb.detectAndCompute(gray_old, None)
        kp_new, des_new = orb.detectAndCompute(gray_new, None)
        
        matrix = None
        if des_old is not None and des_new is not None and len(kp_old) >= 10 and len(kp_new) >= 10:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des_old, des_new), key=lambda x: x.distance)
            good_matches = matches[:max(int(len(matches) * 0.25), 10)]
            if len(good_matches) >= 8:
                pts_old = np.float32([kp_old[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                pts_new = np.float32([kp_new[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                matrix, _ = cv2.estimateAffinePartial2D(pts_new, pts_old, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if matrix is None:
            alignment_success = False
            aligned_new = gray_new.copy()
        else:
            aligned_new = cv2.warpAffine(gray_new, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # 4. Pixel Diff & Noise Filter
        diff = cv2.absdiff(gray_old, aligned_new)
        _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        deletions_mask = (clean_thresh > 0) & (aligned_new > 180) & (gray_old < 180)
        additions_mask = (clean_thresh > 0) & (gray_old > 180) & (aligned_new < 180)

        delta_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        delta_overlay[deletions_mask] = [30, 30, 230, 255]   # Red
        delta_overlay[additions_mask] = [240, 100, 30, 255]  # Blue

        # 5. Extract Crops for Gemini
        alpha_channel = delta_overlay[:, :, 3]
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 120:
                x, y, bw, bh = cv2.boundingRect(cnt)
                pad = 80
                ymin, xmin = max(0, y - pad), max(0, x - pad)
                ymax, xmax = min(h, y + bh + pad), min(w, x + bw + pad)
                boxes.append((area, (ymin, xmin, ymax, xmax)))
        boxes.sort(key=lambda b: b[0], reverse=True)
        
        crops_b64 = []
        for _, (ymin, xmin, ymax, xmax) in boxes[:3]:
            crop_rgb = img_new[ymin:ymax, xmin:xmax]
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            _, enc_crop = cv2.imencode('.png', crop_bgr)
            crops_b64.append(base64.b64encode(enc_crop.tobytes()).decode('utf-8'))

        # 6. Inject Toggleable OCG Layer into PDF
        _, encoded_overlay = cv2.imencode('.png', delta_overlay)
        ocg_id = doc_new.add_ocg("Revision Delta Highlights (Red/Blue)", on=True)
        page_new.insert_image(page_new.rect, stream=encoded_overlay.tobytes(), ocg=ocg_id)
        
        out_pdf_bytes = doc_new.tobytes(garbage=3, deflate=True)
        doc_new.close()

        # 7. Call Gemini 1.5 Flash API directly from Python
        gemini_analysis = "No significant visual deltas detected to analyze."
        if crops_b64 and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "You are a senior construction document controller. Review these visual crop deltas between drawing revisions and format output into 3 strict bullet points: 1. Core Physical Change, 2. Trade Risk, 3. Recommended Field Action."},
                        {"inline_data": {"mime_type": "image/png", "data": crops_b64[0]}}
                    ]
                }]
            }
            res = requests.post(gemini_url, json=payload)
            if res.status_code == 200:
                try:
                    gemini_analysis = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    gemini_analysis = "Gemini response parsed with errors."

        return {
            "status": "success",
            "alignment_success": alignment_success,
            "change_cluster_count": len(boxes),
            "gemini_risk_report": gemini_analysis,
            "ocg_pdf_b64": base64.b64encode(out_pdf_bytes).decode('utf-8')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        # Add this at the very bottom of worker.py
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("worker:app", host="0.0.0.0", port=port)
