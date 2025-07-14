# Versi Stream + Filter LineZone Menuju Reklame
# Penambahan Zona Reklame (Area Fokus untuk Perhitungan)
# stream cctv via virtual cam obs
# input bisa diganti dengan video
# ==============================================

import cv2                      # Untuk video & image processing
import torch                    # Untuk deep learning (YOLOv5 pakai PyTorch)
import numpy as np              # Untuk operasi array
import supervision as sv        # Untuk annotator + tracker (ByteTrack)
import pathlib                  # =========================
import os
import json
import time                     # Untuk file system & waktu

from datetime import datetime
from collections import deque   # Untuk struktur data buffer sliding window

import firebase_admin
from firebase_admin import credentials, firestore

# 🔐 Inisialisasi Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")    # Kunci akses Firebase
firebase_admin.initialize_app(cred)                         # Inisialisasi SDK Firebase
db = firestore.client()                                     # Buat koneksi ke Firestore

# === Path Handling (biar nggak error di Windows)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# === YOLOv5 imports
from ultralytics.utils.plotting import Annotator
from models.common import DetectMultiBackend
from utils.general import scale_boxes, non_max_suppression, check_img_size
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages, LoadStreams

# === Konfigurasi
weights_path = 'yolov5m.pt'             # Path model YOLOv5
source = '2'                            # Input: bisa webcam (index), video, atau URL stream
save_interval = 60                      # Metadata baru, disimpan tiap interval
save_video_interval = 3600              # Video baruhasil deteksi disimpan tiap interval

# === Buat folder output kalau belum ada
os.makedirs('output24/video/uji-finale', exist_ok=True)
os.makedirs('output24/metadata/uji-finale', exist_ok=True)

# === Setup Model YOLOv5
device = select_device('')  # Pilih device (GPU/CPU otomatis)
model = DetectMultiBackend(weights_path, device=device) # Load model YOLO
stride, names, pt = model.stride, model.names, model.pt # Ambil info konfigurasi model
imgsz = check_img_size(640, s=stride)   # Validasi input image size

# === Load Source Input
is_stream = False
if isinstance(source, int) or (
    isinstance(source, str) and (
        source.isnumeric() or
        source.startswith(('rtsp://', 'http://', 'https://')) or
        source.endswith(('.mjpg', '.mjpeg'))
    )
):
    is_stream = True

# === Pilih loader sesuai jenis input
if is_stream:
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
else:
    dataset = LoadImages(source, img_size=imgsz)

# === Setup Tracker & Annotator
tracker = sv.ByteTrack()                # Tracking objek menggunakan ByteTrack
box_annotator = sv.BoxAnnotator()       # Annotator untuk bounding box
label_annotator = sv.LabelAnnotator()   # Annotator untuk label ID
unique_objects = {}                     # Untuk menyimpan ID unik tiap class

# === Zona Area Deteksi Reklame
ZONE_X1, ZONE_Y1 = 90, 200  # Titik kiri atas area reklame
ZONE_X2, ZONE_Y2 = 350, 340  # Titik kanan bawah area reklame
linezone_class_counter = {name: 0 for name in names.values()} # Buat dict untuk menghitung per kelas
crossed_ids = set()                     # Simpan ID yang sudah pernah crossing

# ==== Buffer Penampung Sliding Window
window_size = 10  # Ganti ke 5, 10 sesuai kebutuhan
recent_counts = {name: deque(maxlen=window_size) for name in names.values()} # Sliding window untuk semua class
upload_buffer = deque(maxlen=5) # Buffer untuk mengisi metadata sebelum diupload
estimated_all = {}              # Estimasi rata-rata objek
error_ranges = {}               # Error/selisih estimasi

# === Inisialisasi Waktu dan Video Writer
last_save_time = time.time()
last_video_save_time = time.time()
detected_video_writer = None
captured_video_path = None

# ==== Fungsi Upload ke Firestore
def upload_metadata_to_firestore(timestamp, counts, estimated_all=None, error_ranges=None):
    # Buat 1 dokumen berisi: timestamp, counts, estimasi & error range
    # Upload ke Firestore
    try:
        doc_ref = db.collection("detections").document(timestamp)
        data = {
            "timestamp": timestamp,
            "counts": counts,
            "lokasi": "CCTV"
        }
        if estimated_all is not None:
            data["estimated"] = estimated_all

        if error_ranges is not None:
            data["visual_range"] = {
                cls: {
                    "estimate": estimated_all.get(cls, 0),
                    "error": error_ranges.get(cls, 0)
                } for cls in estimated_all
            }
        doc_ref.set(data)
        print(f"🚀 Metadata diupload ke Firestore: {timestamp}")
        return True
    except Exception as e:
        print(f"❌ Gagal upload metadata: {e}")
        return False

# === Fungsi Logika Perhitungan Objek Masuk Zona
def is_inside_zone(x1, y1, x2, y2):
    """Cek apakah titik tengah bbox berada dalam area reklame."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return ZONE_X1 <= cx <= ZONE_X2 and ZONE_Y1 <= cy <= ZONE_Y2

# === Main Loop
for path, im, im0s, vid_cap, _ in dataset:
    if isinstance(im0s, list):
        im0s = im0s[0]  # Ambil satu frame dari list ketika pakai LoadStreams

    # ==== Preprocessing YOLOv5
    im = torch.from_numpy(im).to(device) # Ubah format gambar (numpy) ke tensor pytorch
    im = im.float() / 255.0              # Ubah nilai piksel dengan konversi float dan dinormalisasi
    if im.ndimension() == 3:             # Menambah 1 dimensi pada tensor, agar sesuai dengan bentuk batch yang dibutuhkan YOLOv5
        im = im.unsqueeze(0)

    frame_start_time = time.time()       # Ambil waktu sebelum proses satu frame dimulai

    # === Inference YOLOv5
    pred = model(im, augment=False, visualize=False)    # Proses inferensi menggunakan model YOLOv5
    pred = non_max_suppression(pred, 0.30, 0.55)[0] # Terapkan NMS untuk buang kotak prediksi yang tumpang tindih

    # ==== Tracking dan Anotasi
    if pred is not None and len(pred):  # Logika untuk cek apakah ada hasil prediksi yang valid
        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], im0s.shape).round() # Skala ulang koordinat bounding box ke ukuran asli frame
        pred_np = pred.cpu().numpy()    # Pindahkan tensor prediksi ke NumPy array

        detections = sv.Detections( # Buat objek deteksi dari bounding box, confidence, dan class ID
            xyxy=pred_np[:, :4],
            confidence=pred_np[:, 4],
            class_id=pred_np[:, 5].astype(int)
        )

        tracked = tracker.update_with_detections(detections)    # Lacak objek berdasarkan hasil deteksi yang baru

        # === Hitung yang melewati LineZOne
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            class_id = tracked.class_id[i]
            track_id = tracked.tracker_id[i]

            # Cek apakah objek sudah dihitung dan berada dalam zona reklame
            if is_inside_zone(x1, y1, x2, y2) and track_id not in crossed_ids:
                #print(f"Objek {names[class_id]} ID {track_id} masuk zona reklame.")
                class_name = names[class_id]
                linezone_class_counter[class_name] += 1
                crossed_ids.add(track_id)

                # Simpan ID unik
                if class_id not in unique_objects:
                    unique_objects[class_id] = set()
                unique_objects[class_id].add(track_id)

        labels = []
        for c, conf, tid in zip(tracked.class_id, tracked.confidence, tracked.tracker_id):
            label_prefix = "IN " if tid in crossed_ids else ""
            label = f"{label_prefix}#ID:{tid} {names[c]} [{conf:.2f}]"
            labels.append(label)
        im0s = box_annotator.annotate(scene=im0s, detections=tracked)                   # Gambar bounding box di frame
        im0s = label_annotator.annotate(scene=im0s, detections=tracked, labels=labels)  # Tampilkan label teks di atas objek

        frame_end_time = time.time()    # Ambil waktu setelah proses satu frame selesai
        fps = 1 / (frame_end_time - frame_start_time)   # Hitung kecepatan FPS (Frame per Second)

        # Tampilkan FPS di kiri atas
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(im0s, fps_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

        # Tampilan Zona
        cv2.rectangle(im0s, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 255, 255), 2)
        cv2.putText(im0s, "ARAH REKLAME", (ZONE_X1 + 5, ZONE_Y1 - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    # === Simpan video hasil deteksi setiap 6 jam
    current_time = time.time()  # Ambil waktu saat ini

    # Cek apakah waktunya untuk menyimpan video baru, atau writer belum diinisialisasi
    if detected_video_writer is None or (current_time - last_video_save_time >= save_video_interval):
        if detected_video_writer:
            detected_video_writer.release() # Tutup video writer sebelumnya
            elapsed_video = int(current_time - last_video_save_time)    # Hitung selisih waktu sejak video sebelumnya
            print("\n" + "━" * 0)
            print(f"Video disimpan di: {captured_video_path}(⏱️ {elapsed_video} detik berlalu)")
        last_video_save_time = current_time  # Update setelah selesai simpan

        # Buka writer baru
        video_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        height, width = im0s.shape[:2]  # Ambil dimensi frame dari frame saat ini
        captured_video_path = f'output24/video/uji-finale/detected_{video_timestamp}.avi'   # Path penyimpanan
        detected_video_writer = cv2.VideoWriter(    # Inisialisasi writer video baru
            captured_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
        print(f"▶️ Mulai simpan video hasil deteksi baru: {captured_video_path}")

    if detected_video_writer:
        detected_video_writer.write(im0s) # Tulis frame saat ini ke file video

    # === Simpan metadata tiap interval
    if current_time - last_save_time >= save_interval:  # Cek apakah waktu interval penyimpanan metadata sudah tercapai
        last_save_time = current_time   # Update waktu terakhir metadata disimpan
        meta_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_path = f"output24/metadata/uji-finale/result_{meta_timestamp}.json"    # Path file metadata yang akan disimpan
        final_counts = {names[class_id]: len(ids) for class_id, ids in unique_objects.items()}   # Hitung jumlah objek unik yang terdeteksi selama interval ini

        # Hitung sliding window estimasi
        for cls in final_counts:
            count_now = final_counts[cls]           # Ambil jumlah objek saat ini
            recent_counts[cls].append(count_now)    # Tambahkan ke sliding window

            if recent_counts[cls]:  # Jika sliding window tidak kosong
                avg = int(sum(recent_counts[cls]) / len(recent_counts[cls]))    # Rata-rata jumlah objek
                err = (max(recent_counts[cls]) - min(recent_counts[cls])) // 2  # Error range (setengah dari rentang)
            else:
                avg, err = count_now, 0 # Jika kosong, gunakan nilai saat ini sebagai estimasi

            estimated_all[cls] = avg    # Simpan estimasi rata-rata
            error_ranges[cls] = err     # Simpan rentang kesalahan (error range)

        metadata_output = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "counts": final_counts  # Jumlah akhir objek yang dihitung
        }
        # Simpan metadata ke file JSON lokal
        with open(meta_path, 'w') as f:
            json.dump(metadata_output, f, indent=2)

        upload_buffer.append({  # Tambahkan metadata hasil deteksi ke dalam buffer untuk dianalisis
            "timestamp": metadata_output["timestamp"],  # Timestamp metadata saat ini
            "counts": final_counts,                     # Jumlah objek yang terdeteksi
            "estimated_all": estimated_all,             # Estimasi objek (sliding window)
            "error_ranges": error_ranges                # Error range dari estimasi tiap objek
        })
        # Tampilan status buffer di terminal
        print(f"Buffer sekarang berisi: {len(upload_buffer)}/{upload_buffer.maxlen}")
        for item in upload_buffer:
            print(" └─", item["timestamp"])

        # Upload metadata hanya jika buffer penuh
        if len(upload_buffer) == upload_buffer.maxlen:
            print(f"Buffer penuh! Mengupload rata-rata dari {len(upload_buffer)} metadata...")
            middle_index = len(upload_buffer) // 2  # Ambil index tengah sebagai timestamp upload
            avg_timestamp = upload_buffer[middle_index]["timestamp"]    # Gunakan timestamp tengah untuk dokumentasi Firestore

            # Gabungkan counts semua item
            total_counts = {}   # Dictionary untuk menjumlahkan semua counts objek dari buffer
            for item in upload_buffer:
                for cls, count in item["counts"].items():
                    total_counts[cls] = total_counts.get(cls, 0) + count    # Akumulasi count setiap kelas objek

            # Hitung rata-rata count untuk setiap kelas objek
            avg_counts = {cls: total // len(upload_buffer) for cls, total in total_counts.items()}  # Hitung rata-rata objek

            # Hitung rata-rata estimated & error juga
            total_estimates = {}    # Akumulasi total estimasi dari buffer
            total_errors = {}       # Akumulasi total error range dari buffer
            for item in upload_buffer:
                for cls in item["estimated_all"]:
                    total_estimates[cls] = total_estimates.get(cls, 0) + item["estimated_all"][cls] # Total estimasi per kelas
                    total_errors[cls] = total_errors.get(cls, 0) + item["error_ranges"][cls]        # Total error per kelas

            # Hitung rata-rata estimasi dan error per kelas objek
            avg_estimates = {cls: est // len(upload_buffer) for cls, est in total_estimates.items()}    # Estimasi rata-rata
            avg_errors = {cls: err // len(upload_buffer) for cls, err in total_errors.items()}          # Error range rata-rata

            # Upload hanya satu dokumen hasil agregasi
            upload_metadata_to_firestore(
                avg_timestamp,
                avg_counts,
                avg_estimates,
                avg_errors
            )
            upload_buffer.clear()       # Bersihkan buffer setelah upload agar siap isi data berikutnya
            unique_objects = {}         # Reset unique_objects untuk interval berikutnya

        print("━" * 50)
        print(f"Metadata disimpan di: {meta_path}(⏱️ {save_interval} detik berlalu)")
        print("━" * 50 + "\n")

    # === Tampilkan hasil (opsional bisa dimatikan saat production)
    cv2.imshow("YOLOv5 + ByteTrack + Supervision", im0s)    # Tampilkan frame hasil deteksi secara real-time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==== Ending
print("\n" + "━" * 50)
print("🛑 Program Selesai!")
print("📦 Semua proses telah diselesaikan:")

if detected_video_writer:
    print(f"📹 Video terakhir disimpan di: {captured_video_path}")
print(f"📝 Metadata terakhir disimpan di: {meta_path}")

if upload_buffer:
    print("⚠️ Belum ada metadata yang diupload ke Firestore pada sesi ini.")
else:
    print("☁️ Metadata berhasil diupload ke Firestore dari buffer terakhir.")

print("✅ Sistem siap digunakan kembali.")
print("━" * 50 + "\n")

# === Cleanup
cv2.destroyAllWindows()
if detected_video_writer:
    detected_video_writer.release()