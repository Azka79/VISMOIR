import flet
from flet import *

from threading import Thread
import base64

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt


from flet.matplotlib_chart import MatplotlibChart

matplotlib.use("svg")

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.sparse.linalg import cg
from skimage.restoration import denoise_tv_chambolle
from matplotlib import cm

class font:
    BOLD        = "Bold"
    EXTRABOLD   = "ExtraBold"
    ITALIC      = "Italic"
    MEDIUM      = "Medium"
    REGULER     = "Reguler"
    SEMIBOLD    = "SemiBold"
    THIN        = "Thin"
    BEBAS_NEUE  = "Bebas"
    AMSTERDAM   = "Amsterdam"
    STAAT       = "Staat"
class color:
    WHITE = "#F8F9FC"
    BLACK = "#1C2938"
    BLUE = "#201E43"
    PURPLE = "#4F1787"
    ORANGE = "#FB773C"
    BLUE_1 = "#134B70"
    RED = "#FB2576"

class VitalSignProcess():
    # === Kode Pelacakan Wajah (SCF) === #
    def initialize_tracking(first_center, padding=1, rho=0.075, lambda_=0.25, num=5, alpha=2.25):        
        target_sz = np.array([first_center[3], first_center[2]])  # ukuran target
        pos = np.array([(first_center[1] + first_center[3] / 2), (first_center[0] + first_center[2] / 2)])  # posisi target (pusat)
        sz = np.floor(target_sz * (1 + padding)).astype(int)  # ukuran wilayah konteks

        rs, cs = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), indexing='ij')
        rs = rs - np.floor(sz[0] / 2)
        cs = cs - np.floor(sz[1] / 2)
        dist = rs**2 + cs**2
        conf = np.exp(-0.5 / alpha * np.sqrt(dist))  # peta kepercayaan
        conf /= np.sum(conf)  # normalisasi
        conff = np.fft.fft2(conf)  # ubah ke domain frekuensi

        hamming_window = np.outer(np.hamming(sz[0]), np.hamming(sz[1]))
        sigma = np.mean(target_sz)
        window = hamming_window * np.exp(-0.5 / (sigma**2) * dist)  # kurangi efek batas gambar
        window /= np.sum(window)  # normalisasi

        return pos, target_sz, sz, conff, window, sigma, rho, lambda_, num, dist

    def get_context(im, pos, sz, window):
        xs = np.floor(pos[1] + np.arange(1, sz[1] + 1) - (sz[1] / 2)).astype(int)
        ys = np.floor(pos[0] + np.arange(1, sz[0] + 1) - (sz[0] / 2)).astype(int)

        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        out = im[np.ix_(ys, xs)]
        out = out.astype(np.float64)
        out = (out - np.mean(out))  # Normalisasi
        out = window * out  # Berikan bobot intensitas sebagai model prior konteks seperti pada Eq.(4)
        
        return out

    def update_scale(maxconf, frame, num, scale, lambda_):
        if (frame % (num + 2)) == 0:
            scale_curr = 0
            for kk in range(num):
                scale_curr += np.sqrt(maxconf[frame - kk - 1] / maxconf[frame - kk - 2])
            scale = (1 - lambda_) * scale + lambda_ * (scale_curr / num)
        return scale

    def process_frame_tracking(im, pos, sz, window, Hstcf, conff, rho):
        contextprior = VitalSignProcess.get_context(im, pos, sz, window)
        hscf = conff / (np.fft.fft2(contextprior) + np.finfo(float).eps)
        
        if Hstcf is None:
            Hstcf = hscf
        else:
            Hstcf = (1 - rho) * Hstcf + rho * hscf

        confmap = np.real(np.fft.ifft2(Hstcf * np.fft.fft2(contextprior)))
        row, col = np.unravel_index(np.argmax(confmap, axis=None), confmap.shape)
        pos = pos - sz / 2 + [row, col]
        
        return pos, Hstcf

    def visualize_tracking_opencv(im, faces, pos, target_sz, frame):
        rect_position = [int(pos[1] - target_sz[1] / 2), int(pos[0] - target_sz[0] / 2),
                        int(target_sz[1]), int(target_sz[0])]
        cv2.rectangle(im, (rect_position[0], rect_position[1]),
                    (rect_position[0] + rect_position[2], rect_position[1] + rect_position[3]), (0, 0, 255), 4)
        cv2.rectangle(im, (faces[0], faces[1]),
                    (faces[2], faces[3]), (255, 0, 0), 4)
        cv2.putText(im, f'#{frame + 1}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # === Kode Ekstraksi Sinyal Vital === #
    def interpolate_signal(signal, frame_rate):
        t = np.linspace(0, len(signal) / frame_rate, len(signal))
        signal=np.array(signal)
        non_zero_mask = signal != 0
        new_amp = signal[non_zero_mask]
        new_t = t[non_zero_mask]
        x = t[~non_zero_mask]
        y = np.interp(x, new_t, new_amp)
        signal[~non_zero_mask] = y
        signal -= signal.mean()
        return signal

    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        b, a = VitalSignProcess.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def graph_laplacian_regularization(signal, alpha=100, ):
        n = len(signal)

        # Construct the graph
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if i !=j :
                    W[i, j] = W[j, i] = np.exp(-np.abs(signal[i] - signal[j]) ** 2)

        # Degree matrix
        D = np.diag(W.sum(axis=1))

        # Laplacian matrix
        L = D - W

        # Identity matrix
        I = np.eye(n)

        # Regularization equation: (I + alpha * L) * f = f_0
        A = I + alpha * L

        # Solve the regularization problem using Conjugate Gradient (CG)
        f_reg, _ = cg(A, signal)

        return f_reg


    def graph_total_variation(signal, weight=0.2, eps=0.09):
        denoised_signal = denoise_tv_chambolle(signal, weight=weight, eps=eps)
        return denoised_signal

    def convert_to_temperature(mean_pixel_value, suhu_max, suhu_min):
        norm = mean_pixel_value / 255
        temp = np.linspace(suhu_min, suhu_max, 10000)
        cmap = cm.get_cmap('gray', 10000)
        x = cmap(range(10000))
        x = x[:, 0]
        difference_array = np.absolute(x - norm)
        index = difference_array.argmin()
        return temp[index]

    def normalize_signal(signal, temperature_min, temperature_max):
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        temperature_scale_signal = normalized_signal * (temperature_max - temperature_min) + temperature_min
        return temperature_scale_signal

    def process_frame_vital_sign(frame, faces, pos, target_sz, suhu_max, suhu_min):
        nose_x1 = int(pos[1] - target_sz[1] / 2)
        nose_y1 = int(pos[0] - target_sz[0] / 2) - 50
        nose_x2 = int(pos[1] + target_sz[1] / 2)
        nose_y2 = int(pos[0] + target_sz[0] / 2)

        face_x1 = faces[0]
        face_y1 = faces[1]
        face_x2 = faces[2]
        face_y2 = faces[3]

        
        cropped_nose = frame[nose_y1:nose_y2, nose_x1:nose_x2]
        nose_frame_gray = cv2.cvtColor(cropped_nose, cv2.COLOR_BGR2GRAY)
        nose_mean = VitalSignProcess.convert_to_temperature(np.mean(nose_frame_gray), suhu_max, suhu_min)

        cropped_face = frame[face_y1:face_y2, face_x1:face_x2]
        face_frame_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        face_mean = VitalSignProcess.convert_to_temperature(np.mean(face_frame_gray), suhu_max, suhu_min)
        
        return face_mean, nose_mean

    def multilevel_otsu_thresholding(image, num_levels):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresholds = []
        for _ in range(num_levels - 1):
            _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold_value = np.max(thresholded_image)
            thresholds.append(threshold_value)
            blurred_image = cv2.bitwise_and(blurred_image, thresholded_image)

        return thresholds, thresholded_image

    def largest_connected_component(image):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_mask = (labels == largest_label).astype(np.uint8) * 255
        return largest_component_mask

    def find_chin_contour(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    def segment_face(frame):
        _, thresholded_image = VitalSignProcess.multilevel_otsu_thresholding(frame, num_levels=9)
        largest_component_mask = VitalSignProcess.largest_connected_component(thresholded_image)
        face_contour = VitalSignProcess.find_chin_contour(largest_component_mask)
        x1, y1, w, h = cv2.boundingRect(face_contour)
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def detect_peaks_valleys(signal):
        peaks = []
        valleys = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                peaks.append(i)
            elif signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
                valleys.append(i)
        return peaks, valleys

class SimpleVideoPlayer(UserControl):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.filepath = ''
        self.frame_rate = None
        self.video_box = Image('Group 1.png',width=400)
        self.respiration_signal = []
        self.heart_rate_signal = []
        self.suhu_max = 37.0
        self.suhu_min = 23.6
        self.gtv_eps = 0.008
        self.gtv_weight = 0.2
        self.glr_alpha = 100

        self.status_title = Text('', size= 15, color=color.WHITE, font_family=font.REGULER, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        
        self.resp_count_title = Text('Respiration Rate (BPM)', size= 16, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        self.resp_count = Text('0', size= 50, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        self.hr_count_title = Text('Heart Rate (BPM)', size= 16, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        self.hr_count = Text('0', size= 50, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        self.temperature_title = Text('Body Temperature (C)', size= 16, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        self.temperature = Text('0', size= 50, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)

        self.title_setup = Text('Temperature Setup', size= 16, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        self.title_GLR = Text('Graph Laplacian Regularization (GLR)', size= 16, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)
        self.title_GTV = Text('Graph Total Variation (GTV)', size= 16, color=color.WHITE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)

        self.advance_title = Text('ADVANCE REPORT', size= 70, color=color.BLUE, font_family=font.AMSTERDAM, selectable=True, no_wrap=True, text_align=TextAlign.CENTER)

        self.max_temp_box = Container(
            content= TextField(
                value="37.0",
                border= border.all(1, color.BLUE),
                border_radius=10,
                content_padding=padding.only(top=0, bottom=0, right=20, left=20),
                label_style=TextStyle(size=15, color=color.WHITE),
                label="Max Temperature",
                cursor_color=color.BLUE,
                text_style=TextStyle(size=18, color=color.WHITE),
                border_width=2,
                border_color=color.PURPLE,
                width=200,
                on_change=self.max_temp_change
            )
        )
        self.min_temp_box = Container(
            content= TextField(
                value="23.0",
                border= border.all(1, color.BLUE),
                border_radius=10,
                content_padding=padding.only(top=0, bottom=0, right=20, left=20),
                label_style=TextStyle(size=15, color=color.WHITE),
                label="Min Temperature",
                cursor_color=color.BLUE,
                text_style=TextStyle(size=18, color=color.WHITE),
                border_width=2,
                border_color=color.PURPLE,
                width=200,
                on_change=self.min_temp_change
            )
        )
        self.glr_alpha_box = Container(
            content= TextField(
                value="100",
                border= border.all(1, color.BLUE),
                border_radius=10,
                content_padding=padding.only(top=0, bottom=0, right=20, left=20),
                label_style=TextStyle(size=15, color=color.WHITE),
                label="Alpha",
                cursor_color=color.BLUE,
                text_style=TextStyle(size=18, color=color.WHITE),
                border_width=2,
                border_color=color.PURPLE,
                width=200,
                on_change=self.glr_alpha_change
            )
        )
        self.gtv_weight_box = Container(
            content= TextField(
                value="0.2",
                border= border.all(1, color.BLUE),
                border_radius=10,
                content_padding=padding.only(top=0, bottom=0, right=20, left=20),
                label_style=TextStyle(size=15, color=color.WHITE),
                label="Weight",
                cursor_color=color.BLUE,
                text_style=TextStyle(size=18, color=color.WHITE),
                border_width=2,
                border_color=color.PURPLE,
                width=200,
                on_change=self.gtv_weight_change
            )
        )
        self.gtv_eps_box = Container(
            content= TextField(
                value="0.008",
                border= border.all(1, color.BLUE),
                border_radius=10,
                content_padding=padding.only(top=0, bottom=0, right=20, left=20),
                label_style=TextStyle(size=15, color=color.WHITE),
                label="Eps",
                cursor_color=color.BLUE,
                text_style=TextStyle(size=18, color=color.WHITE),
                border_width=2,
                border_color=color.PURPLE,
                width=200,
                on_change=self.gtv_eps_change
            )
        )

        self.resp_count_title_box = Container(self.resp_count_title,  width=200, height=50, padding=5, border_radius=5, bgcolor=color.ORANGE, alignment=alignment.center)
        self.resp_count_box = Container(self.resp_count,  width=200, height=110, padding=5, border_radius=5, bgcolor=color.PURPLE, alignment=alignment.center)
        self.hr_count_title_box = Container(self.hr_count_title,  width=200, height=50, padding=5, border_radius=5, bgcolor=color.ORANGE, alignment=alignment.center)
        self.hr_count_box = Container(self.hr_count,  width=200, height=110, padding=5, border_radius=5, bgcolor=color.PURPLE, alignment=alignment.center)
        self.temperature_title_box = Container(self.temperature_title,  width=200, height=50, padding=5, border_radius=5, bgcolor=color.ORANGE, alignment=alignment.center)
        self.temperature_box = Container(self.temperature,  width=200, height=110, padding=5, border_radius=5, bgcolor=color.PURPLE, alignment=alignment.center)

        
        self.fig, self.axs = plt.subplots(1, figsize=(20,10), facecolor=color.WHITE)
        self.axs.set_xlim(0,1)
        self.axs.set_ylim(0,1)
        self.axs.grid()
        self.axs.tick_params(axis="both", colors=color.BLUE)
        self.fig.tight_layout()
        self.chart = MatplotlibChart(self.fig, expand=True, isolated=True)

        self.fig1, self.axs1 = plt.subplots(1, figsize=(20,10), facecolor=color.WHITE)
        self.axs1.set_xlim(0,1)
        self.axs1.set_ylim(0,1)
        self.axs1.grid()
        self.axs1.tick_params(axis="both", colors=color.BLUE)
        self.fig1.tight_layout()
        self.chart1 = MatplotlibChart(self.fig1, expand=True, isolated=True)

        self.fig2, self.axs2 = plt.subplots(1, figsize=(20,10), facecolor=color.WHITE)
        self.axs2.set_xlim(0,1)
        self.axs2.set_ylim(0,1)
        self.axs2.grid()
        self.axs2.tick_params(axis="both", colors=color.BLUE)
        self.fig2.tight_layout()
        self.chart2 = MatplotlibChart(self.fig2, expand=True, isolated=True)

        self.fig3, self.axs3 = plt.subplots(1, figsize=(20,10), facecolor=color.WHITE)
        self.axs3.set_xlim(0,1)
        self.axs3.set_ylim(0,1)
        self.axs3.grid()
        self.axs3.tick_params(axis="both", colors=color.BLUE)
        self.fig3.tight_layout()
        self.chart3 = MatplotlibChart(self.fig3, expand=True, isolated=True)

    def max_temp_change(self,e):
        self.suhu_max = float(e.control.value)
        if self.suhu_max != 0:
            self.result()
    def min_temp_change(self,e):
        self.suhu_min = float(e.control.value)
        if self.suhu_min != 0:
            self.result()
    def glr_alpha_change(self,e):
        self.glr_alpha = float(e.control.value)
        if self.glr_alpha != 0:
            self.result()
    def gtv_weight_change(self,e):
        self.gtv_weight = float(e.control.value)
        if self.gtv_weight != 0:
            self.result()
    def gtv_eps_change(self,e):
        self.gtv_eps = float(e.control.value)
        if self.gtv_eps != 0:
            self.result()
        
    def play_video(self, image_widget, cap):
        ret, frame = cap.read()
        nose_cascade = cv2.CascadeClassifier('assets/cascade.xml')
        net = cv2.dnn.readNetFromCaffe('assets/deploy.prototxt', 'assets/res10_300x300_ssd_iter_140000.caffemodel')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noses = nose_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(noses) == 0:
            first_center = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")

        elif noses[0][1]>400 or noses[0][1]<261:

                max_nose_area = 0
                largest_nose = None

                for (x, y, w, h) in noses:
                    area = w * h
                    if area > max_nose_area:
                        max_nose_area = area
                        largest_nose = (x, y, w, h)

                if largest_nose is not None:
                    x, y, w, h = largest_nose
                    cropped_nose = frame[int(y):int(y+h), int(x):int(x+w)]
                    nose_frame_gray = cv2.cvtColor(cropped_nose, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(nose_frame_gray, 30, 100)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) > 0:
                        leftmost_x = min(min(contour[:, 0, 0]) for contour in contours)
                        rightmost_x = max(max(contour[:, 0, 0]) for contour in contours)
                        topmost_y = min(min(contour[:, 0, 1]) for contour in contours)
                        bottommost_y = max(max(contour[:, 0, 1]) for contour in contours)
                
                first_center = [x + leftmost_x, y + topmost_y,rightmost_x, bottommost_y]
        else:
        
            # Inisialisasi pelacakan dengan nilai tengah dari ROI deteksi
            first_center = noses[0]

        pos, target_sz, sz, conff, window, sigma, rho, lambda_, num, dist = VitalSignProcess.initialize_tracking(first_center)

        frame_count = 0
        maxconf = []
        Hstcf = None
        scale = 1.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame.ndim > 2:
                im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                im = frame

            sigma *= scale
            window = np.outer(np.hamming(sz[0]), np.hamming(sz[1])) * np.exp(-0.5 / (sigma**2) * dist)
            window /= np.sum(window)  # normalisasi
            
            pos, Hstcf = VitalSignProcess.process_frame_tracking(im, pos, sz, window, Hstcf, conff, rho)

            startX = None
            endX =None 
            startY =None 
            endY =None

            resized_frame = cv2.resize(frame, (300, 300))
            blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

            # Perform face detection DNN
            net.setInput(blob)
            detections = net.forward()

            # Loop over the detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                if startX == None or  endX == None or startY == None or  endY == None:
                    startX, startY, endX, endY = VitalSignProcess.segment_face(frame)
                        
            if (endX-startX)*(endY-startY) > 320000:
                faces = [startX + 50, startY, endX-100, endY-250]
            else:
                faces = [startX, startY, endX, endY]

            if frame_count > 0:
                conftmp = np.real(np.fft.ifft2(Hstcf * np.fft.fft2(VitalSignProcess.get_context(im, pos, sz, window))))
                maxconf.append(np.max(conftmp))
                scale = VitalSignProcess.update_scale(maxconf, frame_count, num, scale, lambda_)
            
            self.max_temp = np.max(frame[startY:endY, startX:endX])
            self.temp_in_celsius = VitalSignProcess.convert_to_temperature(self.max_temp, self.suhu_max, self.suhu_min)
            cv2.putText(frame, f"Suhu Tubuh: {self.temp_in_celsius:.1f}C", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            hr_signal , resp_signal= VitalSignProcess.process_frame_vital_sign(frame, faces, pos, target_sz, self.suhu_max, self.suhu_min)
            self.respiration_signal.append(resp_signal)
            self.heart_rate_signal.append(hr_signal)

            VitalSignProcess.visualize_tracking_opencv(frame, faces, pos, target_sz, frame_count)
            frame_count += 1

            # Encode and update frame on Flet Image widget
            _, img_encoded = cv2.imencode('.png', frame)
            img_bytes = img_encoded.tobytes()
            image_widget.src_base64 = base64.b64encode(img_bytes).decode('utf-8')
            image_widget.update()

        cap.release()
        cv2.destroyAllWindows()

        # Replace zero values with the interpolated values
        self.respiration_signal = VitalSignProcess.interpolate_signal(self.respiration_signal, self.frame_rate)
        self.heart_rate_signal = VitalSignProcess.interpolate_signal(self.heart_rate_signal, self.frame_rate)

        self.result()
    
    def result(self):
        
        self.status_title.value = 'Vital Sign Detection...'
        self.status_title.update()

        # Ensure the signals have the same length
        min_length = min(len(self.respiration_signal), len(self.heart_rate_signal))
        self.respiration_signal = self.respiration_signal[:min_length]
        self.heart_rate_signal = self.heart_rate_signal[:min_length]

        if self.frame_rate is not None:
            # Apply bandpass filter
            fs = self.frame_rate
            lowcut_rr, highcut_rr = 0.15, 0.45
            lowcut_hr, highcut_hr = 0.667, 2.5

            # Apply Graph Laplacian Regularization (GLR)
            alpha = self.glr_alpha 
            denoised_resp_signal = VitalSignProcess.graph_laplacian_regularization(self.respiration_signal, alpha)
            denoised_hr_signal = VitalSignProcess.graph_laplacian_regularization(self.heart_rate_signal, alpha)

            # Apply Graph Total Variation (GTV)
            denoised_respiration_signal = VitalSignProcess.graph_total_variation(denoised_resp_signal,weight=self.gtv_weight, eps=1e-4)
            denoised_heart_rate_signal = VitalSignProcess.graph_total_variation(denoised_hr_signal,weight=self.gtv_weight, eps=self.gtv_eps)

            filtered_resp_signal = VitalSignProcess.butter_bandpass_filter(denoised_respiration_signal, lowcut_rr, highcut_rr, fs)
            filtered_hr_signal = VitalSignProcess.butter_bandpass_filter(denoised_heart_rate_signal, lowcut_hr, highcut_hr, fs)

            normalized_respiration_signal = VitalSignProcess.normalize_signal(filtered_resp_signal, self.suhu_min, self.suhu_max)
            normalized_heart_rate_signal = VitalSignProcess.normalize_signal(filtered_hr_signal, self.suhu_min, self.suhu_max)

            smoothed_rr = savgol_filter(normalized_respiration_signal, window_length=25, polyorder=3)

            # Peak detection
            self.rr_peaks, _ = VitalSignProcess.detect_peaks_valleys(smoothed_rr)
            self.hr_peaks, _ = VitalSignProcess.detect_peaks_valleys(normalized_heart_rate_signal)

            # Generate time vector
            t = np.linspace(0, len(normalized_respiration_signal) / self.frame_rate, len(normalized_respiration_signal))

            self.allData = np.array([np.arange(0, len(normalized_respiration_signal)), t, VitalSignProcess.normalize_signal(self.respiration_signal, self.suhu_min, self.suhu_max),  VitalSignProcess.normalize_signal(self.heart_rate_signal, self.suhu_min, self.suhu_max)])

            # Vital sign counter
            threshold_duration_min = 2.5
            threshold_duration_max = 5
            counter_A = 0
            counter_B = 0

            for i in range(len(self.rr_peaks) - 1):
                breath_duration = t[self.rr_peaks[i + 1]] - t[self.rr_peaks[i]]
                if breath_duration < threshold_duration_min or breath_duration > threshold_duration_max:
                    counter_A += 1
                else:
                    counter_B += 1
            if counter_B == (((len(self.rr_peaks)-1) - counter_A)):
                self.rr_total = counter_A + counter_B

            self.respiration_rate = self.rr_total * (self.frame_rate / len(normalized_respiration_signal)) * 60
            self.heart_rate = len(self.hr_peaks) * (self.frame_rate / len(normalized_heart_rate_signal)) * 60
            self.temp_in_celsius = VitalSignProcess.convert_to_temperature(self.max_temp, self.suhu_max, self.suhu_min)

            # DISPLAY UPDATE
            self.resp_count.value = f"{self.respiration_rate:.1f}"
            self.resp_count.update()
            self.hr_count.value = f"{self.heart_rate:.1f}"
            self.hr_count.update()
            self.temperature.value = f"{self.temp_in_celsius:.1f}"
            self.temperature.update()

            if self.respiration_rate<=11 or self.respiration_rate>=21:
                self.resp_count_box.bgcolor = color.RED
                self.resp_count_box.update()
            else:
                self.resp_count_box.bgcolor = color.PURPLE
                self.resp_count_box.update()
            if self.heart_rate<=59 or self.heart_rate>=101:
                self.hr_count_box.bgcolor = color.RED
                self.hr_count_box.update()
            else:
                self.hr_count_box.bgcolor = color.PURPLE
                self.hr_count_box.update()
            if self.temp_in_celsius>38.0:
                self.temperature_box.bgcolor = color.RED
                self.temperature_box.update()
            else:
                self.temperature_box.bgcolor = color.PURPLE
                self.temperature_box.update()

            # CHART UPDATE
            self.axs.clear()
            self.axs.plot(t, normalized_respiration_signal, label=f'Respiration Peaks: {self.rr_total}')
            self.axs.plot(t[self.rr_peaks], normalized_respiration_signal[self.rr_peaks], 'ro')
            self.axs.set_ylabel("Temperature (C)", color=color.BLUE, size= 18)
            self.axs.set_xlabel("Time (s)", color=color.BLUE, size= 18)
            self.axs.grid()
            self.axs.legend()
            self.axs.tick_params(axis="both", colors=color.BLUE)
            self.fig.tight_layout()
            self.chart.update()

            self.axs1.clear()
            self.axs1.plot(t, normalized_heart_rate_signal, label=f'Heart Rate Peaks: {len(self.hr_peaks)}')
            self.axs1.plot(t[self.hr_peaks], normalized_heart_rate_signal[self.hr_peaks], 'ro')
            self.axs1.set_ylabel("Temperature (C)", color=color.BLUE, size= 18)
            self.axs1.set_xlabel("Time (s)", color=color.BLUE, size= 18)
            self.axs1.grid()
            self.axs1.legend()
            self.axs1.tick_params(axis="both", colors=color.BLUE)
            self.fig1.tight_layout()
            self.chart1.update()

            self.axs2.clear()
            self.axs2.plot(t, VitalSignProcess.normalize_signal(self.respiration_signal, self.suhu_min, self.suhu_max), label=f'Respiration Signal')
            self.axs2.set_ylabel("Temperature (C)", color=color.BLUE, size= 18)
            self.axs2.set_xlabel("Time (s)", color=color.BLUE, size= 18)
            self.axs2.grid()
            self.axs2.legend()
            self.axs2.tick_params(axis="both", colors=color.BLUE)
            self.fig2.tight_layout()
            self.chart2.update()

            self.axs3.clear()
            self.axs3.plot(t, VitalSignProcess.normalize_signal(self.heart_rate_signal, self.suhu_min, self.suhu_max), label=f'Heart Rate Signal')
            self.axs3.set_ylabel("Temperature (C)", color=color.BLUE, size= 18)
            self.axs3.set_xlabel("Time (s)", color=color.BLUE, size= 18)
            self.axs3.grid()
            self.axs3.legend()
            self.axs3.tick_params(axis="both", colors=color.BLUE)
            self.fig3.tight_layout()
            self.chart3.update()
            
            self.status_title.value = 'Done'
            self.status_title.color = color.WHITE
            self.status_title.update()
        else:
            self.status_title.value = 'Select your video first !'
            self.status_title.color = color.ORANGE
            self.status_title.update()

    def upload_and_play(self, e: FilePickerResultEvent):
        self.respiration_signal = []
        self.heart_rate_signal = []
        self.respiration_rate = 0
        self.heart_rate  = 0
        self.temp_in_celsius=0

        self.resp_count.value = f"{self.respiration_rate}"
        self.resp_count.update()
        self.hr_count.value = f"{self.heart_rate}"
        self.hr_count.update()
        self.temperature.value = f"{self.heart_rate}"
        self.temperature.update()

        if e.files:
            self.filepath = e.files[0].path
            self.filename = e.files[0].name
            self.status_title.value = f'{e.files[0].name}'
            self.status_title.update()

            cap = cv2.VideoCapture(self.filepath)
            self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

            video_thread = Thread(target=self.play_video, args=(self.video_box, cap))
            video_thread.daemon = True
            video_thread.start()

    def savenow(self, e:FilePickerResultEvent):
            self.location = e.path
            if self.location:
                try:
                    with open(self.location, "w", encoding="utf-8") as file:
                        file.write("VISMOIR REPORT\n")
                        file.write(f"File Path : {self.filename}\n")
                        file.write(f"Frame Rate : {self.frame_rate} fps\n")
                        file.write(f"Epsilon : {self.gtv_eps}\n")
                        header = "n Data\tTime(s)\tRespiration Signal\tHeart Signal"
                        np.savetxt(file, self.allData.T, header=header, comments='', fmt='%d')
                        self.page.snack_bar = SnackBar(
                        Text("File downloaded", size=15, font_family=font.REGULER, color=color.WHITE),
                        bgcolor="green"
                    )
                    self.page.snack_bar.open = True
                    self.page.update()
                except Exception:
                    self.page.snack_bar = SnackBar(
                        Text("Download failed", size=15, font_family=font.REGULER, color=color.WHITE),
                        bgcolor="red"
                    )
                    self.page.snack_bar.open = True
                    self.page.update()

    def build(self):
        file_picker = FilePicker(on_result=self.upload_and_play)
        save_file = FilePicker(
            on_result=self.savenow
        )
        self.page.overlay.append(file_picker)
        self.page.overlay.append(save_file)
        self.start_button = ElevatedButton(
            height=50,
            width= 180,
            style= ButtonStyle(
                elevation=0,
                color={MaterialState.DEFAULT: color.WHITE},
                bgcolor={MaterialState.HOVERED: color.BLUE_1, "":color.ORANGE},
                shape={MaterialState.DEFAULT:RoundedRectangleBorder(radius=30)},
            ),
            content= Text("Let's Analyze !", size= 20, text_align="center", font_family=font.REGULER),
            on_click=lambda _: file_picker.pick_files()
        )

        self.end_button = ElevatedButton(
            height=50,
            width= 180,
            style= ButtonStyle(
                elevation=0,
                color={MaterialState.DEFAULT: color.WHITE},
                bgcolor={MaterialState.HOVERED: color.BLUE_1, "":color.ORANGE},
                shape={MaterialState.DEFAULT:RoundedRectangleBorder(radius=30)},
            ),
            content= Text("Download", size= 20, text_align="center", font_family=font.REGULER),
            on_click=lambda _: save_file.save_file()
        )

        return Column([
            Container(height=100),
            Row(
                controls=[
                    Row(
                        [
                            Column(
                                horizontal_alignment=CrossAxisAlignment.CENTER,
                                controls=[
                                    self.video_box,
                                    self.status_title,
                                    Container(height=20),
                                    self.start_button,
                                    self.end_button
                                ]
                            ),
                            Column(
                                horizontal_alignment=CrossAxisAlignment.CENTER,
                                controls=[
                                    self.resp_count_title_box,
                                    self.resp_count_box,
                                    self.hr_count_title_box,
                                    self.hr_count_box,
                                    self.temperature_title_box,
                                    self.temperature_box,
                                    Container(height=175)
                                ]
                            ),
                        ], alignment=MainAxisAlignment.CENTER, spacing=30
                    ),
                    Container(width=100),
                    Row(
                        [
                            Column(
                                horizontal_alignment=CrossAxisAlignment.CENTER,
                                controls=[
                                    Container(self.title_setup,  width=200, height=50, padding=5, border_radius=5, bgcolor=color.PURPLE, alignment=alignment.center),
                                    self.max_temp_box,
                                    self.min_temp_box,
                                    Container(height=10),
                                    Container(self.title_GLR,  width=200, padding=5, border_radius=5, bgcolor=color.PURPLE, alignment=alignment.center),
                                    self.glr_alpha_box,
                                    Container(height=150)
                                ], spacing=20
                            ),
                            Column(
                                horizontal_alignment=CrossAxisAlignment.CENTER,
                                controls=[
                                    Container(self.title_GTV,  width=200, padding=5, border_radius=5, bgcolor=color.PURPLE, alignment=alignment.center),
                                    self.gtv_weight_box,
                                    self.gtv_eps_box,
                                    Container(height=315)
                                ], spacing= 20
                            ),
                        ], alignment=MainAxisAlignment.CENTER, spacing=30
                    ),
                    
                ],  alignment=MainAxisAlignment.SPACE_AROUND
            ),
            Container(height= 100),
            Container(
                Column(
                    horizontal_alignment=CrossAxisAlignment.CENTER,
                    controls=[
                        self.advance_title,
                        Text('Respiration Signal', size= 25, color=color.BLUE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER),
                        Container(
                            self.chart2,
                            width=1200,
                            height=600,
                        ),
                        Text('Heart Rate Signal', size= 25, color=color.BLUE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER),
                        Container(
                            self.chart3,
                            width=1200,
                            height=600,
                        ),
                        Text('Respiration Signal Peaks', size= 25, color=color.BLUE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER),
                        Container(
                            self.chart,
                            width=1200,
                            height=600,
                        ),
                        Text('Heart Rate Signal Peaks', size= 25, color=color.BLUE, font_family=font.SEMIBOLD, selectable=True, no_wrap=True, text_align=TextAlign.CENTER),
                        Container(
                            self.chart1,
                            width=1200,
                            height=600,
                        ),
                    ]
                ),
                bgcolor=color.WHITE, width=2000, border_radius=50, padding=padding.only(10,10,10, 50)
            ),
        ], horizontal_alignment= CrossAxisAlignment.CENTER)

class BrandingPagey(UserControl):
    def __init__(self, page, bgcolor):
        super().__init__()
        self.page = page
        self.bgcolor = bgcolor
        self.simple_video_player = SimpleVideoPlayer(page)

    def build(self):
        return Column(
            horizontal_alignment=CrossAxisAlignment.CENTER,
            controls=[
                Container(
                    padding=70,
                    content=Column(
                        horizontal_alignment=CrossAxisAlignment.CENTER,
                        controls=[
                            self.simple_video_player.build()
                        ]
                    ),
                    bgcolor=color.BLUE
                )
            ]
        )

class BrandingPage(UserControl):
    def __init__(self, page, bgcolor):
        super().__init__()
        self.page = page
        self.bgcolor = bgcolor
        self.title = Text("ViSMoIR", size= 120, color= color.BLUE, font_family= font.AMSTERDAM, no_wrap=True, selectable=True, text_align=TextAlign.CENTER)
        self.title_2 = Text("Sistem Deteksi dan Klasifikasi Vital Sign secara Non-Kontak Menggunakan Infrared Thermography", size= 35, color= color.BLUE, font_family= font.MEDIUM, no_wrap=True, selectable=True, text_align=TextAlign.CENTER)

        
    def build(self):
        return Column(
            horizontal_alignment= CrossAxisAlignment.CENTER,
            controls=[
                Container(
                    padding= padding.only(40,200,40,250),
                    content= Column(
                        horizontal_alignment= CrossAxisAlignment.CENTER,
                        controls=[
                            self.title,
                            Container(
                                self.title_2,
                                width=1000,
                                alignment=alignment.center
                            )
                        ]
                    )
                )
            ]
        )

def main(page: Page):
    page.padding = padding.all(0)
    page.scroll = ScrollMode.AUTO
    page.horizontal_alignment = CrossAxisAlignment.CENTER
    page.bgcolor = color.WHITE
    page.fonts = {
        "Bold"          :"/fonts/Poppins-Bold.ttf",
        "ExtraBold"     :"/fonts/Poppins-ExtraBold.ttf",
        "Italic"        :"/fonts/Poppins-Italic.ttf",
        "Medium"        :"/fonts/Poppins-Medium.ttf",
        "Reguler"       :"/fonts/Poppins-Reguler.ttf",
        "SemiBold"      :"/fonts/Poppins-SemiBold.ttf",
        "Thin"          :"/fonts/Poppins-Thin.ttf",
        "Bebas"         :"/fonts/BebasNeue-Regular.ttf",
        "Amsterdam"     :"/fonts/NewAmsterdam-Regular.ttf",
        "Staat"         :"/fonts/Staatliches-Regular.ttf",
    }
    page.add(
        BrandingPage(page, color.WHITE),
        BrandingPagey(page, color.WHITE),
    )
    page.update()
flet.app(target=main, assets_dir="assets")
