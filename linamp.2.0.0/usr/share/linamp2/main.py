import sys
import os
import json
import locale
import gettext
import random
import threading
import subprocess
import numpy as np
import mpv
from mutagen import File as MutagenFile

from PyQt6.QtCore import Qt, QTimer, QRect, QSize, QUrl, QMimeData
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QListWidget, QListWidgetItem,
    QFrame, QFileDialog, QSplitter, QStyle, QMessageBox
)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QDragEnterEvent, QDropEvent, QIcon, QMovie, QFontDatabase, QAction, QPalette

# Uygulamanın nereden çağrıldığından bağımsız, kendi çalıştığı mutlak dizinler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.join(BASE_DIR, "gui-elements")
LOCALE_DIR = os.path.join(BASE_DIR, "locale")
CONFIG_DIR = os.path.expanduser("~/.config/Linamp2")
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")

# ==============================================================================
# GETTEXT ÇOKLU DİL (I18N) ALTYAPISI
# ==============================================================================
CURRENT_LANG = "en"

def setup_translation(lang_code="en"):
    """Seçilen dile göre gettext çevirisini yükler, dosya yoksa İngilizce kalır."""
    global _, CURRENT_LANG
    CURRENT_LANG = lang_code
    try:
        trans = gettext.translation(
            "linamp2",
            localedir=LOCALE_DIR,
            languages=[lang_code],
            fallback=True
        )
        trans.install()
        _ = trans.gettext
    except Exception:
        _ = lambda s: s

# Başlangıçta İngilizce yükle
_ = lambda s: s
setup_translation("en")

class TrackAnalyzer:
    """Parçayı ffmpeg ile çözüp 30 kare/sn'lik 6 bantlı vumetre verisi üretir.

    Analiz, ses slider'ından ve ses sunucusundan tamamen bağımsızdır; parçanın
    kendi sinyalini kullanır. Arka planda gerçek zamandan çok daha hızlı çalışır.
    """
    FPS = 30                # saniyede kare sayısı
    RATE = 44100            # analiz örnekleme hızı (mono)
    FFT_N = 1024            # FFT pencere boyu
    BINS = [(1, 6), (6, 15), (15, 36), (36, 95), (95, 230), (230, 420)]
    SPAN_DB = 35.0          # ekranın kapsadığı dB aralığı (ayar: 25-45)
    PERCENTILE = 98.0       # bant tavanı: karelerin %98'inin altında kaldığı seviye
    SYNC_OFFSET = 0.0       # saniye; çubuklar sesten önde ise eksi, geride ise artı yap

    def __init__(self, path):
        self.path = path
        self.db_frames = []                    # her kare için 6 bant dB değeri
        self.ref_db = np.full(6, -40.0)        # bant başına tavan (dB)
        self.done = False
        self.failed = False
        self._stop = False
        self._proc = None
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        """Analizi durdurur ve ffmpeg sürecini kapatır."""
        self._stop = True
        try:
            if self._proc:
                self._proc.terminate()
        except Exception:
            pass

    def _update_ref(self):
        """Şimdiye kadarki karelerden bant tavanlarını hesaplar."""
        if not self.db_frames:
            return
        arr = np.asarray(self.db_frames)
        ref = np.percentile(arr, self.PERCENTILE, axis=0)
        ref = np.maximum(ref, ref.max() - 55.0)   # boş bir bandı gürültüyle şişirme
        self.ref_db = np.maximum(ref, -100.0)     # tamamen sessiz parçayı şişirme

    def _run(self):
        hop = self.RATE // self.FPS
        n = self.FFT_N
        window = np.hanning(n).astype(np.float32)
        fft_ref = n / 4.0                      # tam ölçekli sinüsün Hann pencereli tepesi

        cmd = [
            'ffmpeg', '-v', 'error', '-nostdin',
            '-i', self.path,
            '-vn', '-ac', '1', '-ar', str(self.RATE),
            '-f', 's16le', '-'
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[Linamp2] ffmpeg başlatılamadı: {e}")
            self.failed = True
            return

        buf = np.zeros(0, dtype=np.float32)
        chunks = 0
        try:
            while not self._stop:
                raw = self._proc.stdout.read(self.RATE * 2)   # yaklaşık 1 saniyelik ses
                if not raw:
                    break
                raw = raw[: len(raw) // 2 * 2]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                buf = np.concatenate([buf, samples])
                if len(buf) < n:
                    continue

                count = (len(buf) - n) // hop + 1
                idx = np.arange(count)[:, None] * hop + np.arange(n)[None, :]
                mag = np.abs(np.fft.rfft(buf[idx] * window, axis=1))
                cols = [np.mean(mag[:, s:e], axis=1) for (s, e) in self.BINS]
                vals = np.stack(cols, axis=1) / fft_ref
                self.db_frames.extend(20.0 * np.log10(vals + 1e-9))
                buf = buf[count * hop:]

                chunks += 1
                if (chunks & (chunks - 1)) == 0:   # 1., 2., 4., 8. ... saniyelerde güncelle
                    self._update_ref()
        except Exception as e:
            print(f"[Linamp2] Parça analizi hatası: {e}")

        if self._stop:
            return
        if not self.db_frames:
            self.failed = True                     # çözülemedi -> parec yedeği devreye girer
            return
        self._update_ref()
        self.done = True

    def bands_at(self, pos):
        """Verilen saniyedeki 6 bant değerini (0.0-1.0) döndürür; veri yoksa None."""
        k = int((pos + self.SYNC_OFFSET) * self.FPS)
        if k < 0 or k >= len(self.db_frames):
            return None
        level = (self.db_frames[k] - (self.ref_db - self.SPAN_DB)) / self.SPAN_DB
        return np.clip(level, 0.0, 1.0).tolist()
# ==============================================================================
# TIKLANAN YERE ANINDA ZIPLAYAN SLIDER (JUMP TO CLICK)
# ==============================================================================
class ClickableSlider(QSlider):
    """Tıklanan piksel konumuna doğrudan zıplayan ve pürüzsüz sürünen slider."""
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                int(event.position().x()), self.width()
            )
            self.setValue(val)
            self.sliderMoved.emit(val)
            self.setSliderDown(True)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                int(event.position().x()), self.width()
            )
            self.setValue(val)
            self.sliderMoved.emit(val)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(False)
        super().mouseReleaseEvent(event)


# ==============================================================================
# GÖRSELLEŞTİRME BİLEŞENİ (SPECTRUM ANALYZER & VU METER)
# ==============================================================================
class AudioVisualizerWidget(QFrame):
    """Solda 126x97 GIF animasyonu, sağda 8-band spektrum barındıran bileşen."""
    def __init__(self, gui_dir, parent=None):
        super().__init__(parent)
        self.gui_dir = gui_dir
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(105)  # 97px GIF ve kenarlıklar için ideal yükseklik
        
        # 1. SOL-ÜST: Durum GIF Oynatıcısı (Yarı yarıya: 63x48 px)
        self.lbl_status_gif = QLabel(self)
        self.lbl_status_gif.setGeometry(6, 6, 63, 48)
        self.lbl_status_gif.setScaledContents(True)

        # Şimdilik sadece stopped.gif gösteriliyor
        stopped_path = os.path.join(self.gui_dir, "stopped.gif")
        self.status_movie = QMovie(stopped_path)
        self.status_movie.setScaledSize(QSize(63, 48))
        self.lbl_status_gif.setMovie(self.status_movie)
        self.status_movie.start()

        # Özel Fontu Yükle (Doto-VariableFont_ROND,wght.ttf - mutlak dizinden)
        font_path = os.path.join(BASE_DIR, "Doto-VariableFont_ROND,wght.ttf")
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            custom_font = QFont(font_family, 14, QFont.Weight.Bold)
            time_font = QFont(font_family, 14, QFont.Weight.Bold)
        else:
            custom_font = QFont("Monospace", 9, QFont.Weight.Bold)
            time_font = QFont("Monospace", 9, QFont.Weight.Bold)

        # SOL-ORTA: Playing GIF'in Altındaki Dijital Süre Göstergesi (Turuncu)
        self.lbl_time = QLabel("00:00", self)
        self.lbl_time.setFont(time_font)
        self.lbl_time.setStyleSheet("color: #00ff94; background: transparent;")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_time.setGeometry(6, 56, 63, 22)

        # Ekranın En Altındaki Şarkı Başlığı / Kayan Yazı Etiketi
        self.lbl_title = QLabel(_("Ready"), self)
        self.lbl_title.setFont(custom_font)
        self.lbl_title.setStyleSheet("color: #ffc800; background: transparent;")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Kayan Yazı (Marquee) Değişkenleri ve Zamanlayıcısı
        self.raw_title = _("Ready")
        self.scroll_display_text = _("Ready")
        self.scroll_timer = QTimer(self)
        self.scroll_timer.setInterval(220)  # Akma hızı (milisaniye)
        self.scroll_timer.timeout.connect(self.tick_scroll_text)
        # "Hazır" durumunda kayma olmayacağı için timer'ı burada başlatmıyoruz

        # 3. SAĞ-ÜST: Albüm Kapağı Alanı (64x64 px)
        self.lbl_album_art = QLabel(self)
        self.lbl_album_art.setFixedSize(64, 64)
        self.lbl_album_art.setScaledContents(True)
        self.set_album_art(None)  # Başlangıçta "NO ALBUM ART" göster

        # VU Metre Bant Sayısı ve İlk Değerleri
        self.num_bands = 6
        self.band_values = [0.0] * self.num_bands

    def set_status_animation(self, state):
        """Oynatma durumuna göre (playing, paused, stopped) sol-üstteki GIF'i günceller."""
        gif_files = {
            "playing": "playing.gif",
            "paused": "paused.gif",
            "stopped": "stopped.gif"
        }
        filename = gif_files.get(state, "stopped.gif")
        gif_path = os.path.join(self.gui_dir, filename)
        
        self.status_movie.stop()
        self.status_movie.setFileName(gif_path)
        self.status_movie.setScaledSize(QSize(63, 48))
        self.lbl_status_gif.setMovie(self.status_movie)
        self.status_movie.start()

    def set_title_text(self, text):
        """Yeni şarkı adını alır; 'Ready' ise sabit tutar, şarkı ise kaydırır."""
        self.raw_title = text
        if text == "Ready":
            self.scroll_timer.stop()
            self.lbl_title.setText("Ready")
        else:
            self.scroll_display_text = text + "   ***   "
            self.lbl_title.setText(self.scroll_display_text)
            self.scroll_timer.start()

    def tick_scroll_text(self):
        """Yazıyı bir karakter sola kaydırır."""
        if len(self.scroll_display_text) > 0:
            self.scroll_display_text = self.scroll_display_text[1:] + self.scroll_display_text[0]
            self.lbl_title.setText(self.scroll_display_text)

    def set_album_art(self, image_data):
        """Gömülü albüm kapağını gösterir, yoksa NO ALBUM ART yazar."""
        from PyQt6.QtGui import QPixmap
        if image_data:
            pixmap = QPixmap()
            if pixmap.loadFromData(image_data):
                self.lbl_album_art.setPixmap(pixmap)
                self.lbl_album_art.setStyleSheet("border: 1px solid #333344;")
                return

        # Kapak resmi yoksa alt alta metin göster
        self.lbl_album_art.clear()
        self.lbl_album_art.setText(_("NO\nALBUM\nART"))
        self.lbl_album_art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_album_art.setStyleSheet("""
            color: #555566;
            border: 1px dashed #333344;
            font-size: 8px;
            font-weight: bold;
            background-color: #0d0d14;
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Şarkı adını ekranın tam tabanına yay
        self.lbl_title.setGeometry(8, self.height() - 24, self.width() - 16, 20)
        
        # Albüm kapağının tabanını VU metre tabanıyla milimetrik eşitle
        bottom_y = self.height() - 24
        art_y = bottom_y - self.lbl_album_art.height()
        self.lbl_album_art.move(self.width() - 64 - 8, art_y)

    def update_data(self, bands):
        """Dışarıdan gelen frekans verilerini günceller (İlk 6 bandı alır)."""
        self.band_values = bands[:6]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        width = self.width()
        height = self.height()
        
        # Arka plan (Saf Siyah)
        painter.fillRect(0, 0, width, height, QColor(0, 0, 0))

        # ----------------------------------------------------------------------
        # 2. ORTA TARAF: 6-Band Spektrum Analizörü (VU Metre Çubukları)
        # ----------------------------------------------------------------------
        bar_margin = 3
        spec_x = 63 + 14                     # Soldaki GIF'in bittiği yer
        art_w = 64 + 12                       # Sağdaki Albüm Kapağının kapladığı alan
        spec_w = width - spec_x - art_w       # Ortada kalan net alan
        
        total_margin = bar_margin * (self.num_bands - 1)
        bar_w = max(4, (spec_w - total_margin) // self.num_bands)

        bottom_y = height - 24  # Taban çizgisi (yazının hemen üzeri)
        total_leds = 13         # 5 LED azaltılmış toplam kapasite (18 -> 13)

        for i in range(self.num_bands):
            val = self.band_values[i]
            lit_leds = int(val * total_leds)  # O an yanan LED adedi
            
            x = spec_x + i * (bar_w + bar_margin)

            # Önce tüm 13 LED yuvasını çiz (Sönük / Arka Plan LED'ler)
            for seg in range(total_leds):
                sub_y = bottom_y - (seg * 4)

                if seg < lit_leds:
                    # YANAN LED'LER
                    if seg >= total_leds - 2:      # En üstteki 2 LED -> KIRMIZI
                        col = QColor(255, 50, 50)
                    elif seg >= total_leds - 6:    # Ortadaki 4 LED -> SARI
                        col = QColor(255, 200, 0)
                    else:                          # Alttaki 7 LED -> YEŞİL
                        col = QColor(0, 225, 100)
                else:
                    # SÖNÜK LED YUVALARI (Koyu Mat Retro Kılavuz Çizgileri)
                    col = QColor(18, 24, 20)

                painter.fillRect(x, sub_y - 3, bar_w, 3, col)


# ==============================================================================
# SÜRÜKLE-BIRAK DESTEKLİ PLAYLIST WIDGET
# ==============================================================================
class DropPlaylistWidget(QListWidget):
    """Dosya ve klasör sürükle-bırak desteği olan liste bileşeni."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        files = []
        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path):
                files.append(path)
            elif os.path.isdir(path):
                for root, _, filenames in os.walk(path):
                    for fn in filenames:
                        files.append(os.path.join(root, fn))
        
        if files and hasattr(self.window(), "add_files_to_playlist"):
            self.window().add_files_to_playlist(files)


# ==============================================================================
# ANA UYGULAMA PENCERESİ (Linamp2)
# ==============================================================================
class Linamp2Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Mutlak dizin referansları
        self.base_dir = BASE_DIR
        self.gui_dir = GUI_DIR

        self.setWindowTitle("Linamp2 Music Player")
        self.setFixedSize(380, 555)

        # Uygulama Simgesi (linamp2.png)
        app_icon_path = os.path.join(self.base_dir, "linamp2.png")
        if os.path.exists(app_icon_path):
            self.setWindowIcon(QIcon(app_icon_path))
        
        self.playlist_data = []  # Metadata tutar
        self.current_index = -1

        # mpv çökmesini önlemek için LC_NUMERIC ayarını C standardına çekiyoruz
        try:
            locale.setlocale(locale.LC_NUMERIC, 'C')
        except Exception as e:
            print(f"Locale ayarlanırken uyarı: {e}")

        # mpv Motorunu Başlatma (volume_max=100 ile skala taşmalarını önlüyoruz)
        self.player = mpv.MPV(vo='null', audio_display='no', keep_open='yes', volume_max=100)
        self.is_playing = False

        # Ayarları JSON dosyasından oku
        self.settings = self.read_settings_file()
        self.current_language = self.settings.get("language", "en")
        setup_translation(self.current_language)
        
        # UI Kurulumu
        self.init_ui()

        # Kaydedilmiş ayarları arayüze uygula (Ses, EQ, Liste vb.)
        self.apply_loaded_settings()

        # Dil menüsünü doldur
        self.populate_language_menu()

        # Gerçek Spektrum Verileri için Değişkenler
        self.raw_fft_bands = [0.0] * 6
        self.smooth_bands = [0.0] * 6
        self.pa_stream = None
        self.pa = None
        # Vumetre verisi önce parçanın kendi analizinden (ffmpeg) gelir;
        # ffmpeg yoksa ya da dosya çözülemezse parec yedeği devreye girer
        self.analyzer = None

        # Görselleştirme ve Arayüz Güncelleme Zamanlayıcısı (approx. 30 FPS)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_ui_loop)
        self.timer.start()

    def init_ui(self):
        # Üst Menü Çubuğunu Kur
        self.create_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # 1. BÖLÜM: BİLGİ EKRANI & GÖRSELLEŞTİRME (SİYAH PANEL)
        info_box = QVBoxLayout()
        self.visualizer = AudioVisualizerWidget(self.gui_dir)
        self.lbl_title = self.visualizer.lbl_title  # Başlık artık panelin içinde
        info_box.addWidget(self.visualizer)
        main_layout.addLayout(info_box)

        # 2. BÖLÜM: SÜRE & ARAMA ÇUBUĞU (SEEK BAR)
        seek_layout = QHBoxLayout()
        # Geçen süre artık siyah panelin içinde turuncu olarak gösteriliyor
        self.lbl_time_cur = self.visualizer.lbl_time
        self.slider_seek = ClickableSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.sliderMoved.connect(self.on_seek_moved)
        self.lbl_time_tot = QLabel("00:00")
        
        seek_layout.addWidget(self.slider_seek)
        seek_layout.addWidget(self.lbl_time_tot)
        main_layout.addLayout(seek_layout)

        # İlerletme çubuğu ile butonlar arasındaki dikey boşluk (8 piksel)
        main_layout.addSpacing(8)

        # 3. BÖLÜM: ANA KONTROL BUTONLARI
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(6)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_prev = QPushButton()
        self.btn_play = QPushButton()
        self.btn_pause = QPushButton()
        self.btn_stop = QPushButton()
        self.btn_next = QPushButton()

        # Boyutları özel olarak ayarla (Play: 136x46, Diğerleri: 52x46)
        for b in [self.btn_prev, self.btn_pause, self.btn_stop, self.btn_next]:
            b.setFixedSize(52, 46)
            b.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_play.setFixedSize(136, 46)
        self.btn_play.setCursor(Qt.CursorShape.PointingHandCursor)

        # Pause animasyonu ikon boyutu (52x46)
        self.btn_pause.setIconSize(QSize(52, 46))
        self.pause_movie = QMovie(os.path.join(self.gui_dir, "pause_pushed_animated.gif"))
        self.pause_movie.frameChanged.connect(self.update_pause_icon)

        self.btn_prev.clicked.connect(self.play_prev)
        self.btn_play.clicked.connect(self.play_music)
        self.btn_pause.clicked.connect(self.pause_music)
        self.btn_stop.clicked.connect(self.stop_music)
        self.btn_next.clicked.connect(self.play_next)

        ctrl_layout.addWidget(self.btn_prev)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_next)
        main_layout.addLayout(ctrl_layout)

        # Butonlar ile ses sürgüsü arasındaki dikey boşluk (8 piksel)
        main_layout.addSpacing(8)

        # 3.1 BÖLÜM: İKİNCİL KONTROLLER (TEKRARLA, KARIŞIK ÇAL VE SES)
        sub_ctrl_layout = QHBoxLayout()
        sub_ctrl_layout.setContentsMargins(4, 2, 4, 2)
        sub_ctrl_layout.setSpacing(4)
        
        # Tekrarla ve Karışık Çal Butonları (52x29 px - Toggle / Checkable)
        self.btn_repeat = QPushButton()
        self.btn_repeat.setCheckable(True)
        self.btn_repeat.setFixedSize(52, 29)
        self.btn_repeat.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_repeat.setToolTip(_("Repeat"))

        self.btn_shuffle = QPushButton()
        self.btn_shuffle.setCheckable(True)
        self.btn_shuffle.setFixedSize(52, 29)
        self.btn_shuffle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_shuffle.setToolTip(_("Shuffle"))

        # Ekolayzır Aç/Kapat Butonu (Görselli)
        self.btn_toggle_eq = QPushButton()
        self.btn_toggle_eq.setCheckable(True)
        self.btn_toggle_eq.setChecked(True)  # Başlangıçta açık (eq-on)
        self.btn_toggle_eq.setFixedSize(52, 29)
        self.btn_toggle_eq.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_eq.setToolTip(_("Toggle Equalizer"))
        self.btn_toggle_eq.clicked.connect(self.toggle_equalizer)

        # Sol tarafa butonları yerleştir
        sub_ctrl_layout.addWidget(self.btn_repeat)
        sub_ctrl_layout.addWidget(self.btn_shuffle)
        sub_ctrl_layout.addWidget(self.btn_toggle_eq)
        
        # Araya esnek boşluk koyarak sesi sağa itiyoruz
        sub_ctrl_layout.addStretch()

        lbl_vol = QLabel(_("Vol:"))
        lbl_vol.setFont(QFont("Sans", 8, QFont.Weight.Bold))
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(100)
        self.slider_vol.setFixedWidth(100)
        self.slider_vol.valueChanged.connect(self.on_volume_changed)

        sub_ctrl_layout.addWidget(lbl_vol)
        sub_ctrl_layout.addWidget(self.slider_vol)
        main_layout.addLayout(sub_ctrl_layout)

        # 4. BÖLÜM: EQUALIZER (10 BANT)
        self.eq_frame = QFrame()
        self.eq_frame.setFrameShape(QFrame.Shape.StyledPanel)
        eq_layout = QHBoxLayout(self.eq_frame)
        eq_layout.setContentsMargins(4, 4, 4, 4)
        eq_layout.setSpacing(2)

        self.eq_bands = [60, 170, 310, 600, 1000, 3000, 6000, 12000, 14000, 16000]
        self.eq_sliders = []

        for freq in self.eq_bands:
            band_box = QVBoxLayout()
            band_box.setSpacing(2)
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setValue(0)
            slider.setFixedHeight(70)
            slider.valueChanged.connect(self.apply_equalizer)
            self.eq_sliders.append(slider)

            lbl_f = QLabel(f"{freq if freq < 1000 else str(freq//1000)+'k'}")
            lbl_f.setFont(QFont("Sans", 7))
            lbl_f.setAlignment(Qt.AlignmentFlag.AlignCenter)

            band_box.addWidget(slider, alignment=Qt.AlignmentFlag.AlignCenter)
            band_box.addWidget(lbl_f, alignment=Qt.AlignmentFlag.AlignCenter)
            eq_layout.addLayout(band_box)

        main_layout.addWidget(self.eq_frame)

        # 5. BÖLÜM: PLAYLIST VE DOSYA İŞLEMLERİ
        playlist_btn_layout = QHBoxLayout()
        self.btn_add_file = QPushButton(_("+ Add File(s)"))
        self.btn_clear_list = QPushButton(_("Clear"))
        self.btn_add_file.clicked.connect(self.open_file_dialog)
        self.btn_clear_list.clicked.connect(self.clear_playlist)

        playlist_btn_layout.addWidget(self.btn_add_file)
        playlist_btn_layout.addWidget(self.btn_clear_list)
        playlist_btn_layout.addStretch()
        main_layout.addLayout(playlist_btn_layout)

        self.playlist_widget = DropPlaylistWidget()
        self.playlist_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        main_layout.addWidget(self.playlist_widget)

        # Tüm arayüz kurulduktan sonra buton stillerini uygula
        self.apply_button_styles()

    def init_audio_capture(self):
        """Pardus / PulseAudio hoparlör çıkışını (parec) doğrudan dinleyen arka plan thread'ini başlatır."""
        self.capture_running = True
        self.capture_proc = None

        def capture_worker():
            try:
                # Varsayılan hoparlör sink'ini al
                try:
                    sink = subprocess.check_output(['pactl', 'get-default-sink'], text=True).strip()
                    mon_device = f"{sink}.monitor"
                except Exception:
                    mon_device = "@DEFAULT_MONITOR@"

                # Hoparlörden çıkan sesi doğrudan PCM int16 olarak borudan (pipe) oku
                cmd = [
                    'parec',
                    '-d', mon_device,
                    '--format=s16le',
                    '--rate=44100',
                    '--channels=1',
                    '--latency-msec=30'
                ]
                self.capture_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=2048
                )

                chunk_size = 1024 * 2  # 1024 örnek (her örnek 2 bayt int16)
                bins = [(1, 6), (6, 15), (15, 36), (36, 95), (95, 230), (230, 420)]
                weights = [2.5, 3.2, 4.5, 6.5, 9.5, 14.0]
                                # Bant başına uyarlanır tavan (dB): ses seviyesinden bağımsız çalışır
                span_db = 35.0         # ekranın kapsadığı dB aralığı (ayar: 25-45)
                ceil_min = -60.0       # tavanın inebileceği en düşük değer
                ceilings = [ceil_min] * 6
                fft_ref = 1024 / 4.0   # tam ölçekli sinüsün Hann pencereli tepe değeri

                while self.capture_running:
                    raw_data = self.capture_proc.stdout.read(chunk_size)
                    if not raw_data:
                        print("[Linamp2] parec akışı kapandı, ses yakalama durdu.")
                        break
                    if len(raw_data) < chunk_size:
                        continue

                    if not self.is_playing or getattr(self.player, 'pause', False):
                        self.raw_fft_bands = [0.0] * 6
                        continue

                    # PCM int16 -> float32 normalize (-1.0 ile 1.0)
                    data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    windowed = data * np.hanning(len(data))
                    fft_mag = np.abs(np.fft.rfft(windowed))

                    bands = []
                    for i, ((start, end), w) in enumerate(zip(bins, weights)):
                        val = float(np.mean(fft_mag[start:end]) * w) / fft_ref
                        db = 20.0 * np.log10(val + 1e-9)

                        # Yeni tepe gelince tavan hemen yükselir, sonra yavaşça iner
                        ceilings[i] = max(db, ceilings[i] - 0.15, ceil_min)

                        level = (db - (ceilings[i] - span_db)) / span_db
                        bands.append(max(0.0, min(1.0, level)))

                    self.raw_fft_bands = bands

            except Exception as e:
                print(f"[Linamp2] Ses Yakalama Thread Hatası: {e}")

        # Arka planda donma yapmadan sürekli dinleyen thread
        t = threading.Thread(target=capture_worker, daemon=True)
        t.start()

    # ==========================================================================
    # AYARLAR (SETTINGS JSON) YÖNETİMİ
    # ==========================================================================
    def read_settings_file(self):
        """~/.config/Linamp2/settings.json dosyasını okur; yoksa boş sözlük döner."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Linamp2] Ayar dosyası okunamadı: {e}")
        return {}

    def apply_loaded_settings(self):
        """JSON'dan okunan ayarları ilgili bileşenlere uygular."""
        s = self.settings

        # 1. Ses Seviyesi
        vol = s.get("volume", 100)
        self.slider_vol.setValue(vol)
        self.on_volume_changed(vol)

        # 2. Tekrarla ve Karışık Çal Buton Durumları
        self.btn_repeat.setChecked(s.get("repeat", False))
        self.btn_shuffle.setChecked(s.get("shuffle", False))

        # 3. Ekolayzır Açık/Kapalı Durumu
        eq_on = s.get("equalizer_enabled", True)
        self.btn_toggle_eq.setChecked(eq_on)
        self.toggle_equalizer()

        # 4. Ekolayzır Bant Değerleri (10 Bant)
        gains = s.get("equalizer_gains", [0] * 10)
        for slider, gain in zip(self.eq_sliders, gains):
            slider.setValue(gain)
        self.apply_equalizer()

        # 5. En Son Çalma Listesini Yükle
        playlist_paths = s.get("playlist", [])
        valid_paths = [p for p in playlist_paths if os.path.exists(p)]
        if valid_paths:
            self.add_files_to_playlist(valid_paths)

    def save_settings(self):
        """Mevcut durumları ~/.config/Linamp2/settings.json dosyasına yazar."""
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            data = {
                "language": getattr(self, "current_language", "en"),
                "volume": self.slider_vol.value(),
                "repeat": self.btn_repeat.isChecked(),
                "shuffle": self.btn_shuffle.isChecked(),
                "equalizer_enabled": self.btn_toggle_eq.isChecked(),
                "equalizer_gains": [slider.value() for slider in self.eq_sliders],
                "playlist": [item['path'] for item in self.playlist_data if os.path.exists(item['path'])]
            }
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Linamp2] Ayarlar kaydedilirken hata: {e}")

    def populate_language_menu(self):
        """locale klasöründeki mevcut dilleri tespit edip Language menüsüne ekler."""
        self.menu_language.clear()
        
        # İngilizce varsayılandır
        available_langs = {"en": "English"}

        # locale klasöründeki derlenmiş .mo dillerini tara
        if os.path.exists(LOCALE_DIR):
            for item in os.listdir(LOCALE_DIR):
                mo_path = os.path.join(LOCALE_DIR, item, "LC_MESSAGES", "linamp2.mo")
                if os.path.isfile(mo_path):
                    # İleride genişletilebilir dil isim haritası
                    names = {"tr": "Türkçe", "de": "Deutsch", "fr": "Français", "es": "Español"}
                    available_langs[item] = names.get(item, item.upper())

        from PyQt6.QtGui import QActionGroup
        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)

        for code, name in available_langs.items():
            act = QAction(name, self, checkable=True)
            if code == self.current_language:
                act.setChecked(True)
            act.triggered.connect(lambda checked, c=code: self.change_language(c))
            lang_group.addAction(act)
            self.menu_language.addAction(act)

    def change_language(self, code):
        """Kullanıcı yeni bir dil seçtiğinde ayara kaydeder ve bilgi verir."""
        if self.current_language != code:
            self.current_language = code
            self.save_settings()
            QMessageBox.information(
                self,
                _("Language Changed"),
                _("Language preference saved. Please restart Linamp2 to apply all changes.")
            )

    def create_menu_bar(self):
        """Üst menü çubuğunu ve eylemlerini (Actions) yapılandırır."""
        menubar = self.menuBar()

        # 1. FILE MENÜSÜ
        file_menu = menubar.addMenu(_("&File"))
        
        act_open_files = QAction(_("&Open File(s)..."), self)
        act_open_files.triggered.connect(self.open_file_dialog)
        file_menu.addAction(act_open_files)

        act_open_playlist = QAction(_("Open &Playlist..."), self)
        act_open_playlist.triggered.connect(self.open_playlist_dialog)
        file_menu.addAction(act_open_playlist)

        act_save_playlist = QAction(_("&Save Playlist As..."), self)
        act_save_playlist.triggered.connect(self.save_playlist_dialog)
        file_menu.addAction(act_save_playlist)

        file_menu.addSeparator()

        act_exit = QAction(_("E&xit"), self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # 2. EDIT MENÜSÜ
        edit_menu = menubar.addMenu(_("&Edit"))
        
        act_clear = QAction(_("&Clear Playlist"), self)
        act_clear.triggered.connect(self.clear_playlist)
        edit_menu.addAction(act_clear)

        act_select_all = QAction(_("&Select All"), self)
        edit_menu.addAction(act_select_all)

        act_remove_sel = QAction(_("&Remove Selected"), self)
        edit_menu.addAction(act_remove_sel)

        # 3. HELP MENÜSÜ
        help_menu = menubar.addMenu(_("&Help"))
        
        self.menu_language = help_menu.addMenu(_("&Language"))

        act_about = QAction(_("&About"), self)
        act_about.triggered.connect(self.show_about_dialog)
        help_menu.addAction(act_about)

    def show_about_dialog(self):
        """Uygulama hakkında penceresini gösterir."""
        msg = QMessageBox(self)
        msg.setWindowTitle(_("About Linamp2 Music Player"))

        # Program ikonunu yükle (önce linamp2.png, sonra pencere ikonu)
        from PyQt6.QtGui import QPixmap
        icon_path = os.path.join(self.base_dir, "linamp2.png")
        if os.path.exists(icon_path):
            msg.setIconPixmap(QPixmap(icon_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        elif not self.windowIcon().isNull():
            msg.setIconPixmap(self.windowIcon().pixmap(64, 64))
        else:
            msg.setIcon(QMessageBox.Icon.Information)

        about_desc = _(
            "This is a lightweight, minimal-dependency, fast, beautifully designed music player application "
            "that you can use to listen to music on your Linux computer. It features a pleasant panel display "
            "and a Bar VU meter. It also has an easy-to-use equalizer. It supports opening and creating playlists."
        )
        no_warranty = _("This program comes with ABSOLUTELY NO WARRANTY.")

        about_text = (
            "<h2 style='margin-bottom: 4px;'>Linamp2 Music Player</h2>"
            "<p style='line-height: 140%; margin-top: 0px;'>"
            f"<b>{_('Version:')}</b> 2.0.0<br>"
            f"<b>{_('License:')}</b> GNU GPLv3<br>"
            "<b>GUI/UX:</b> PyQt6<br>"
            f"<b>{_('Programming Language:')}</b> Python 3<br>"
            f"<b>{_('Audio Playback Engine:')}</b> MPV, FFmpeg<br>"
            f"<b>{_('Developer:')}</b> A. Serhat KILIÇOĞLU (shampuan)<br>"
            "<b>GitHub:</b> <a href='https://www.github.com/shampuan' style='color: #00ff94;'>www.github.com/shampuan</a>"
            "</p>"
            f"<p style='text-align: justify; line-height: 130%;'>{about_desc}</p>"
            f"<p><i>{no_warranty}</i></p>"
            "<p style='color: #888888;'>Copyright (C) 2026 - A. Serhat KILIÇOĞLU</p>"
        )

        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(about_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)

        # Tıklanabilir bağlantının varsayılan tarayıcıda açılmasını sağla
        for lbl in msg.findChildren(QLabel):
            lbl.setOpenExternalLinks(True)

        msg.exec()

    def apply_button_styles(self):
        """Tüm buton görsellerini ve durum stillerini yükler."""
        # Dosya yolları (prev-pushed.png tire ile yazıldığı için özel tanımlandı)
        paths = {
            "prev_n": os.path.join(self.gui_dir, "prev_normal.png").replace("\\", "/"),
            "prev_c": os.path.join(self.gui_dir, "prev_cursor.png").replace("\\", "/"),
            "prev_p": os.path.join(self.gui_dir, "prev-pushed.png").replace("\\", "/"),
            "next_n": os.path.join(self.gui_dir, "next_normal.png").replace("\\", "/"),
            "next_c": os.path.join(self.gui_dir, "next_cursor.png").replace("\\", "/"),
            "next_p": os.path.join(self.gui_dir, "next_pushed.png").replace("\\", "/"),
            "stop_n": os.path.join(self.gui_dir, "stop_normal.png").replace("\\", "/"),
            "stop_c": os.path.join(self.gui_dir, "stop_cursor.png").replace("\\", "/"),
            "stop_p": os.path.join(self.gui_dir, "stop_pushed.png").replace("\\", "/"),
            "play_n": os.path.join(self.gui_dir, "play_normal.png").replace("\\", "/"),
            "play_c": os.path.join(self.gui_dir, "play_cursor.png").replace("\\", "/"),
            "play_p": os.path.join(self.gui_dir, "play_pushed.png").replace("\\", "/"),
            "pause_n": os.path.join(self.gui_dir, "pause_normal.png").replace("\\", "/"),
            "pause_c": os.path.join(self.gui_dir, "pause_cursor.png").replace("\\", "/")
        }

        # Anlık bas-bırak butonları (Prev, Next, Stop)
        self.btn_prev.setStyleSheet(f"""
            QPushButton {{ border: none; background: url('{paths["prev_n"]}') center no-repeat; }}
            QPushButton:hover {{ background-image: url('{paths["prev_c"]}'); }}
            QPushButton:pressed {{ background-image: url('{paths["prev_p"]}'); }}
        """)
        self.btn_next.setStyleSheet(f"""
            QPushButton {{ border: none; background: url('{paths["next_n"]}') center no-repeat; }}
            QPushButton:hover {{ background-image: url('{paths["next_c"]}'); }}
            QPushButton:pressed {{ background-image: url('{paths["next_p"]}'); }}
        """)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{ border: none; background: url('{paths["stop_n"]}') center no-repeat; }}
            QPushButton:hover {{ background-image: url('{paths["stop_c"]}'); }}
            QPushButton:pressed {{ background-image: url('{paths["stop_p"]}'); }}
        """)

        # Tekrarla (Repeat) Stili
        rep_n = os.path.join(self.gui_dir, "repeat_normal.png").replace("\\", "/")
        rep_c = os.path.join(self.gui_dir, "repeat_cursor.png").replace("\\", "/")
        rep_p = os.path.join(self.gui_dir, "repeat_pushed.png").replace("\\", "/")
        self.btn_repeat.setStyleSheet(f"""
            QPushButton {{ border: none; background: url('{rep_n}') center no-repeat; }}
            QPushButton:hover {{ background-image: url('{rep_c}'); }}
            QPushButton:checked {{ background-image: url('{rep_p}'); }}
        """)

        # Karışık Çal (Shuffle) Stili
        shf_n = os.path.join(self.gui_dir, "shuffle_normal.png").replace("\\", "/")
        shf_c = os.path.join(self.gui_dir, "shuffle_cursor.png").replace("\\", "/")
        shf_p = os.path.join(self.gui_dir, "shuffle_pushed.png").replace("\\", "/")
        self.btn_shuffle.setStyleSheet(f"""
            QPushButton {{ border: none; background: url('{shf_n}') center no-repeat; }}
            QPushButton:hover {{ background-image: url('{shf_c}'); }}
            QPushButton:checked {{ background-image: url('{shf_p}'); }}
        """)

        # EQ Butonu Stili (eq-off / eq-on)
        eq_off = os.path.join(self.gui_dir, "eq-off.png").replace("\\", "/")
        eq_on = os.path.join(self.gui_dir, "eq-on.png").replace("\\", "/")
        self.btn_toggle_eq.setStyleSheet(f"""
            QPushButton {{ border: none; background: url('{eq_off}') center no-repeat; }}
            QPushButton:checked {{ background-image: url('{eq_on}'); }}
        """)

        self.paths = paths

        # Duruma göre değişen butonlar (Play & Pause)
        self.paths = paths
        self.set_playback_ui_state("stopped")

    def set_playback_ui_state(self, state):
        """Oynatma durumuna göre Play/Pause butonlarını ve ekrandaki durum GIF'ini günceller."""
        # Ekrandaki durum GIF'ini güncelle
        if hasattr(self, 'visualizer'):
            self.visualizer.set_status_animation(state)

        if state == "playing":
            # Play basılı kalır, Pause normal haline döner
            self.pause_movie.stop()
            self.btn_pause.setIcon(QIcon())
            self.btn_play.setStyleSheet(f"QPushButton {{ border: none; background: url('{self.paths['play_p']}') center no-repeat; }}")
            self.btn_pause.setStyleSheet(f"""
                QPushButton {{ border: none; background: url('{self.paths['pause_n']}') center no-repeat; }}
                QPushButton:hover {{ background-image: url('{self.paths['pause_c']}'); }}
            """)
        elif state == "paused":
            # Play normal haline döner, Pause animated gif'i oynatır
            self.btn_play.setStyleSheet(f"""
                QPushButton {{ border: none; background: url('{self.paths['play_n']}') center no-repeat; }}
                QPushButton:hover {{ background-image: url('{self.paths['play_c']}'); }}
            """)
            self.btn_pause.setStyleSheet("QPushButton { border: none; background-color: transparent; }")
            self.pause_movie.start()
        elif state == "stopped":
            # İkisi de normal haline döner
            self.pause_movie.stop()
            self.btn_pause.setIcon(QIcon())
            self.btn_play.setStyleSheet(f"""
                QPushButton {{ border: none; background: url('{self.paths['play_n']}') center no-repeat; }}
                QPushButton:hover {{ background-image: url('{self.paths['play_c']}'); }}
            """)
            self.btn_pause.setStyleSheet(f"""
                QPushButton {{ border: none; background: url('{self.paths['pause_n']}') center no-repeat; }}
                QPushButton:hover {{ background-image: url('{self.paths['pause_c']}'); }}
            """)

    def update_pause_icon(self):
        """Pause GIF animasyonunun her karesini buton ikonuna basar."""
        if self.pause_movie.state() == QMovie.MovieState.Running:
            self.btn_pause.setIcon(QIcon(self.pause_movie.currentPixmap()))

    # ==========================================================================
    # MEDYA VE OYNATMA MANTIĞI
    # ==========================================================================
    def add_files_to_playlist(self, file_paths):
        valid_exts = ('.mp3', '.flac', '.wav', '.ogg', '.m4a')
        for path in file_paths:
            if path.lower().endswith(valid_exts):
                # Dosya adını uzantısız olarak alıyoruz
                file_name = os.path.splitext(os.path.basename(path))[0]
                
                item = QListWidgetItem(file_name)
                self.playlist_widget.addItem(item)
                
                self.playlist_data.append({
                    'path': path,
                    'title': file_name
                })

    def open_file_dialog(self):
        home_dir = os.path.expanduser("~")
        files, _ = QFileDialog.getOpenFileNames(
            self, _("Select Audio Files"), home_dir, _("Audio Files (*.mp3 *.flac *.wav *.ogg *.m4a)")
        )
        if files:
            self.add_files_to_playlist(files)

    def open_playlist_dialog(self):
        """M3U / M3U8 çalma listelerini okur ve listeye ekler."""
        home_dir = os.path.expanduser("~")
        pl_path, _ = QFileDialog.getOpenFileName(
            self, _("Open Playlist"), home_dir, _("Playlist Files (*.m3u *.m3u8)")
        )
        if not pl_path:
            return

        pl_dir = os.path.dirname(pl_path)
        tracks = []
        try:
            with open(pl_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    # Açıklama satırlarını ve boşlukları atla
                    if not line or line.startswith('#'):
                        continue
                    
                    # Göreceli (relative) yol ise tam yola çevir
                    if not os.path.isabs(line):
                        full_path = os.path.normpath(os.path.join(pl_dir, line))
                    else:
                        full_path = line

                    if os.path.exists(full_path):
                        tracks.append(full_path)

            if tracks:
                self.add_files_to_playlist(tracks)
        except Exception as e:
            print(f"Playlist okuma hatası: {e}")

    def save_playlist_dialog(self):
        """Mevcut çalma listesini (boş olsa dahi) M3U dosyası olarak kaydeder."""
        home_dir = os.path.expanduser("~")
        pl_path, _ = QFileDialog.getSaveFileName(
            self, _("Save Playlist As"), home_dir, _("Playlist Files (*.m3u *.m3u8)")
        )
        if not pl_path:
            return

        # Uzantı yazılmadıysa otomatik .m3u ekle
        if not pl_path.lower().endswith(('.m3u', '.m3u8')):
            pl_path += '.m3u'

        try:
            with open(pl_path, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")
                for item in self.playlist_data:
                    f.write(f"{item['path']}\n")
        except Exception as e:
            print(f"Playlist kaydetme hatası: {e}")

    def clear_playlist(self):
        self.stop_music()
        self.playlist_widget.clear()
        self.playlist_data.clear()
        self.current_index = -1
        self.visualizer.set_title_text("Ready")
        self.visualizer.set_album_art(None)

    def extract_album_art(self, file_path):
        """Mutagen ile ses dosyasındaki gömülü albüm kapağı baytlarını çeker."""
        try:
            audio = MutagenFile(file_path)
            if audio is None:
                return None
            # MP3 (ID3 APIC)
            if hasattr(audio, 'tags') and audio.tags:
                for tag in audio.tags.values():
                    if tag.FrameID == 'APIC':
                        return tag.data
            # FLAC
            if hasattr(audio, 'pictures') and audio.pictures:
                return audio.pictures[0].data
        except Exception:
            pass
        return None

    def play_index(self, index):
        if 0 <= index < len(self.playlist_data):
            self.current_index = index
            item_info = self.playlist_data[index]
            self.playlist_widget.setCurrentRow(index)
            self.player.play(item_info['path'])
            self.player.pause = False
            self.is_playing = True

            # Vumetre analizi: aynı parça tekrar çalınıyorsa mevcut analizi kullan
            if not (self.analyzer and self.analyzer.path == item_info['path'] and not self.analyzer.failed):
                if self.analyzer:
                    self.analyzer.stop()
                self.analyzer = TrackAnalyzer(item_info['path'])
            self.set_playback_ui_state("playing")
            
            # Ekolayzır ayarlarını çalan parçaya uygula
            self.apply_equalizer()

            # Kayan yazıya temiz dosya adını gönder ve Albüm Kapağını Güncelle
            self.visualizer.set_title_text(item_info['title'])
            
            art_data = self.extract_album_art(item_info['path'])
            self.visualizer.set_album_art(art_data)

    def play_music(self):
        if self.current_index == -1 and self.playlist_data:
            self.play_index(0)
        elif self.current_index != -1:
            if not self.is_playing:
                # Şarkı bittiyse veya stop edildiyse baştan yükleyip çal
                self.play_index(self.current_index)
            else:
                self.player.pause = False
                self.set_playback_ui_state("playing")

    def pause_music(self):
        if self.current_index != -1:
            is_paused = not getattr(self.player, 'pause', False)
            self.player.pause = is_paused
            if is_paused:
                self.set_playback_ui_state("paused")
            else:
                self.set_playback_ui_state("playing")

    def stop_music(self):
        self.is_playing = False
        self.player.stop()
        self.set_playback_ui_state("stopped")
        self.slider_seek.setValue(0)
        self.lbl_time_cur.setText("00:00")
        self.visualizer.update_data([0.0] * 6)

    def on_track_finished(self):
        """Şarkı bittiğinde Repeat ve Shuffle ayarlarına göre sonraki adımı belirler."""
        if not self.playlist_data:
            self.stop_music()
            return

        is_repeat = self.btn_repeat.isChecked()
        is_shuffle = self.btn_shuffle.isChecked()

        # Karışık Çalma Modu
        if is_shuffle:
            if len(self.playlist_data) > 1:
                # Çalan şarkı dışındaki diğer şarkılardan rastgele seç
                candidates = [i for i in range(len(self.playlist_data)) if i != self.current_index]
                self.play_index(random.choice(candidates))
            elif is_repeat:
                self.play_index(0)
            else:
                self.stop_music()
            return

        # Sıralı Çalma Modu
        if self.current_index + 1 < len(self.playlist_data):
            self.play_index(self.current_index + 1)
        elif is_repeat:
            # Listenin sonuna gelindiğinde Repeat açıksa en başa dön
            self.play_index(0)
        else:
            # Listenin sonuna gelindiğinde Repeat kapalıysa dur
            self.stop_music()

    def play_next(self):
        if not self.playlist_data:
            return

        if self.btn_shuffle.isChecked() and len(self.playlist_data) > 1:
            candidates = [i for i in range(len(self.playlist_data)) if i != self.current_index]
            self.play_index(random.choice(candidates))
        elif self.current_index + 1 < len(self.playlist_data):
            self.play_index(self.current_index + 1)
        elif self.btn_repeat.isChecked():
            # Listenin sonundayken Next'e basılırsa ve Repeat açıksa başa sar
            self.play_index(0)

    def play_prev(self):
        if not self.playlist_data:
            return

        # Şarkı 3 saniyeden fazla ilerlemişse önce parçayı başa sar
        pos = getattr(self.player, 'time_pos', 0) or 0
        if pos > 3:
            self.player.time_pos = 0
            return

        if self.btn_shuffle.isChecked() and len(self.playlist_data) > 1:
            candidates = [i for i in range(len(self.playlist_data)) if i != self.current_index]
            self.play_index(random.choice(candidates))
        elif self.current_index - 1 >= 0:
            self.play_index(self.current_index - 1)
        elif self.btn_repeat.isChecked():
            # İlk şarkıdayken Prev'e basılırsa ve Repeat açıksa listenin sonuna git
            self.play_index(len(self.playlist_data) - 1)

    def on_item_double_clicked(self, item):
        row = self.playlist_widget.row(item)
        self.play_index(row)

    def on_volume_changed(self, value):
        if value <= 0:
            self.player.volume = 0
        else:
            # Algısal (Perceptual) ses eğrisi:
            # Doğrusal slider hareketini kulağın duyum hassasiyetine eşit dağıtır
            factor = value / 100.0
            adjusted_volume = (factor ** 0.5) * 100.0
            self.player.volume = adjusted_volume

    def on_seek_moved(self, value):
        duration = self.player.duration
        if duration:
            target_pos = (value / 1000.0) * duration
            self.player.time_pos = target_pos

    def closeEvent(self, event):
        """Uygulama kapanırken ses yakalama sürecini güvenle sonlandırır ve ayarları kaydeder."""
        self.save_settings()
        self.capture_running = False
        if self.analyzer:
            self.analyzer.stop()
        try:
            if self.capture_proc:
                self.capture_proc.terminate()
        except Exception:
            pass
        super().closeEvent(event)

    # ==========================================================================
    # EQUALIZER MANTIĞI (mpv Audio Filter)
    # ==========================================================================
    def toggle_equalizer(self):
        """Ekolayzırı gizler veya gösterir, pencere yüksekliğini ve ses filtresini günceller."""
        is_visible = self.btn_toggle_eq.isChecked()
        self.eq_frame.setVisible(is_visible)
        
        if is_visible:
            self.setFixedSize(380, 580)
        else:
            self.setFixedSize(380, 475)

        # Filtreyi aç veya kapa
        self.apply_equalizer()

    def apply_equalizer(self):
        """10-Bant parametrik FFmpeg ekolayzır filtresini gerçek zamanlı uygular."""
        # Eğer EQ butonu kapalıysa filtreyi temizle (Flat/Saf ses)
        if not self.btn_toggle_eq.isChecked():
            try:
                self.player.af = ""
            except Exception:
                pass
            return

        gains = [slider.value() for slider in self.eq_sliders]

        # Eğer tüm bantlar 0 dB ise filtre çalıştırmayıp sesi saf bırak
        if all(g == 0 for g in gains):
            try:
                self.player.af = ""
            except Exception:
                pass
            return

        # FFmpeg equalizer zinciri: 1 oktav genişlik (width_type=o:width=1) ile 10 bant
        eq_filters = []
        for freq, gain in zip(self.eq_bands, gains):
            if gain != 0:
                eq_filters.append(f"equalizer=f={freq}:width_type=o:width=1:g={gain}")

        filter_str = ",".join(eq_filters)

        try:
            self.player.af = filter_str
        except Exception as e:
            print(f"Equalizer Hatası: {e}")

    # ==========================================================================
    # ANA ZAMANLAYICI VE GÖRSELLEŞTİRME DÖNGÜSÜ
    # ==========================================================================
    def update_ui_loop(self):
        # Şarkı oynatılıyor durumdaysa bitiş sinyalini denetle
        if self.is_playing:
            try:
                if getattr(self.player, 'eof_reached', False):
                    self.on_track_finished()
                    return
            except Exception:
                pass

        # Duraklatıldıysa veya parça çalmıyorsa görselleştirmeyi sıfırla ve çık
        is_paused = getattr(self.player, 'pause', False)
        if not self.is_playing or is_paused:
            self.visualizer.update_data([0.0] * 6)
            return

        # 1. Süre ve Progress Güncelleme
        try:
            pos = self.player.time_pos or 0
            dur = self.player.duration or 0

            if dur > 0:
                if not self.slider_seek.isSliderDown():
                    self.slider_seek.setValue(int((pos / dur) * 1000))
                
                m_c, s_c = divmod(int(pos), 60)
                m_t, s_t = divmod(int(dur), 60)
                self.lbl_time_cur.setText(f"{m_c:02d}:{s_c:02d}")
                self.lbl_time_tot.setText(f"{m_t:02d}:{s_t:02d}")
        except Exception:
            pass

        # 2. Vumetre verisi: önce parçanın kendi analizi, olmazsa parec yedeği
        source_bands = self.raw_fft_bands
        if self.analyzer is not None:
            if self.analyzer.failed:
                # ffmpeg yok ya da bu dosya çözülemedi -> parec yedeğini bir kez başlat
                if not getattr(self, 'capture_running', False):
                    self.init_audio_capture()
            else:
                try:
                    cur_pos = self.player.time_pos or 0
                except Exception:
                    cur_pos = 0
                analyzed = self.analyzer.bands_at(cur_pos)
                source_bands = analyzed if analyzed is not None else [0.0] * 6

        for i in range(6):
            target = source_bands[i]
            # Hızlı fırla, yavaş ve pürüzsüz süzülerek düş (Decay efekti)
            if target > self.smooth_bands[i]:
                self.smooth_bands[i] = target
            else:
                self.smooth_bands[i] = max(0.0, self.smooth_bands[i] * 0.78)

        self.visualizer.update_data(self.smooth_bands)


# ==============================================================================
# UYGULAMA GİRİŞ NOKTASI
# ==============================================================================
if __name__ == "__main__":
    try:
        locale.setlocale(locale.LC_NUMERIC, 'C')
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("Linamp2 Music Player")
    
    # Uygulama genel ikonu (Linux panel/dock için)
    app_icon_path = os.path.join(BASE_DIR, "linamp2.png")
    if os.path.exists(app_icon_path):
        app.setWindowIcon(QIcon(app_icon_path))

    # Qt Resmi Fusion Koyu Paleti (Adwaita Dark ile birebir aynı 53, 53, 53 tonu)
    app.setStyle("Fusion")
    
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(53, 132, 228))  # Adwaita Mavisi (#3584e4)
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    # Devre dışı (Disabled) durumlar
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(130, 130, 130))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(130, 130, 130))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(130, 130, 130))
    
    app.setPalette(dark_palette)

    window = Linamp2Window()
    window.show()
    sys.exit(app.exec())