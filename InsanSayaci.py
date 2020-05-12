# Gerekli paketleri (kütüphaneleri) dahil edin
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# bağımsız değişken ayrıştırmasını oluşturma ve bağımsız değişkenleri ayrıştırma
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

# Sınıf etiketlerinin listesini başlat
# MobileNet SSD algılamak üzere eğitildi
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# Serileştirilmiş modelimizi diskten yükle
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Eğer bir video yolu sağlanmadıysa, web kamerasına bir referans alın
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# aksi takdirde video dosyasına bir referans alın
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# video yazıcıyı başlat (gerekirse daha sonra başlatacağız)
writer = None

# kare boyutlarını başlatın
# (videodan ilk kareyi okuduğumuzda bunları ayarlayacağız)
W = None
H = None
count = 1
# sentroid izleyicimizi örnekleyin,
# ardından her dlib korelasyon izleyicimizi saklamak için bir liste başlatın,
# ardından her benzersiz nesne kimliğini İzlenebilir Bir Nesne ile eşlemek için bir sözlük izleyin
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# şimdiye kadar işlenen toplam kare sayısını ve
# yukarı veya aşağı hareket eden toplam nesne sayısını başlat
totalFrames = 0
totalDown = 0
totalUp = 0

# saniyedeki kare sayısını başlat verim tahmincisi
fps = FPS().start()

# video akışından kareler üzerinde döngü
while True:
    # Bir sonraki kareyi yakalayın ve VideoCapture veya
    # VideoStream'den okuyorsak işleyin
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # bir video izliyor ve bir kare yakalamamışsak
    # videonun sonuna ulaşmışırız
    if args["input"] is not None and frame is None:
        break

    # çerçeveyi maksimum 500 piksel olacak şekilde yeniden boyutlandırın
    # (ne kadar az veriye sahipsek, o kadar hızlı işleme koyabiliriz),
    # ardından dlib için çerçeveyi BGR'den RGB'ye dönüştürün
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # çerçeve boyutları boşsa, bunları ayarlayın
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # diske bir video yazmamız gerekiyorsa,
    # yazıcıyı başlatın

    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    # (1) nesne detektörümüz veya
    # (2) korelasyon izleyicileri tarafından döndürülen
    # sınırlayıcı kutu dikdörtgenler listemizle birlikte geçerli durumu başlat
    status = "Waiting"
    rects = []

    # izleyicimize yardımcı olmak için daha pahalı
    # bir nesne algılama yöntemi çalıştırıp çalıştırmamamız gerektiğini kontrol edin
    if totalFrames % args["skip_frames"] == 0:
        # durumu ayarlayın ve yeni nesne izleyici setimizi başlatın
        status = "Detecting"
        trackers = []

        # çerçeveyi bir bloba dönüştürün ve
        # blob'u ağ üzerinden geçirin ve tespitleri alın
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Tespitler üzerinde bir döngü başlatın
        for i in np.arange(0, detections.shape[2]):
            # tahminle ilişkili güveni
            # (yani olasılık) çıkarın
            confidence = detections[0, 0, i, 2]

            # minimum güven gerektirerek
            # zayıf tespitleri filtreleyin
            if confidence > args["confidence"]:
                # sınıf etiketinin dizinini
                # algılama listesinden çıkarın
                idx = int(detections[0, 0, i, 1])

                # sınıf etiketi bir kişi değilse, yok sayın
                if CLASSES[idx] != "person":
                    continue

                    # nesne için sınırlama kutusunun
                    # (x, y) koordinatlarını hesapla
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # sınırlayıcı kutu koordinatlarından
                    # bir dlib dikdörtgen nesnesi oluşturun
                    # ve sonra dlib korelasyon izleyicisini başlatın
                    tracker = trim.correlation_tracker()
                    rect = trim.rectangle(startX, startY, endX, endY);
                    tracker.start_track(rgb, rect)

                    # izleyiciyi izleyiciler listemize ekleyin,
                    # böylece kare atlama sırasında kullanın
                    trackers.append(tracker)


    # aksi takdirde, daha yüksek bir çerçeve işleme verimi elde etmek için
    # nesne * dedektörleri * yerine nesne * izleyicilerimizi * kullanmalıyız
    else:
        # izleyiciler üzerinde döngü
        for tracker in trackers:
            # sistemimizin durumunu 'beklemek' veya
            # 'tespit etmek' yerine 'izleme' olarak ayarlayın
            status = "Tracking"

            # izleyiciyi güncelleyin ve güncellenmiş konumu yakalayın
            tracker.update(rgb)
            pos = tracker.get_position()

            # konum nesnesini aç
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # sınırlayıcı kutu koordinatlarını dikdörtgenler listesine ekleyin
            rects.append((startX, startY, endX, endY))

        # çerçevenin ortasına yatay bir çizgi çizin
        # - bir nesne bu çizgiyi geçtiğinde
        # 'yukarı' veya 'aşağı' hareket edip etmediklerini belirleyeceğiz
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

        # (1) eski nesne centroidlerini (2) yeni hesaplanan nesne centroidleri
        # ile ilişkilendirmek için centroid izleyiciyi kullanın
        objects = ct.update(rects)

        # izlenen nesnelerin üzerinde döngü başlatın
        for (objectID, centroid) in objects.items():
            # geçerli nesne kimliği için izlenebilir
            # bir nesne olup olmadığını kontrol edin
            to = trackableObjects.get(objectID, None)

            # izlenebilir bir nesne yoksa bir tane oluşturun
            if to is None:
                to = TrackableObject(objectID, centroid)

            # aksi takdirde, izlenebilir bir nesne vardır,
            # böylece yönü belirlemek için kullanabiliriz
            else:
                # * akım * sentroidinin y koordinatı ile
                # * önceki * sentroidlerin ortalaması arasındaki fark bize
                # nesnenin hangi yönde hareket ettiğini söyler
                # ('yukarı' için negatif ve 'aşağı' için pozitif)
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # nesnenin sayılıp sayılmadığını kontrol edin
                if not to.counted:
                    # yön negatifse (nesnenin yukarı hareket ettiğini gösterir)
                    # ve sentroid merkez çizginin üzerindeyse,
                    # nesneyi sayın
                    # ve nesneyi kaydedin
                    if direction < 0 and H // 2 > centroid[1] > (H // 2) - 10:
                        totalUp += 1
                        to.counted = True
                        cv2.imwrite("output\kisi%d.jpg" % count, frame)
                        count += 1

                    # yön pozitifse (nesnenin aşağı doğru hareket ettiğini gösterir)
                    # ve sentroid merkez çizginin altındaysa,
                    # nesneyi sayın
                    # ve nesneyi kaydedin
                    elif direction > 0 and H // 2 < centroid[1] < (H // 2) + 10:
                        totalDown += 1
                        to.counted = True
                        cv2.imwrite("output\kisi%d.jpg" % count, frame)
                        count += 1

            # izlenebilir nesneyi sözlüğümüzde sakla
            trackableObjects[objectID] = to

        # Çıktı çerçevesine hem nesnenin kimliğini
        # hem de nesnenin merkezini çizin
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # çerçeve üzerinde göstereceğimiz
    # bir dizi bilgi oluşturun
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]

    # bilgi tuples üzerinde döngü başlatın ve onları bizim çerçeve üzerine çizin
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # çerçeveyi diske yazmamız gerekip gerekmediğini kontrol edin
        if writer is not None:
            writer.write(frame)

        # çerçevenin çıktısını göster
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # eğer 'q' tuşuna basılırsa, döngüden çıkar
        if key == ord("q"):
            break

        # şimdiye kadar işlenen toplam kare sayısını artırın
        # ve ardından FPS sayacını güncelleyin
        totalFrames += 1
        fps.update()

# zamanlayıcıyı durdurma ve FPS bilgilerini görüntüleme
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# video yazarı işaretçisini bırakmamız gerekip gerekmediğini kontrol edin
if writer is not None:
	writer.release()

# video dosyası kullanmıyorsak, kamera video akışını durdurun
if not args.get("input", False):
	vs.stop()

# aksi takdirde video dosyası işaretçisini serbest bırakın
else:
	vs.release()

# açık pencereleri kapat
cv2.destroyAllWindows()
