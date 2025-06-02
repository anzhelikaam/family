import cv2

# Завантаження локального каскаду Хаара
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Перевірка, чи каскад завантажено правильно
if face_cascade.empty():
    print("❌ Помилка: не вдалося завантажити каскад.")
    exit()

# Завантаження зображення
image = cv2.imread("family.jpg")
if image is None:
    print("❌ Помилка: зображення family.jpg не знайдено.")
    exit()

# Конвертація в чорно-біле
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Виявлення облич
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Малювання рамок навколо облич
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Масштабування зображення до ширини 800 пікселів
width = 800
height = int(image.shape[0] * (width / image.shape[1]))
resized = cv2.resize(image, (width, height))

# Виведення результату
cv2.imshow("Detected Faces", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


