import re
from PIL import ImageDraw, ImageFont
from datasets import load_dataset

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parquet_file = os.path.join(ROOT_DIR, "parquets", "pisc", "pisc_train.parquet")

print(f"📂 Chargement de {parquet_file}...")

try:
    ds = load_dataset("parquet", data_files={'train': parquet_file}, split='train')
    print(f"✅ Dataset chargé avec succès ! Nombre d'exemples : {len(ds)}")

    index_to_test = 0  # Tu peux changer pour voir d'autres exemples
    example = ds[index_to_test]

    # --- QUESTION / RÉPONSE ---
    user_msg      = next(m for m in example['messages'] if m['role'] == 'user')
    assistant_msg = next(m for m in example['messages'] if m['role'] == 'assistant')

    print("\n--- 📝 QUESTION (User) ---")
    print(user_msg['content'])

    print("\n--- 💬 RÉPONSE (Assistant) ---")
    print(assistant_msg['content'])

    # --- IMAGE AVEC BOUNDING BOXES ---
    print("\n--- 🖼️ CONTENU IMAGE ---")
    img = example['images'][0]
    imgW, imgH = img.size
    print(f"Taille : {imgW}x{imgH}")

    # Parse les bounding boxes depuis le message user  (format: (x1,y1,x2,y2) en [0,1000])
    raw_boxes = re.findall(r'Person \d+ bounding box: \[(\d+), (\d+), (\d+), (\d+)\]', user_msg['content'])

    # Dénormaliser vers les coordonnées pixels réelles
    def denormalize(coords, w, h):
        x1, y1, x2, y2 = coords
        return (
            round(x1 / 1000 * w),
            round(y1 / 1000 * h),
            round(x2 / 1000 * w),
            round(y2 / 1000 * h),
        )

    boxes = [denormalize((int(x1), int(y1), int(x2), int(y2)), imgW, imgH)
             for x1, y1, x2, y2 in raw_boxes]

    # Dessiner les bounding boxes sur l'image
    colors = ["#FF4444", "#4488FF"]
    labels = ["Person 1", "Person 2"]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    for i, (box, color, label) in enumerate(zip(boxes, colors, labels)):
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), label, fill=color)

    print(f"Bounding boxes dessinées : {boxes}")
    print("Affichage de l'image annotée en cours...")
    annotated.show()

except Exception as e:
    print(f"❌ Erreur : {e}")
    raise
