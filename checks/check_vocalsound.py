import io
import os

import soundfile as sf
from datasets import load_dataset

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parquet_file = os.path.join(ROOT_DIR, "parquets", "vocalsound", "vocalsound_train.parquet")

print(f"📂 Chargement de {parquet_file}...")

try:
    ds = load_dataset("parquet", data_files={'train': parquet_file}, split='train')
    print(f"✅ Dataset chargé avec succès ! Total : {len(ds)} exemples.\n")

    # --- RECHERCHE D'UN EXEMPLE AVEC AUDIO ---
    index_to_test = -1
    for i in range(len(ds)):
        if ds[i]['audios'] and len(ds[i]['audios']) > 0:
            index_to_test = i
            break

    if index_to_test == -1:
        print("❌ Aucun audio trouvé dans le dataset !")
        exit()

    print(f"🎯 Exemple avec audio trouvé à l'index : {index_to_test}\n")
    example = ds[index_to_test]

    # ================= CONTENU TEXTUEL =================
    user_msg      = next(m for m in example['messages'] if m['role'] == 'user')
    assistant_msg = next(m for m in example['messages'] if m['role'] == 'assistant')

    print("--- 📝 QUESTION (User) ---")
    print(user_msg['content'])

    print("\n--- 💬 RÉPONSE (Assistant) ---")
    print(assistant_msg['content'])

    # ================= CONTENU AUDIO =================
    print("\n--- 🎵 CONTENU AUDIO ---")
    audio_bytes = example['audios'][0]
    print(f"✅ Audio trouvé ! ({len(audio_bytes)} bytes)")

    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        duration = len(data) / samplerate
        print(f"   Validité    : OK (Décodé avec SoundFile)")
        print(f"   Durée       : {duration:.2f} secondes")
        print(f"   Fréquence   : {samplerate} Hz")
        print(f"   Canaux      : {1 if data.ndim == 1 else data.shape[1]}")


    except Exception as e:
        print(f"❌ Erreur : Impossible de décoder l'audio : {e}")

except Exception as e:
    print(f"❌ Erreur critique : {e}")
