import torch 
from ultralytics import YOLO 
import sys 
 
def train_pothole_detector_medium_config(): 
 
    # --- 1. Sprawdzenie GPU --- 
    print("--- Sprawdzanie dostępności GPU (Tryb Medium) ---") 
    if torch.cuda.is_available(): 
        device = torch.device("cuda:0") 
        gpu_name = torch.cuda.get_device_name(0) 
        print(f"Znaleziono: {gpu_name}") 
    else:
        print("OSTRZEŻENIE: Zalecane jest potężne GPU (CUDA). Użycie CPU zajmie BARDZO długo.") 
        device = torch.device("cpu")
 
    print("-----------------------------------") 
 
    # --- 2. Zmiana Modelu na MEDIUM --- 
    model_name = 'yolov10m.pt'
    print(f"Ładowanie średniego modelu: {model_name}") 
 
    try:
        model = YOLO(model_name) 
        model.to(device) 
    except Exception as e:
        sys.exit(f"BŁĄD: Nie można załadować modelu {model_name}. Upewnij się, że masz połączenie z internetem lub model jest pobrany. {e}")
 
    # --- 3. Parametry pod konfigurację MEDIUM --- 
    data_yaml_path = 'data.yaml' 

    image_size = 960
    batch_size = 20
    num_epochs = 300 
 
    # Parametry Optymalizatora
    optimizer_name = 'sgd' 
    initial_learning_rate = 0.001
 
    project_folder = 'runs_medium' 
    run_name = 'yolov10m_960px_300ep_MediumAugs_sgd'
 
    print("\n--- Rozpoczynanie treningu ---") 
    print(f"Model: {model_name} (Medium)") 
    print(f"Rozdzielczość: {image_size}x{image_size}") 
    print(f"Batch size: {batch_size}") 
    print(f"Epoki: {num_epochs}") 
    print(f"Optymalizator: {optimizer_name}") 
    print(f"Learning Rate (początkowy): {initial_learning_rate}") 
    print("---------------------------------") 
 
    try: 
        model.train( 
            data=data_yaml_path, 
            epochs=num_epochs, 
            batch=batch_size, 
            imgsz=image_size, 
            project=project_folder, 
            name=run_name, 
            device=0 if device.type == 'cuda' else None,
 
            # Optymalizacja (Adam) 
            optimizer=optimizer_name, 
            lr0=initial_learning_rate, 
 
            # Optymalizacja wydajności danych
            workers=4, 
            cache=True, 
 
            # --- STRATEGIA KOŃCOWA --- 
            close_mosaic=50,
 
            # --- GEOMETRIA ---
            mosaic=1.0,  
            degrees=10.0, 
            translate=0.1, 
            scale=0.5,
            shear=0.0,
            perspective=0.0,
 
            # --- MIESZANIE I GENEROWANIE PRÓBEK ---
            mixup=0.1,
            copy_paste=0.0,
            erasing=0.0,
 
            # --- KOLORY --- 
            hsv_h=0.015, 
            hsv_s=0.7, 
            hsv_v=0.4, 
            fliplr=0.5, 
        ) 
 
        print("\n--- Trening MEDIUM zakończony ---") 
        print(f"Najlepszy model: {project_folder}/{run_name}/weights/best.pt") 
 
    except torch.cuda.OutOfMemoryError: 
        print("\nBŁĄD PAMIĘCI: Zmniejsz 'batch_size' do 16 lub 8 i spróbuj ponownie.") 
    except Exception as e: 
        print(f"\nBłąd: {e}") 
 
 
if __name__ == '__main__':
    train_pothole_detector_medium_config()