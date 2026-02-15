import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import os
from ultralytics import YOLO
import csv
from datetime import datetime
import glob


# KONFIGURACJA UI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

MODEL_PATH = 'runs_medium/yolov10m_960px_300ep_MediumAugs_sgd/weights/best.pt'
PREVIEW_SIZE = (550, 550)


class PotholeDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- KONFIGURACJA OKNA ---
        self.title("System detekcji i klasyfikacji uszkodzeń drogowych")
        self.geometry("1200x850")  # Nieco wyższe okno

        # Maksymalizacja po uruchomieniu
        self.after(0, self.maximize_window)

        # Zmienne logiczne
        self.current_image_path = None
        self.model = None

        # Zmienne do eksportu pojedynczego
        self.current_result_bgr = None
        self.current_stats = None

        # --- UKŁAD GŁÓWNY (GRID) ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- 1. NAGŁÓWEK ---
        self.lbl_title = ctk.CTkLabel(self, text="Panel Analizy Nawierzchni Drogowej",
                                      font=("Roboto Medium", 36))
        self.lbl_title.grid(row=0, column=0, columnspan=2, pady=(20, 20), sticky="ew")

        # --- 2. LEWA KOLUMNA (DANE) ---
        self.frame_left = ctk.CTkFrame(self, corner_radius=15)
        self.frame_left.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.frame_left.grid_columnconfigure(0, weight=1)

        # Header Lewy
        self.lbl_input_header = ctk.CTkLabel(self.frame_left, text="Dane wejściowe",
                                             font=("Roboto", 24, "bold"))
        self.lbl_input_header.grid(row=0, column=0, pady=15)

        # Przycisk Load Single
        self.btn_load = ctk.CTkButton(self.frame_left, text="Załaduj pojedyncze zdjęcie",
                                      command=self.load_image,
                                      height=40, font=("Roboto", 24),
                                      fg_color="#3a7ebf", hover_color="#32669a")
        self.btn_load.grid(row=1, column=0, pady=(0, 10), padx=20, sticky="ew")

        # --- NOWOŚĆ: Przycisk Batch ---
        self.btn_batch = ctk.CTkButton(self.frame_left, text="📂 Tryb Wsadowy (Folder)",
                                       command=self.run_batch_analysis,
                                       height=40, font=("Roboto", 24),
                                       fg_color="#D35400", hover_color="#A04000")
        self.btn_batch.grid(row=2, column=0, pady=(0, 20), padx=20, sticky="ew")

        # Placeholder / Obraz Lewy
        self.lbl_preview_img = ctk.CTkLabel(self.frame_left, text="",
                                            fg_color="transparent")
        self.lbl_preview_img.grid(row=3, column=0, padx=10, pady=10)

        self.lbl_placeholder_left = ctk.CTkLabel(self.frame_left, text="[Brak zdjęcia]",
                                                 text_color="gray50",
                                                 font=("Roboto", 16))  # <--- Dodano ten parametr
        self.lbl_placeholder_left.grid(row=3, column=0)

        # --- 3. PRAWA KOLUMNA (WYNIKI) ---
        self.frame_right = ctk.CTkFrame(self, corner_radius=15)
        self.frame_right.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_rowconfigure(5, weight=1)

        # Header Prawy
        self.lbl_output_header = ctk.CTkLabel(self.frame_right, text="Wynik detekcji",
                                              font=("Roboto", 24, "bold"))
        self.lbl_output_header.grid(row=0, column=0, pady=15)

        # Przycisk Detect (Single)
        self.btn_detect = ctk.CTkButton(self.frame_right, text="Wykryj (Pojedyncze)",
                                        command=self.detect_potholes,
                                        height=40, font=("Roboto", 24),
                                        fg_color="#2CC985", hover_color="#25A56D",
                                        text_color="white")
        self.btn_detect.grid(row=1, column=0, pady=(0, 20), padx=20, sticky="ew")

        # --- NOWOŚĆ: Pasek Postępu dla Batch ---
        self.progress_bar = ctk.CTkProgressBar(self.frame_right, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="ew")
        self.progress_bar.grid_remove()  # Ukrywamy na start

        self.lbl_progress_text = ctk.CTkLabel(self.frame_right, text="", font=("Roboto", 12))
        self.lbl_progress_text.grid(row=3, column=0, pady=(0, 10))
        self.lbl_progress_text.grid_remove()

        # Placeholder / Obraz Prawy
        self.lbl_result_img = ctk.CTkLabel(self.frame_right, text="",
                                           fg_color="transparent")
        self.lbl_result_img.grid(row=4, column=0, padx=10, pady=10)

        self.lbl_placeholder_right = ctk.CTkLabel(self.frame_right, text="[Wynik pojawi się tutaj]",
                                                  text_color="gray50",
                                                  font=("Roboto", 16))  # <--- Dodano rozmiar 16
        self.lbl_placeholder_right.grid(row=4, column=0)

        # Licznik
        self.lbl_count = ctk.CTkLabel(self.frame_right, text="Wykryto uszkodzeń: -",
                                      font=("Roboto", 18, "bold"), text_color="#2CC985")
        self.lbl_count.grid(row=5, column=0, pady=(10, 5))

        # Panel szczegółów
        self.txt_details = ctk.CTkTextbox(self.frame_right, height=120, corner_radius=10,
                                          fg_color="#2b2b2b", text_color="#e0e0e0",
                                          font=("Consolas", 14))
        self.txt_details.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="nsew")
        self.txt_details.insert("0.0", "Oczekiwanie na analizę...")
        self.txt_details.configure(state="disabled")

        # --- SEKCJA EKSPORTU ---
        self.frame_export = ctk.CTkFrame(self.frame_right, fg_color="transparent")
        self.frame_export.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.frame_export.grid_columnconfigure(0, weight=1)
        self.frame_export.grid_columnconfigure(1, weight=1)

        self.btn_save_img = ctk.CTkButton(self.frame_export, text="💾 Zapisz Obraz",
                                          command=self.save_result_image,
                                          height=35, fg_color="#555555", hover_color="#444444")
        self.btn_save_img.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.btn_save_csv = ctk.CTkButton(self.frame_export, text="📄 Eksport CSV",
                                          command=self.save_report_csv,
                                          height=35, fg_color="#555555", hover_color="#444444")
        self.btn_save_csv.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # --- 4. PASEK STATUSU ---
        self.lbl_status = ctk.CTkLabel(self, text="Inicjalizacja...", anchor="w", padx=10)
        self.lbl_status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        self.after(100, self.load_model)

    def maximize_window(self):
        try:
            self.state("zoomed")
        except:
            try:
                self.attributes("-zoomed", True)
            except:
                screen_width = self.winfo_screenwidth()
                screen_height = self.winfo_screenheight()
                self.geometry(f"{screen_width}x{screen_height}+0+0")

    def load_model(self):
        self.lbl_status.configure(text="Ładowanie modelu w tle...")
        self.update()
        try:
            if os.path.exists(MODEL_PATH):
                self.model = YOLO(MODEL_PATH)
                self.lbl_status.configure(text=f"Gotowy. Model: {os.path.basename(MODEL_PATH)}")
            else:
                self.model = YOLO("yolov10m.pt")
                self.lbl_status.configure(text="OSTRZEŻENIE: Użyto domyślnego modelu YOLOv10m")
                messagebox.showwarning("Brak modelu", "Nie znaleziono modelu. Załadowano YOLOv10m.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się załadować modelu:\n{e}")


    # LOGIKA POJEDYNCZEGO ZDJĘCIA
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.current_image_path = file_path
            try:
                pil_img = Image.open(file_path)
                my_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=PREVIEW_SIZE)
                self.lbl_preview_img.configure(image=my_image)
                self.lbl_placeholder_left.grid_forget()
                self.lbl_status.configure(text=f"Załadowano plik: {os.path.basename(file_path)}")

                # Reset
                self.lbl_result_img.configure(image=None)
                self.lbl_placeholder_right.grid(row=4, column=0)
                self.lbl_count.configure(text="Wykryto uszkodzeń: -", text_color="gray")
                self.current_result_bgr = None
                self.current_stats = None

                # Reset progress bar visibility if it was showing
                self.progress_bar.grid_remove()
                self.lbl_progress_text.grid_remove()

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można otworzyć pliku: {e}")

    def detect_potholes(self):
        if not self.current_image_path:
            messagebox.showinfo("Info", "Proszę najpierw załadować zdjęcie.")
            return
        if not self.model: return

        self.lbl_status.configure(text="Przetwarzanie obrazu...")
        self.update()  # Odśwież UI

        try:
            results = self.model.predict(source=self.current_image_path, conf=0.25)

            # Rysowanie
            res_plotted = results[0].plot()
            self.current_result_bgr = res_plotted

            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)
            result_image = ctk.CTkImage(light_image=res_pil, dark_image=res_pil, size=PREVIEW_SIZE)

            self.lbl_result_img.configure(image=result_image)
            self.lbl_placeholder_right.grid_forget()

            # Statystyki
            boxes = results[0].boxes
            total_count = len(boxes)
            class_names = results[0].names
            detected_classes = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                detected_classes.append(class_names[cls_id])

            counts = {name: detected_classes.count(name) for name in set(detected_classes)}

            self.current_stats = {
                "filename": os.path.basename(self.current_image_path),
                "total": total_count,
                "breakdown": counts,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.update_results_ui(total_count, counts)
            self.lbl_status.configure(text=f"Zakończono. Obiektów: {total_count}")

        except Exception as e:
            messagebox.showerror("Błąd detekcji", f"Błąd: {e}")

    def update_results_ui(self, total_count, counts):
        color = "#2CC985" if total_count > 0 else "gray"
        self.lbl_count.configure(text=f"Wykryto uszkodzeń: {total_count}", text_color=color)

        report_text = f"RAPORT DETEKCJI:\n----------------\n"
        if total_count == 0:
            report_text += "Brak wykrytych uszkodzeń."
        else:
            for name, count in counts.items():
                report_text += f"• {name}: {count}\n"

        self.txt_details.configure(state="normal")
        self.txt_details.delete("0.0", "end")
        self.txt_details.insert("0.0", report_text)
        self.txt_details.configure(state="disabled")


    # LOGIKA TRYBU WSADOWEGO (BATCH)
    def run_batch_analysis(self):
        if not self.model: return

        # 1. Wybierz folder wejściowy
        input_folder = filedialog.askdirectory(title="Wybierz folder ze zdjęciami")
        if not input_folder: return

        # 2. Znajdź zdjęcia
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))  # obsługa JPG/PNG

        # --- POPRAWKA: USUWANIE DUPLIKATÓW ---
        image_files = list(set(image_files))
        # -------------------------------------

        total_files = len(image_files)
        if total_files == 0:
            messagebox.showinfo("Brak zdjęć", "W wybranym folderze nie znaleziono zdjęć.")
            return

        # 3. Przygotuj folder wyjściowy i plik CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(input_folder, f"WYNIKI_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)

        csv_path = os.path.join(output_folder, "RAPORT_ZBIORCZY.csv")

        # Pokaż Progress Bar
        self.progress_bar.grid()
        self.lbl_progress_text.grid()
        self.progress_bar.set(0)
        self.btn_batch.configure(state="disabled", text="Przetwarzanie...")

        self.txt_details.configure(state="normal")
        self.txt_details.delete("0.0", "end")
        self.txt_details.insert("0.0", f"Rozpoczynam analizę {total_files} plików...\n")
        self.txt_details.configure(state="disabled")

        # 4. Pętla przetwarzania
        processed_count = 0
        csv_data = []

        try:
            # Otwórz CSV od razu
            with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['Plik', 'Liczba_Uszkodzen', 'Detale', 'Sciezka_Wynikowa']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()

                for img_path in image_files:
                    filename = os.path.basename(img_path)

                    # Predykcja
                    results = self.model.predict(source=img_path, conf=0.25, verbose=False)

                    # Zapisz obraz z ramkami
                    res_plotted = results[0].plot()
                    output_img_path = os.path.join(output_folder, f"RES_{filename}")
                    cv2.imwrite(output_img_path, res_plotted)

                    # Zbieranie statystyk
                    boxes = results[0].boxes
                    count = len(boxes)
                    class_names = results[0].names
                    detected = [class_names[int(box.cls[0])] for box in boxes]
                    counts_map = {name: detected.count(name) for name in set(detected)}
                    details_str = ", ".join([f"{k}:{v}" for k, v in counts_map.items()]) if counts_map else "Brak"

                    # Zapis do CSV
                    writer.writerow({
                        'Plik': filename,
                        'Liczba_Uszkodzen': count,
                        'Detale': details_str,
                        'Sciezka_Wynikowa': output_img_path
                    })

                    processed_count += 1

                    # Aktualizacja UI
                    progress = processed_count / total_files
                    self.progress_bar.set(progress)
                    self.lbl_progress_text.configure(
                        text=f"Przetworzono: {processed_count}/{total_files} ({int(progress * 100)}%)")

                    # Wymuszamy odświeżenie okna, żeby nie "zamarzło"
                    self.update()

            messagebox.showinfo("Sukces", f"Zakończono przetwarzanie wsadowe.\nWyniki zapisane w:\n{output_folder}")
            self.lbl_status.configure(text=f"Gotowe. Przetworzono {total_files} zdjęć.", font=("Roboto", 16))

            # Otwórz folder wyników (tylko Windows)
            os.startfile(output_folder)

        except Exception as e:
            messagebox.showerror("Błąd krytyczny", f"Przerwano przetwarzanie:\n{e}")

        finally:
            # Sprzątanie UI
            self.btn_batch.configure(state="normal", text="📂 Tryb Wsadowy (Folder)")
            self.progress_bar.grid_remove()
            self.lbl_progress_text.grid_remove()


    # ZAPIS MANUALNY (POJEDYNCZY)
    def save_result_image(self):
        if self.current_result_bgr is None:
            messagebox.showinfo("Info", "Brak wyniku do zapisania.")
            return
        original_name = os.path.basename(self.current_image_path)
        default_name = f"RESULT_{original_name}"
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile=default_name)
        if file_path:
            cv2.imwrite(file_path, self.current_result_bgr)
            messagebox.showinfo("Zapisano", f"Plik: {file_path}")

    def save_report_csv(self):
        if self.current_stats is None:
            messagebox.showinfo("Info", "Brak danych do zapisu.")
            return
        default_name = f"REPORT_{os.path.splitext(self.current_stats['filename'])[0]}.csv"
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default_name)
        if file_path:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Plik", "Ilosc", "Detale"])
                writer.writerow(
                    [self.current_stats['filename'], self.current_stats['total'], self.current_stats['breakdown']])
            messagebox.showinfo("Zapisano", "Raport CSV gotowy.")


if __name__ == "__main__":
    app = PotholeDetectorApp()
    app.mainloop()