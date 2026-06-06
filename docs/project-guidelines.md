# Alur Proyek End-to-End: Integrasi RST + CBR untuk Prediksi Tingkat Kecanduan Alkohol

Dokumen ini menjelaskan aliran data dan proses komputasi sistem secara menyeluruh, dibagi menjadi 2 fase utama: offline (pembentukan pengetahuan) dan online (prediksi real-time).

## Data Flow End-to-End

```mermaid
flowchart TD
    subgraph Fase_Offline [FASE OFFLINE: Proses Training & Pengetahuan]
        A[Raw Data Student-Mat] --> B[Perhitungan Skor Bobot:<br>0.6 * DALC + 0.4 * WALC]
        B --> C[Penentuan Label Target Keputusan:<br>Normal / Waspada / Bahaya]

        %% Jalur Preprocessing
        A --> D[Preprocessing RST:<br>Diskretisasi Fitur ke Data Kategori]
        C --> D
        A --> E[Preprocessing CBR:<br>Normalisasi Fitur ke Skala Min-Max 0-1]
        C --> E

        %% Penyimpanan File Hasil Preprocessing
        D --> F[(dataset_rst.csv:<br>Basis Kasus Kategorikal + Label Keputusan)]
        E --> O_cbr[(dataset_cbr.csv:<br>Basis Kasus Numerik Min-Max + Label Solusi)]

        %% Jalur Inti Teori Rough Set (RST)
        F --> G[Pengelompokkan Data:<br>Siswa dengan Fitur Kondisi Sama]
        G --> H[Penyaringan Wilayah Pasti:<br>Cari Kelompok Siswa yang Solusinya Konsisten]

        %% Alur 1: Pembentukan Bobot Dinamis
        H --> I[Evaluasi Ketergantungan Atribut:<br>Hitung Dampak Jika Atribut Dihapus]
        I --> J[Kalkulasi Signifikansi Atribut:<br>Ubah Nilai Penurunan Menjadi Bobot]
        J --> K_weights[(cbr_weights.json:<br>File Bobot Dinamis Variatif CBR)]

        %% Alur 2: Pembentukan Basis Aturan (Rules) Hasil Reduksi Murni
        J --> L[Penyaringan Atribut Valid:<br>Pilih Atribut dengan Signifikansi Asli > 0.0]
        L --> M[Ekstraksi Aturan Keputusan:<br>Ambil Kombinasi Unik dari Atribut Terpilih]
        M --> N[Kalkulasi Probabilitas Aturan:<br>Hitung Skor Confidence Tiap Pola Aturan]
        N --> N_rules[(rst_rules.json:<br>File Aturan IF-THEN + Confidence)]

    end

    subgraph Fase_Online [FASE ONLINE: REST API Real-Time]
        P[Input Data Angka User Baru] --> Q[Proses Preprocessing di Memori]

        %% Mapping Input Baru ke Layer Masing-Masing
        Q --> R[Diskretisasi Otomatis:<br>Ubah Angka Menjadi Kategori Kata]
        Q --> S[Normalisasi Otomatis:<br>Ubah Angka Menjadi Skala 0-1]

        %% Alur Proses Layer RST
        R --> T[Pencocokan Aturan:<br>Cari Pola Kondisi di rst_rules.json]
        N_rules --> T
        T --> U[Ambil Nilai Confidence:<br>Menjadi Rough Set Score SR per Kelas]

        %% Alur Proses Layer CBR
        S --> V[Pencarian Kedekatan Kasus:<br>Hitung Jarak Kasus Baru ke dataset_cbr.csv]
        O_cbr --> V
        K_weights --> V[Gunakan Rumus Jarak Berbobot:<br>Gunakan Nilai dari cbr_weights.json]
        V --> W[Ambil Nilai Kemiripan Tertinggi:<br>Menjadi CBR Score SC per Kelas]

        %% Alur Integrasi dan Output Akhir
        U --> X[Integrasi Solusi:<br>Hitung Skor Akhir Kombinasi<br>Final Score = 0.6 * SR + 0.4 * SC]
        W --> X

        X --> Y[Pemilihan Keputusan:<br>Pilih Label dengan Skor Akhir Tertinggi]
        Y --> Z[Respon JSON API:<br>Hasil Prediksi Akhir & Skor Persentase]
    end
```
