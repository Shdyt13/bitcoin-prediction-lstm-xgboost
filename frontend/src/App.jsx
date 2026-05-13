import React, { useState, useMemo } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import zoomPlugin from 'chartjs-plugin-zoom';
import { 
  Upload, 
  BrainCircuit, 
  TrendingUp, 
  Target, 
  Bot, 
  BarChart3, 
  RotateCcw, 
  AlertTriangle, 
  LineChart, 
  Inbox, 
  History, 
  CalendarCheck,
  FileSpreadsheet // Icon baru untuk Excel
} from 'lucide-react';
// Import library pembuat Excel
import * as XLSX from 'xlsx';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler, zoomPlugin);

function App() {
  const [fileTrain, setFileTrain] = useState(null);
  const [fileTest, setFileTest] = useState(null);
  
  const [activeTab, setActiveTab] = useState('xgboost');

  const [resultXGB, setResultXGB] = useState(null);
  const [loadingXGB, setLoadingXGB] = useState(false);
  const [errorXGB, setErrorXGB] = useState('');

  const [resultLSTM, setResultLSTM] = useState(null);
  const [loadingLSTM, setLoadingLSTM] = useState(false);
  const [errorLSTM, setErrorLSTM] = useState('');

  const [resultHybrid, setResultHybrid] = useState(null);
  const [loadingHybrid, setLoadingHybrid] = useState(false);
  const [errorHybrid, setErrorHybrid] = useState('');

  const [evalHorizon, setEvalHorizon] = useState('1');

  const COLORS = {
    actual: '#2980b9',
    actualBg: 'rgba(41, 128, 185, 0.1)',
    xgb: '#e74c3c',
    xgbBg: 'rgba(231, 76, 60, 0.8)',
    lstm: '#27ae60',
    lstmBg: 'rgba(39, 174, 96, 0.8)',
    hybrid: '#8e44ad',
    hybridBg: 'rgba(142, 68, 173, 0.8)'
  };

  const handleFileTrainChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
      alert('Akses Ditolak: Mohon unggah dataset Training dalam format .csv!');
      e.target.value = '';
      setFileTrain(null);
      return;
    }

    if (selectedFile.size > 5 * 1024 * 1024) {
      alert('Ukuran file terlalu besar! Maksimal 5MB.');
      e.target.value = '';
      setFileTrain(null);
      return;
    }

    setFileTrain(selectedFile);
    resetErrors();
  };

  const handleFileTestChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
      alert('Akses Ditolak: Mohon unggah dataset Testing aktual dalam format .csv!');
      e.target.value = '';
      setFileTest(null);
      return;
    }

    if (selectedFile.size > 5 * 1024 * 1024) {
      alert('Ukuran file terlalu besar! Maksimal 5MB.');
      e.target.value = '';
      setFileTest(null);
      return;
    }

    setFileTest(selectedFile);
    resetErrors();
  };

  const resetErrors = () => {
    setErrorXGB('');
    setErrorLSTM('');
    setErrorHybrid('');
  };

  const handlePredict = async (algo) => {
    if (!fileTrain || !fileTest) {
      const msg = 'Silakan upload KEDUA file (Data Training & Data Testing) terlebih dahulu!';
      if (algo === 'xgboost') setErrorXGB(msg);
      else if (algo === 'lstm') setErrorLSTM(msg);
      else setErrorHybrid(msg);
      return;
    }

    const formData = new FormData();
    formData.append('dataset_train', fileTrain);
    formData.append('dataset_test', fileTest);
    formData.append('algorithm', algo);

    if (algo === 'xgboost') { setLoadingXGB(true); setErrorXGB(''); setResultXGB(null); }
    else if (algo === 'lstm') { setLoadingLSTM(true); setErrorLSTM(''); setResultLSTM(null); }
    else { setLoadingHybrid(true); setErrorHybrid(''); setResultHybrid(null); }

    try {
      const response = await axios.post('http://localhost:5000/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.status === 'success') {
        if (algo === 'xgboost') setResultXGB(response.data);
        else if (algo === 'lstm') setResultLSTM(response.data);
        else setResultHybrid(response.data);
      } else {
        if (algo === 'xgboost') setErrorXGB(response.data.message);
        else if (algo === 'lstm') setErrorLSTM(response.data.message);
        else setErrorHybrid(response.data.message);
      }
    } catch (err) {
      const errMsg = err.response?.data?.message || 'Gagal terhubung ke server backend AI. Pastikan Flask berjalan.';
      if (algo === 'xgboost') setErrorXGB(`Error Backend: ${errMsg}`);
      else if (algo === 'lstm') setErrorLSTM(`Error Backend: ${errMsg}`);
      else setErrorHybrid(`Error Backend: ${errMsg}`);
    } finally {
      if (algo === 'xgboost') setLoadingXGB(false);
      else if (algo === 'lstm') setLoadingLSTM(false);
      else setLoadingHybrid(false);
    }
  };

  const handleReset = () => {
    if(window.confirm('Apakah Anda yakin ingin menghapus semua hasil analisis dan file?')) {
      setFileTrain(null);
      setFileTest(null);
      
      document.getElementById('input-train').value = "";
      document.getElementById('input-test').value = "";

      setResultXGB(null);
      setResultLSTM(null);
      setResultHybrid(null);
      setActiveTab('xgboost');
    }
  };

  // --- FUNGSI BARU: EXPORT KE EXCEL ---
  const handleExportExcel = () => {
    const wb = XLSX.utils.book_new();

    // 1. Menyiapkan Data Metrik Evaluasi
    const metricsData = [];
    const models = [
      { name: 'XGBoost', data: resultXGB },
      { name: 'LSTM', data: resultLSTM },
      { name: 'Hybrid (LSTM+XGB)', data: resultHybrid }
    ];
    
    models.forEach(model => {
      if (model.data) {
        ['1', '3', '7'].forEach(h => {
          if (model.data.evaluasi[h]) {
            metricsData.push({
              'Model Algoritma': model.name,
              'Horizon Target (Hari)': parseInt(h),
              'RMSE': model.data.evaluasi[h].RMSE,
              'MAE': model.data.evaluasi[h].MAE,
              'MAPE (%)': model.data.evaluasi[h].MAPE
            });
          }
        });
      }
    });

    const wsMetrics = XLSX.utils.json_to_sheet(metricsData);
    XLSX.utils.book_append_sheet(wb, wsMetrics, "1. Metrik Evaluasi");

    // 2. Menyiapkan Data Prediksi Masa Depan (Blind Forecast)
    const availableResult = resultHybrid || resultLSTM || resultXGB;
    if (availableResult) {
      const forecastData = [];
      availableResult.grafik.future_dates.forEach((date, index) => {
        let row = { 'Tanggal': date };
        if (resultXGB) row['Prediksi XGBoost (Rp)'] = resultXGB.grafik.future_prices[index];
        if (resultLSTM) row['Prediksi LSTM (Rp)'] = resultLSTM.grafik.future_prices[index];
        if (resultHybrid) row['Prediksi Hybrid (Rp)'] = resultHybrid.grafik.future_prices[index];
        forecastData.push(row);
      });
      
      const wsForecast = XLSX.utils.json_to_sheet(forecastData);
      XLSX.utils.book_append_sheet(wb, wsForecast, "2. Prediksi Masa Depan");

      // 3. Menyiapkan Data Detail Evaluasi (Testing)
      const testingData = [];
      const actuals = availableResult.evaluasi['1'].evaluasi_detail.actual; // Pakai referensi horizon 1
      
      actuals.forEach((actualVal, index) => {
        let row = {
          'Data Uji Ke-': index + 1,
          'Harga Aktual (Rp)': actualVal
        };
        // Masukkan prediksi masing-masing model jika ada
        if (resultXGB) row['Prediksi XGBoost (Rp)'] = resultXGB.evaluasi['1'].evaluasi_detail.predicted[index];
        if (resultLSTM) row['Prediksi LSTM (Rp)'] = resultLSTM.evaluasi['1'].evaluasi_detail.predicted[index];
        if (resultHybrid) row['Prediksi Hybrid (Rp)'] = resultHybrid.evaluasi['1'].evaluasi_detail.predicted[index];
        
        // Hitung selisih error untuk model hybrid jika ada
        if (resultHybrid) {
           row['Selisih Error Hybrid (Rp)'] = Math.abs(actualVal - resultHybrid.evaluasi['1'].evaluasi_detail.predicted[index]);
        }

        testingData.push(row);
      });

      const wsTesting = XLSX.utils.json_to_sheet(testingData);
      XLSX.utils.book_append_sheet(wb, wsTesting, "3. Detail Pengujian (Horizon 1)");
    }

    // Eksekusi Download File
    XLSX.writeFile(wb, "Data_Analisis_Bitcoin.xlsx");
  };

  const formatRupiah = (angka) => {
    return new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(angka);
  };

  const chartOptionsRupiah = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    elements: { line: { tension: 0.2 } },
    plugins: {
      decimation: { enabled: true, algorithm: 'lttb' },
      zoom: {
        pan: { enabled: false, mode: 'x' }, 
        zoom: { wheel: { enabled: false }, pinch: { enabled: false }, mode: 'x' }
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            let label = context.dataset.label || '';
            if (label) { label += ': '; }
            if (context.parsed.y !== null) { label += formatRupiah(context.parsed.y); }
            return label;
          }
        }
      }
    }
  };

  const getChartData = (resultData, colorConfig) => {
    if (!resultData) return null;
    const { history_dates, history_prices, future_dates, future_prices } = resultData.grafik;
    const allDates = [...history_dates, ...future_dates];

    const lastHistPrice = history_prices[history_prices.length - 1];
    const dataHistorical = [...history_prices, null, null, null];
    const dataPredicted = Array(history_prices.length - 1).fill(null);
    dataPredicted.push(lastHistPrice, ...future_prices);

    return {
      labels: allDates,
      datasets: [
        { label: 'Harga Historis Aktual', data: dataHistorical, borderColor: COLORS.actual, backgroundColor: COLORS.actualBg, borderWidth: 2, fill: true },
        { label: 'Proyeksi Prediksi (Forecasting)', data: dataPredicted, borderColor: colorConfig, borderDash: [5, 5], borderWidth: 2, pointBackgroundColor: colorConfig, pointRadius: 5 }
      ]
    };
  };

  const getBarChartData = (metric) => {
    return {
      labels: ['1 Hari Kedepan', '3 Hari Kedepan', '7 Hari Kedepan'],
      datasets: [
        { label: `XGBoost`, data: resultXGB ? [resultXGB.evaluasi['1'][metric], resultXGB.evaluasi['3'][metric], resultXGB.evaluasi['7'][metric]] : [0, 0, 0], backgroundColor: COLORS.xgbBg, borderRadius: 4 },
        { label: `LSTM`, data: resultLSTM ? [resultLSTM.evaluasi['1'][metric], resultLSTM.evaluasi['3'][metric], resultLSTM.evaluasi['7'][metric]] : [0, 0, 0], backgroundColor: COLORS.lstmBg, borderRadius: 4 },
        { label: `Hybrid (LSTM+XGB)`, data: resultHybrid ? [resultHybrid.evaluasi['1'][metric], resultHybrid.evaluasi['3'][metric], resultHybrid.evaluasi['7'][metric]] : [0, 0, 0], backgroundColor: COLORS.hybridBg, borderRadius: 4 }
      ]
    };
  };

  const getLineComparasiData = (horizon) => {
    const hasDetailXGB = resultXGB && resultXGB.evaluasi[horizon]?.evaluasi_detail;
    const hasDetailLSTM = resultLSTM && resultLSTM.evaluasi[horizon]?.evaluasi_detail;
    const hasDetailHybrid = resultHybrid && resultHybrid.evaluasi[horizon]?.evaluasi_detail;

    if (!hasDetailXGB && !hasDetailLSTM && !hasDetailHybrid) return null;

    const actualData = hasDetailXGB ? resultXGB.evaluasi[horizon].evaluasi_detail.actual 
                     : hasDetailLSTM ? resultLSTM.evaluasi[horizon].evaluasi_detail.actual 
                     : resultHybrid.evaluasi[horizon].evaluasi_detail.actual;

    const labels = Array.from({ length: actualData.length }, (_, i) => `Data Uji Ke-${i + 1}`);
    const datasets = [{ label: 'Harga Aktual (Testing Set)', data: actualData, borderColor: COLORS.actual, backgroundColor: COLORS.actualBg, borderWidth: 2, fill: true, pointRadius: 0, pointHoverRadius: 5 }];

    if (hasDetailXGB) datasets.push({ label: 'Prediksi XGBoost', data: resultXGB.evaluasi[horizon].evaluasi_detail.predicted, borderColor: COLORS.xgb, borderDash: [4, 4], borderWidth: 2, pointRadius: 0, pointHoverRadius: 5 });
    if (hasDetailLSTM) datasets.push({ label: 'Prediksi LSTM', data: resultLSTM.evaluasi[horizon].evaluasi_detail.predicted, borderColor: COLORS.lstm, borderDash: [4, 4], borderWidth: 2, pointRadius: 0, pointHoverRadius: 5 });
    if (hasDetailHybrid) datasets.push({ label: 'Prediksi Hybrid (LSTM+XGB)', data: resultHybrid.evaluasi[horizon].evaluasi_detail.predicted, borderColor: COLORS.hybrid, borderDash: [4, 4], borderWidth: 2, pointRadius: 0, pointHoverRadius: 5 });

    return { labels, datasets };
  };

  const insightSummary = useMemo(() => {
    if (!resultXGB && !resultLSTM && !resultHybrid) return null;

    let mapeXGB = Infinity, mapeLSTM = Infinity, mapeHybrid = Infinity;

    if (resultXGB) mapeXGB = (resultXGB.evaluasi['1'].MAPE + resultXGB.evaluasi['3'].MAPE + resultXGB.evaluasi['7'].MAPE) / 3;
    if (resultLSTM) mapeLSTM = (resultLSTM.evaluasi['1'].MAPE + resultLSTM.evaluasi['3'].MAPE + resultLSTM.evaluasi['7'].MAPE) / 3;
    if (resultHybrid) mapeHybrid = (resultHybrid.evaluasi['1'].MAPE + resultHybrid.evaluasi['3'].MAPE + resultHybrid.evaluasi['7'].MAPE) / 3;

    const minMape = Math.min(mapeXGB, mapeLSTM, mapeHybrid);
    let winner = ''; let winnerColor = '';

    if (minMape === mapeHybrid) { winner = 'Hybrid (LSTM+XGB)'; winnerColor = COLORS.hybrid; }
    else if (minMape === mapeLSTM) { winner = 'LSTM'; winnerColor = COLORS.lstm; }
    else if (minMape === mapeXGB) { winner = 'XGBoost'; winnerColor = COLORS.xgb; }

    const formatNumber = (num) => num === Infinity ? '-' : new Intl.NumberFormat('id-ID', { maximumFractionDigits: 2 }).format(num);

    return (
      <div className="card-hover" style={{ display: 'flex', gap: '20px', marginBottom: '30px' }}>
        <div style={{ flex: 1, padding: '20px', backgroundColor: '#fff', borderLeft: `6px solid ${winnerColor}`, borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h4 style={{ margin: '0 0 5px 0', color: '#555', fontSize: '14px', textTransform: 'uppercase' }}>Juara Akurasi (Model Terbaik)</h4>
          <strong style={{ fontSize: '28px', color: winnerColor }}>{winner}</strong>
        </div>
        <div style={{ flex: 1, padding: '20px', backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #eee', boxShadow: '0 4px 6px rgba(0,0,0,0.02)' }}>
          <h4 style={{ margin: '0 0 10px 0', color: '#555', fontSize: '14px' }}>Rata-Rata Kesalahan (MAPE)</h4>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ color: COLORS.xgb, fontWeight: 'bold' }}>XGBoost:</span>
            <span style={{ fontWeight: 'bold' }}>{resultXGB ? `${formatNumber(mapeXGB)}%` : 'Belum dilatih'}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ color: COLORS.lstm, fontWeight: 'bold' }}>LSTM:</span>
            <span style={{ fontWeight: 'bold' }}>{resultLSTM ? `${formatNumber(mapeLSTM)}%` : 'Belum dilatih'}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: COLORS.hybrid, fontWeight: 'bold' }}>Hybrid:</span>
            <span style={{ fontWeight: 'bold' }}>{resultHybrid ? `${formatNumber(mapeHybrid)}%` : 'Belum dilatih'}</span>
          </div>
        </div>
      </div>
    );
  }, [resultXGB, resultLSTM, resultHybrid]);

  const renderTabContent = (algo, result, loading, error, colorBtn, title, description) => (
    <div style={{ padding: '30px', backgroundColor: 'white', border: '1px solid #ddd', borderTop: 'none', borderBottomLeftRadius: '12px', borderBottomRightRadius: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3 style={{ color: '#333', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          {algo === 'xgboost' && <Target size={24} color={colorBtn} />}
          {algo === 'lstm' && <BrainCircuit size={24} color={colorBtn} />}
          {algo === 'hybrid' && <Bot size={24} color={colorBtn} />}
          Mesin Prediksi {title}
        </h3>
        <span style={{ fontSize: '12px', padding: '5px 10px', backgroundColor: '#f0f2f5', borderRadius: '20px', color: '#666', fontWeight: 'bold' }}>{description}</span>
      </div>

      {error && (
        <div style={{ backgroundColor: '#f8d7da', color: '#842029', padding: '15px', borderRadius: '8px', marginBottom: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontWeight: 'bold', border: '1px solid #f5c2c7' }}>
          <AlertTriangle size={20} /> {error}
        </div>
      )}

      <button onClick={() => handlePredict(algo)} disabled={loading} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px', width: '100%', padding: '16px', backgroundColor: loading ? '#6c757d' : colorBtn, color: 'white', border: 'none', borderRadius: '8px', fontWeight: 'bold', fontSize: '16px', cursor: loading ? 'wait' : 'pointer', transition: 'all 0.3s', marginBottom: '30px', boxShadow: loading ? 'none' : `0 4px 10px ${colorBtn}40` }}>
        {loading ? (
          <>
            <span style={{ width: '20px', height: '20px', border: '3px solid rgba(255,255,255,0.3)', borderTop: '3px solid white', borderRadius: '50%', display: 'inline-block', animation: 'spin 1s linear infinite' }} />
            Sedang Memproses Model Analitik...
          </>
        ) : (
          <>
            <TrendingUp size={20} /> Jalankan Prediksi {title}
          </>
        )}
      </button>

      {result && (
        <div className="card-hover" style={{ border: '1px solid #eee', padding: '25px', borderRadius: '12px', backgroundColor: '#fafbfc' }}>
          <h4 style={{ color: '#444', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <LineChart size={20} color={colorBtn} /> Proyeksi Harga Masa Depan (Blind Forecasting)
          </h4>
          <div style={{ display: 'flex', gap: '15px', marginBottom: '30px' }}>
            {[1, 3, 7].map(hari => (
              <div key={hari} style={{ flex: 1, backgroundColor: '#fff', padding: '20px', borderRadius: '10px', textAlign: 'center', border: '1px solid #e9ecef', boxShadow: '0 2px 8px rgba(0,0,0,0.03)' }}>
                <div style={{ fontSize: '14px', color: '#888', fontWeight: '600', marginBottom: '10px', textTransform: 'uppercase' }}>Target {hari} Hari</div>
                <div style={{ fontSize: '20px', color: colorBtn, fontWeight: 'bold' }}>{formatRupiah(result.hasil_prediksi[hari])}</div>
              </div>
            ))}
          </div>
          <div style={{ height: '400px', width: '100%', padding: '15px', backgroundColor: '#fff', border: '1px solid #eee', borderRadius: '10px' }}>
            <Line data={getChartData(result, colorBtn)} options={chartOptionsRupiah} />
          </div>
          <p style={{ textAlign: 'center', fontSize: '12px', color: '#999', marginTop: '15px' }}>
            *Grafik menyambungkan data historis bulan terakhir dengan proyeksi masa depan. <b>Gunakan scroll/pinch untuk Zoom, dan drag untuk Geser (Pan).</b>
          </p>
        </div>
      )}
    </div>
  );
  // 532E68647974
  const renderEvaluasiTab = () => {
    const availableResult = resultXGB || resultLSTM || resultHybrid;

    return (
      <div style={{ padding: '30px', backgroundColor: 'white', border: '1px solid #ddd', borderTop: 'none', borderBottomLeftRadius: '12px', borderBottomRightRadius: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
          <h3 style={{ color: '#333', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={24} color={COLORS.actual} /> Evaluasi & Komparasi Model (Out-of-Sample Testing)
          </h3>
          {/* --- TOMBOL CETAK EXCEL --- */}
          {availableResult && ( 
            <button onClick={handleExportExcel} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '10px 18px', backgroundColor: '#107c41', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '14px', fontWeight: 'bold', transition: '0.2s', boxShadow: '0 2px 6px rgba(16,124,65,0.4)' }}>
              <FileSpreadsheet size={18} /> Export ke Excel
            </button> 
          )}
        </div>

        {!availableResult ? (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: '#666', backgroundColor: '#f8f9fa', borderRadius: '12px', border: '2px dashed #ddd' }}>
            <Inbox size={48} color="#adb5bd" style={{ margin: '0 auto 15px auto' }} />
            <h3 style={{ margin: '0 0 10px 0', color: '#444' }}>Belum Ada Data Evaluasi</h3>
            <p style={{ margin: 0, fontSize: '15px' }}>Silakan jalankan prediksi di salah satu tab algoritma terlebih dahulu untuk melihat analisis komprehensif.</p>
          </div>
        ) : (
          <div>
            {insightSummary}
            <h4 style={{ color: '#444', borderBottom: '2px solid #f0f0f0', paddingBottom: '10px', marginBottom: '20px' }}>1. Proporsi Dataset Penelitan (Explicit Split)</h4>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginBottom: '40px' }}>
              <div className="card-hover" style={{ flex: 1, backgroundColor: '#fff', padding: '20px', borderRadius: '10px', border: '1px solid #b8daff', textAlign: 'center', boxShadow: '0 2px 8px rgba(41,128,185,0.1)' }}>
                <span style={{ fontSize: '13px', color: COLORS.actual, display: 'block', fontWeight: 'bold', textTransform: 'uppercase', marginBottom: '5px' }}>Data Training (Pembelajaran)</span>
                <strong style={{ fontSize: '26px', color: COLORS.actual }}>{availableResult.info_data.data_training_mentah} Baris</strong>
                <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>({availableResult.info_data.sekuens_training} Sekuens Fit)</div>
              </div>
              <div className="card-hover" style={{ flex: 1, backgroundColor: '#fff', padding: '20px', borderRadius: '10px', border: '1px solid #f5c2c7', textAlign: 'center', boxShadow: '0 2px 8px rgba(231,76,60,0.1)' }}>
                <span style={{ fontSize: '13px', color: COLORS.xgb, display: 'block', fontWeight: 'bold', textTransform: 'uppercase', marginBottom: '5px' }}>Data Testing (Ujian)</span>
                <strong style={{ fontSize: '26px', color: COLORS.xgb }}>{availableResult.info_data.data_testing_mentah} Baris</strong>
                <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>({availableResult.info_data.sekuens_testing} Sekuens Uji)</div>
              </div>
            </div>

            <h4 style={{ color: '#444', borderBottom: '2px solid #f0f0f0', paddingBottom: '10px', marginBottom: '20px' }}>2. Visualisasi Prediksi vs Aktual (Data Testing)</h4>
            <div className="card-hover" style={{ marginBottom: '40px', padding: '25px', border: '1px solid #e9ecef', borderRadius: '12px', backgroundColor: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.03)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <span style={{ fontSize: '15px', color: '#555', fontWeight: '600' }}>Tinjauan Horizon Target:</span>
                <select value={evalHorizon} onChange={(e) => setEvalHorizon(e.target.value)} style={{ padding: '10px 20px', borderRadius: '8px', border: '1px solid #ccc', fontWeight: 'bold', outline: 'none', cursor: 'pointer', backgroundColor: '#f8f9fa' }}>
                  <option value="1">1 Hari Kedepan</option>
                  <option value="3">3 Hari Kedepan</option>
                  <option value="7">7 Hari Kedepan</option>
                </select>
              </div>
              {getLineComparasiData(evalHorizon) ? (
                <div style={{ height: '380px', width: '100%' }}><Line data={getLineComparasiData(evalHorizon)} options={chartOptionsRupiah} /></div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#888', fontStyle: 'italic', backgroundColor: '#fdfdfd', borderRadius: '8px', border: '1px dashed #ddd' }}>Menunggu ekstraksi data evaluasi detail.</div>
              )}
            </div>

            <h4 style={{ color: '#444', borderBottom: '2px solid #f0f0f0', paddingBottom: '10px', marginBottom: '20px' }}>3. Rekapitulasi Metrik Kesalahan (Error)</h4>
            <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
              <div className="card-hover" style={{ flex: '1 1 30%', height: '320px', backgroundColor: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e9ecef', boxShadow: '0 2px 8px rgba(0,0,0,0.03)' }}>
                <h4 style={{ textAlign: 'center', margin: '0 0 15px 0', color: '#555' }}>RMSE (Root Mean Squared Error)</h4>
                <Bar data={getBarChartData('RMSE')} options={chartOptionsRupiah} />
              </div>
              <div className="card-hover" style={{ flex: '1 1 30%', height: '320px', backgroundColor: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e9ecef', boxShadow: '0 2px 8px rgba(0,0,0,0.03)' }}>
                <h4 style={{ textAlign: 'center', margin: '0 0 15px 0', color: '#555' }}>MAE (Mean Absolute Error)</h4>
                <Bar data={getBarChartData('MAE')} options={chartOptionsRupiah} />
              </div>
              <div className="card-hover" style={{ flex: '1 1 30%', height: '320px', backgroundColor: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e9ecef', boxShadow: '0 2px 8px rgba(0,0,0,0.03)' }}>
                <h4 style={{ textAlign: 'center', margin: '0 0 15px 0', color: '#555' }}>MAPE (Persentase Kesalahan)</h4>
                <Bar data={getBarChartData('MAPE')} options={{ responsive: true, maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${new Intl.NumberFormat('id-ID', { maximumFractionDigits: 2 }).format(ctx.raw)}%` } } } }} />
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const isTabDisabled = loadingXGB || loadingLSTM || loadingHybrid;

  return (
    <div style={{ fontFamily: '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', backgroundColor: '#f0f2f5', minHeight: '100vh', padding: '40px 20px' }}>
      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .card-hover { transition: all 0.25s ease-in-out; }
        .card-hover:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.08) !important; border-color: #d1d5db !important; }
      `}</style>

      <div style={{ maxWidth: '1150px', margin: '0 auto' }}>
        
        {/* Header Section */}
        <div className="card-hover" style={{ backgroundColor: '#fff', padding: '35px', borderRadius: '16px', boxShadow: '0 4px 15px rgba(0,0,0,0.04)', marginBottom: '25px', position: 'relative' }}>
          {(resultXGB || resultLSTM || resultHybrid) && (
            <button onClick={handleReset} style={{ position: 'absolute', top: '25px', right: '25px', padding: '8px 15px', backgroundColor: '#f8d7da', color: '#842029', border: '1px solid #f5c2c7', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold', fontSize: '13px', transition: '0.2s', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <RotateCcw size={16} /> Reset Data
            </button>
          )}
          
          <h2 style={{ textAlign: 'center', color: '#1a1a1a', margin: '0 0 8px 0', fontSize: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
            <LineChart size={32} color="#0d6efd" /> Dashboard Analisis Time-Series Bitcoin
          </h2>
          <p style={{ textAlign: 'center', color: '#7f8c8d', fontSize: '16px', margin: '0 0 25px 0', fontWeight: '500' }}>Sapar Hidayat. S (2201020003)</p>
          
          <div style={{ display: 'flex', gap: '20px', transition: '0.3s', opacity: isTabDisabled ? 0.6 : 1, marginTop: '30px' }}>
            {/* Input Data Training */}
            <div style={{ flex: 1, backgroundColor: '#f8f9fa', padding: '20px', borderRadius: '10px', border: '2px dashed #6ea8fe' }}>
              <label style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', color: '#0d6efd', fontSize: '15px' }}>
                <History size={20} /> 1. Upload CSV Historis (Data Training)
              </label>
              <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#6c757d' }}>Data Bitcoin 1 Jan 2020 - 31 Des 2025.</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Upload size={18} color="#0d6efd" />
                <input id="input-train" type="file" accept=".csv" onChange={handleFileTrainChange} disabled={isTabDisabled} style={{ flex: 1, padding: '10px', backgroundColor: '#fff', border: '1px solid #ced4da', borderRadius: '6px', cursor: isTabDisabled ? 'not-allowed' : 'pointer', fontSize: '14px' }} />
              </div>
            </div>

            {/* Input Data Testing */}
            <div style={{ flex: 1, backgroundColor: '#f8f9fa', padding: '20px', borderRadius: '10px', border: '2px dashed #f5c2c7' }}>
              <label style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', color: '#dc3545', fontSize: '15px' }}>
                <CalendarCheck size={20} /> 2. Upload CSV Aktual (Data Testing)
              </label>
              <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#6c757d' }}>Data Bitcoin 1 Jan 2026 - Sekarang.</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Upload size={18} color="#dc3545" />
                <input id="input-test" type="file" accept=".csv" onChange={handleFileTestChange} disabled={isTabDisabled} style={{ flex: 1, padding: '10px', backgroundColor: '#fff', border: '1px solid #ced4da', borderRadius: '6px', cursor: isTabDisabled ? 'not-allowed' : 'pointer', fontSize: '14px' }} />
              </div>
            </div>
          </div>
        </div>

        {/* Navigasi Tab */}
        <div style={{ display: 'flex', opacity: isTabDisabled ? 0.6 : 1, pointerEvents: isTabDisabled ? 'none' : 'auto', gap: '5px', marginBottom: '-1px', zIndex: 1, position: 'relative' }}>
          <button onClick={() => setActiveTab('xgboost')} style={{ flex: 1, padding: '16px', background: activeTab === 'xgboost' ? 'white' : '#e9ecef', color: activeTab === 'xgboost' ? COLORS.xgb : '#6c757d', border: '1px solid #ddd', borderBottom: activeTab === 'xgboost' ? 'none' : '1px solid #ddd', borderTopLeftRadius: '12px', borderTopRightRadius: '12px', cursor: 'pointer', fontWeight: 'bold', fontSize: '15px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
            <Target size={20} /> Prediksi XGBoost
          </button>
          <button onClick={() => setActiveTab('lstm')} style={{ flex: 1, padding: '16px', background: activeTab === 'lstm' ? 'white' : '#e9ecef', color: activeTab === 'lstm' ? COLORS.lstm : '#6c757d', border: '1px solid #ddd', borderBottom: activeTab === 'lstm' ? 'none' : '1px solid #ddd', borderTopLeftRadius: '12px', borderTopRightRadius: '12px', cursor: 'pointer', fontWeight: 'bold', fontSize: '15px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
            <BrainCircuit size={20} /> Prediksi LSTM
          </button>
          <button onClick={() => setActiveTab('hybrid')} style={{ flex: 1, padding: '16px', background: activeTab === 'hybrid' ? 'white' : '#e9ecef', color: activeTab === 'hybrid' ? COLORS.hybrid : '#6c757d', border: '1px solid #ddd', borderBottom: activeTab === 'hybrid' ? 'none' : '1px solid #ddd', borderTopLeftRadius: '12px', borderTopRightRadius: '12px', cursor: 'pointer', fontWeight: 'bold', fontSize: '15px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
            <Bot size={20} /> Prediksi Hybrid
          </button>
          <button onClick={() => setActiveTab('evaluasi')} style={{ flex: 1, padding: '16px', background: activeTab === 'evaluasi' ? 'white' : '#e9ecef', color: activeTab === 'evaluasi' ? COLORS.actual : '#6c757d', border: '1px solid #ddd', borderBottom: activeTab === 'evaluasi' ? 'none' : '1px solid #ddd', borderTopLeftRadius: '12px', borderTopRightRadius: '12px', cursor: 'pointer', fontWeight: 'bold', fontSize: '15px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={20} /> Komparasi Model
          </button>
        </div>

        {/* Konten Tab */}
        <div style={{ zIndex: 0, position: 'relative' }}>
          {activeTab === 'xgboost' && renderTabContent('xgboost', resultXGB, loadingXGB, errorXGB, COLORS.xgb, 'XGBoost', 'Boosting Tree (Tuning)')}
          {activeTab === 'lstm' && renderTabContent('lstm', resultLSTM, loadingLSTM, errorLSTM, COLORS.lstm, 'LSTM', 'Neural Network (50 Epochs)')}
          {activeTab === 'hybrid' && renderTabContent('hybrid', resultHybrid, loadingHybrid, errorHybrid, COLORS.hybrid, 'Hybrid (LSTM ➔ XGB)', 'Stacking Sequential')}
          {activeTab === 'evaluasi' && renderEvaluasiTab()}
        </div>

      </div>
    </div>
  );
}

export default App;