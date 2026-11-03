// ---------------------------------------------------------
// Copyright (c) Shdyt13
// ---------------------------------------------------------
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
  Sun,
  Moon,
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
  FileSpreadsheet
} from 'lucide-react';
import * as XLSX from 'xlsx';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler, zoomPlugin);

const barValueLabelsPlugin = {
  id: 'barValueLabels',
  afterDatasetsDraw(chart, _args, pluginOptions) {
    const { ctx } = chart;
    const formatter = pluginOptions?.formatter || ((value) => value);

    ctx.save();
    ctx.font = '600 9px Inter, Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = pluginOptions?.color || '#18212f';

    chart.data.datasets.forEach((_dataset, datasetIndex) => {
      const meta = chart.getDatasetMeta(datasetIndex);
      meta.data.forEach((bar, index) => {
        const value = chart.data.datasets[datasetIndex].data[index];
        if (value === null || value === undefined || value === 0 || !bar || !chart.isDatasetVisible(datasetIndex)) return;

        const position = bar.tooltipPosition();
        ctx.fillText(formatter(value), position.x, position.y - 6);
      });
    });

    ctx.restore();
  }
};

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  const theme = {
    bg: isDarkMode ? '#090d16' : '#f3f6fc',
    card: isDarkMode ? 'rgba(17, 24, 39, 0.72)' : 'rgba(255, 255, 255, 0.85)',
    innerCard: isDarkMode ? 'rgba(15, 23, 42, 0.55)' : 'rgba(255, 255, 255, 0.92)',
    textMain: isDarkMode ? '#f8fafc' : '#0f2851',
    textMuted: isDarkMode ? '#94a3b8' : '#526583',
    border: isDarkMode ? 'rgba(255, 255, 255, 0.08)' : '#e2e8f0',
    tabInactiveBg: isDarkMode ? 'rgba(17, 24, 39, 0.55)' : 'rgba(255, 255, 255, 0.82)',
    tabActiveBg: isDarkMode ? 'rgba(30, 41, 59, 0.85)' : '#ffffff',
    inputBg: isDarkMode ? 'rgba(9, 13, 22, 0.6)' : '#ffffff',
    inputBorder: isDarkMode ? 'rgba(255, 255, 255, 0.12)' : '#dbe5f5',
    uploadBoxBg: isDarkMode ? 'rgba(15, 23, 42, 0.45)' : 'rgba(248, 250, 252, 0.75)',
    uploadBorder: isDarkMode ? 'rgba(148, 163, 184, 0.22)' : '#cad8f2',
    badgeBg: isDarkMode ? 'rgba(56, 189, 248, 0.1)' : '#e6f0ff',
    badgeBorder: isDarkMode ? '1px solid rgba(56, 189, 248, 0.22)' : '1px solid #cfe2ff',
    badgeText: isDarkMode ? '#7dd3fc' : '#1d5bbf',
    toggleBg: isDarkMode ? 'rgba(30, 41, 59, 0.65)' : '#eef5ff',
    toggleBorder: isDarkMode ? 'rgba(255, 255, 255, 0.12)' : '#c2d8f8',
    toggleText: isDarkMode ? '#e2e8f0' : '#1d5bbf',
    iconBlue: isDarkMode ? '#60a5fa' : '#3b82f6'
  };

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
    actual: isDarkMode ? '#38bdf8' : '#1d8cf8',
    actualBg: isDarkMode ? 'rgba(56, 189, 248, 0.12)' : 'rgba(29, 140, 248, 0.12)',
    xgb: isDarkMode ? '#fb7185' : '#e63946',
    xgbBg: isDarkMode ? 'rgba(251, 113, 133, 0.85)' : 'rgba(230, 57, 70, 0.85)',
    xgbBtnGradient: 'linear-gradient(90deg, #f43f5e 0%, #fb7185 100%)',
    lstm: isDarkMode ? '#34d399' : '#10b981',
    lstmBg: isDarkMode ? 'rgba(52, 211, 153, 0.85)' : 'rgba(16, 185, 129, 0.85)',
    lstmBtnGradient: 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
    hybrid: isDarkMode ? '#a78bfa' : '#8b5cf6',
    hybridBg: isDarkMode ? 'rgba(167, 139, 250, 0.85)' : 'rgba(139, 92, 246, 0.85)',
    hybridBtnGradient: 'linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%)'
  };

  const validateCsvFile = (file, label) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert(`Access Denied: Please upload the ${label} dataset in .csv format!`);
      return false;
    }
    if (file.size > 5 * 1024 * 1024) {
      alert('File is too large! Maximum allowed is 5MB.');
      return false;
    }
    return true;
  };

  const handleFileTrainChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (!validateCsvFile(selectedFile, 'Training')) {
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

    if (!validateCsvFile(selectedFile, 'Testing')) {
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
      const msg = 'Please upload BOTH files (Training & Testing Data) first!';
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
      const errMsg = err.response?.data?.message || 'Failed to connect to the AI backend. Please ensure the server is running.';
      if (algo === 'xgboost') setErrorXGB(`Backend Error: ${errMsg}`);
      else if (algo === 'lstm') setErrorLSTM(`Backend Error: ${errMsg}`);
      else setErrorHybrid(`Backend Error: ${errMsg}`);
    } finally {
      if (algo === 'xgboost') setLoadingXGB(false);
      else if (algo === 'lstm') setLoadingLSTM(false);
      else setLoadingHybrid(false);
    }
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to clear all analysis results and uploaded files?')) {
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

  const handleExportExcel = () => {
    const wb = XLSX.utils.book_new();

    const appendSheetWithAutoWidth = (workbook, rows, sheetName) => {
      if (!rows || rows.length === 0) return;
      const ws = XLSX.utils.json_to_sheet(rows);
      const allKeys = Array.from(
        rows.reduce((keySet, r) => {
          Object.keys(r || {}).forEach((k) => keySet.add(k));
          return keySet;
        }, new Set())
      );
      ws['!cols'] = allKeys.map((key) => {
        const maxCellLen = rows.reduce((maxLen, r) => {
          const val = r && r[key] !== undefined && r[key] !== null ? String(r[key]) : '';
          return Math.max(maxLen, val.length);
        }, key.length);
        return { wch: Math.min(Math.max(maxCellLen + 3, 14), 42) };
      });
      XLSX.utils.book_append_sheet(workbook, ws, sheetName.slice(0, 31));
    };

    const models = [
      { key: 'xgboost', name: 'XGBoost', data: resultXGB },
      { key: 'lstm', name: 'LSTM', data: resultLSTM },
      { key: 'hybrid', name: 'Hybrid (LSTM+XGB)', data: resultHybrid }
    ];
    const availableResult = resultHybrid || resultLSTM || resultXGB;
    if (!availableResult) return;

    // =========================================================================
    // SHEET 1: 1. Evaluation Metrics (RMSE, MAE, MAPE & Best Model per Horizon)
    // =========================================================================
    const metricsData = [];
    const bestMapeByHorizon = {};
    ['1', '3', '7'].forEach((h) => {
      let bestModel = null;
      let bestMape = Infinity;
      models.forEach((m) => {
        const mapeVal = m.data?.evaluasi?.[h]?.MAPE;
        if (typeof mapeVal === 'number' && mapeVal < bestMape) {
          bestMape = mapeVal;
          bestModel = m.name;
        }
      });
      bestMapeByHorizon[h] = bestModel;
    });

    models.forEach((model) => {
      if (model.data) {
        ['1', '3', '7'].forEach((h) => {
          if (model.data.evaluasi?.[h]) {
            metricsData.push({
              'Algorithm Model': model.name,
              'Target Horizon (Days)': parseInt(h, 10),
              'RMSE': model.data.evaluasi[h].RMSE,
              'MAE': model.data.evaluasi[h].MAE,
              'MAPE (%)': model.data.evaluasi[h].MAPE,
              'Evaluation Status':
                bestMapeByHorizon[h] === model.name ? 'Best Accuracy (Lowest MAPE)' : 'Evaluated'
            });
          }
        });
      }
    });
    appendSheetWithAutoWidth(wb, metricsData, "1. Evaluation Metrics");

    // =========================================================================
    // SHEET 2: 2. Future Projections (Blind Forecasting H+1, H+3, H+7)
    // =========================================================================
    const horizonLabels = ['1 Day Ahead (H+1)', '3 Days Ahead (H+3)', '7 Days Ahead (H+7)'];
    const forecastData = [];
    availableResult.grafik.future_dates.forEach((date, index) => {
      const row = {
        'Forecast Horizon': horizonLabels[index] || `Horizon ${index + 1}`,
        'Date': date
      };
      if (resultXGB) row['XGBoost Prediction (IDR)'] = resultXGB.grafik.future_prices[index];
      if (resultLSTM) row['LSTM Prediction (IDR)'] = resultLSTM.grafik.future_prices[index];
      if (resultHybrid) row['Hybrid Prediction (IDR)'] = resultHybrid.grafik.future_prices[index];
      forecastData.push(row);
    });
    appendSheetWithAutoWidth(wb, forecastData, "2. Future Projections");

    // =========================================================================
    // SHEET 3: 3. Dataset & Preprocessing (Split, Scaler & Sliding Window Info)
    // =========================================================================
    const info = availableResult.info_data || {};
    const datasetSummaryRows = [
      {
        'Category': 'Dataset Partition',
        'Parameter / Metric': 'Total Raw Dataset Rows',
        'Value': info.total_baris ?? '-',
        'Description': 'Gabungan seluruh observasi harian Data Training + Data Testing'
      },
      {
        'Category': 'Dataset Partition',
        'Parameter / Metric': 'Raw Training Rows (Data Latih)',
        'Value': info.data_training_mentah ?? '-',
        'Description': info.train_start_date
          ? `Periode: ${info.train_start_date} s.d. ${info.train_end_date}`
          : 'Data historis pelatihan model'
      },
      {
        'Category': 'Dataset Partition',
        'Parameter / Metric': 'Raw Testing Rows (Data Uji)',
        'Value': info.data_testing_mentah ?? '-',
        'Description': info.test_start_date
          ? `Periode: ${info.test_start_date} s.d. ${info.test_end_date}`
          : 'Data evaluasi pengujian model (Out-of-Sample)'
      },
      {
        'Category': 'Feature Engineering',
        'Parameter / Metric': 'Selected Input Features',
        'Value': 'Open (Index 0), Close (Index 1)',
        'Description': '2 fitur harga harian dengan target prediksi harga Close'
      },
      {
        'Category': 'Sliding Window',
        'Parameter / Metric': 'Lookback Window Size (w)',
        'Value': info.window_size ?? 30,
        'Description': 'Jumlah hari historis (30 hari) sebagai matriks sekuens input'
      },
      {
        'Category': 'Min-Max Normalization',
        'Parameter / Metric': 'Training Open Price Min (X_min)',
        'Value': info.scaler_min_open ?? '-',
        'Description': 'Batas minimum fitur Open pada Data Latih (fit hanya pada Data Latih)'
      },
      {
        'Category': 'Min-Max Normalization',
        'Parameter / Metric': 'Training Open Price Max (X_max)',
        'Value': info.scaler_max_open ?? '-',
        'Description': 'Batas maksimum fitur Open pada Data Latih'
      },
      {
        'Category': 'Min-Max Normalization',
        'Parameter / Metric': 'Training Close Price Min (X_min)',
        'Value': info.scaler_min_close ?? '-',
        'Description': 'Batas minimum fitur Close pada Data Latih (acuan denormalisasi target)'
      },
      {
        'Category': 'Min-Max Normalization',
        'Parameter / Metric': 'Training Close Price Max (X_max)',
        'Value': info.scaler_max_close ?? '-',
        'Description': 'Batas maksimum fitur Close pada Data Latih (acuan denormalisasi target)'
      }
    ];

    const horizonSeqInfo = info.horizons_sequence_info || {};
    ['1', '3', '7'].forEach((h) => {
      const hInfo = horizonSeqInfo[h];
      if (hInfo) {
        datasetSummaryRows.push({
          'Category': `Sliding Window (Horizon ${h} Hari)`,
          'Parameter / Metric': `Training & Testing Sequences (H+${h})`,
          'Value': `Train: ${hInfo.sekuens_training} sekuens | Test: ${hInfo.sekuens_testing} sekuens`,
          'Description': `Context Stitching: ${hInfo.context_stitching_rows} baris terakhir data latih (30 + ${h} - 1)`
        });
      }
    });
    appendSheetWithAutoWidth(wb, datasetSummaryRows, "3. Dataset & Preprocessing");

    // =========================================================================
    // SHEET 4: 4. Hyperparameter Tuning (XGBoost & Hybrid RandomizedSearchCV)
    // =========================================================================
    const tuningRows = [];
    models.forEach((model) => {
      if (!model.data?.analisis_training) return;
      ['1', '3', '7'].forEach((h) => {
        const xgbReport = model.data.analisis_training[h]?.xgboost_tuning;
        if (!xgbReport) return;

        const roleLabel =
          model.key === 'hybrid'
            ? 'Hybrid Stage-2 (XGBoost Residual Learner)'
            : 'Standalone XGBoost Regressor';

        (xgbReport.candidates || []).forEach((cand) => {
          tuningRows.push({
            'Algorithm Model': model.name,
            'Model Role': roleLabel,
            'Target Horizon (Days)': parseInt(h, 10),
            'Search Method': `${xgbReport.search_method} (${xgbReport.cv_folds}-Fold CV)`,
            'Iteration': cand.iteration,
            'Rank': cand.rank,
            'Is Best Parameter': cand.rank === 1 ? 'YES (Selected)' : 'No',
            'n_estimators': cand.n_estimators,
            'learning_rate': cand.learning_rate,
            'max_depth': cand.max_depth,
            'subsample': cand.subsample,
            'colsample_bytree': cand.colsample_bytree,
            'Fold 1 MAE (Scaled)': cand.fold_1_mae,
            'Fold 2 MAE (Scaled)': cand.fold_2_mae,
            'Fold 3 MAE (Scaled)': cand.fold_3_mae,
            'Mean CV MAE (Scaled)': cand.mean_cv_mae,
            'Std CV MAE (Scaled)': cand.std_cv_mae,
            'Mean Fit Time (sec)': cand.mean_fit_time_sec
          });
        });
      });
    });
    if (tuningRows.length > 0) {
      appendSheetWithAutoWidth(wb, tuningRows, "4. Hyperparameter Tuning");
    }

    // =========================================================================
    // SHEET 5: 5. Training Summary (LSTM EarlyStopping & Hybrid Residuals)
    // =========================================================================
    const trainingSummaryRows = [];
    const epochHistoryRows = [];

    models.forEach((model) => {
      if (!model.data?.analisis_training) return;
      ['1', '3', '7'].forEach((h) => {
        const hMeta = model.data.analisis_training[h];
        const lstmReport = hMeta?.lstm_training;
        const resStats = hMeta?.residual_stats;
        if (!lstmReport) return;

        trainingSummaryRows.push({
          'Algorithm Model': model.name,
          'Target Horizon (Days)': parseInt(h, 10),
          'Max Epochs Configured': lstmReport.max_epochs,
          'Total Epochs Executed': lstmReport.epochs_trained,
          'Best Epoch (Restored Weights)': lstmReport.best_epoch,
          'Early Stopping Triggered': lstmReport.stopped_early ? 'Yes (Stopped Early)' : 'No (Full Epochs)',
          'Early Stopping Patience': lstmReport.patience,
          'Batch Size': lstmReport.batch_size,
          'Train Samples (90%)': lstmReport.train_samples,
          'Validation Samples (10%)': lstmReport.val_samples,
          'Best Epoch Train Loss (MSE)': lstmReport.best_train_loss,
          'Best Epoch Val Loss (MSE)': lstmReport.best_val_loss,
          'Final Epoch Train Loss (MSE)': lstmReport.final_train_loss,
          'Final Epoch Val Loss (MSE)': lstmReport.final_val_loss,
          'Hybrid Train Residual Mean (Scaled)': resStats ? resStats.train_residual_mean_scaled : '-',
          'Hybrid Train Residual MAE (Scaled)': resStats ? resStats.train_residual_mae_scaled : '-',
          'Hybrid Test Residual Correction MAE (Scaled)': resStats ? resStats.test_residual_mae_scaled : '-'
        });

        (lstmReport.epoch_history || []).forEach((ep) => {
          epochHistoryRows.push({
            'Algorithm Model': model.name,
            'Target Horizon (Days)': parseInt(h, 10),
            'Epoch': ep.epoch,
            'Training Loss (MSE)': ep.train_loss,
            'Validation Loss (MSE)': ep.val_loss,
            'Generalization Gap (Val - Train)':
              ep.val_loss !== null && ep.val_loss !== undefined ? ep.val_loss - ep.train_loss : null,
            'Epoch Status': ep.is_best_epoch ? 'BEST EPOCH (Weights Restored)' : 'Trained'
          });
        });
      });
    });

    if (trainingSummaryRows.length > 0) {
      appendSheetWithAutoWidth(wb, trainingSummaryRows, "5. Training Process Summary");
    }
    if (epochHistoryRows.length > 0) {
      appendSheetWithAutoWidth(wb, epochHistoryRows, "6. LSTM Epoch Loss History");
    }

    // =========================================================================
    // SHEETS 7, 8, 9: Detail Testing per Horizon (Horizon 1, 3, and 7 Days)
    // =========================================================================
    const testDates = info.test_dates || [];
    const horizonSheetMap = [
      { horizon: '1', sheetName: '7. Detail Testing (Horizon 1)' },
      { horizon: '3', sheetName: '8. Detail Testing (Horizon 3)' },
      { horizon: '7', sheetName: '9. Detail Testing (Horizon 7)' }
    ];

    horizonSheetMap.forEach(({ horizon, sheetName }) => {
      const actuals = availableResult.evaluasi?.[horizon]?.evaluasi_detail?.actual;
      if (!actuals || actuals.length === 0) return;

      const testingData = [];
      actuals.forEach((actualVal, index) => {
        const row = {
          'Test Data Instance': index + 1,
          'Date': testDates[index] || `Day ${index + 1}`,
          'Target Horizon (Days)': parseInt(horizon, 10),
          'Actual Price (IDR)': actualVal
        };

        if (resultXGB?.evaluasi?.[horizon]) {
          const predXgb = resultXGB.evaluasi[horizon].evaluasi_detail.predicted[index];
          const errXgb = Math.abs(actualVal - predXgb);
          row['XGBoost Prediction (IDR)'] = predXgb;
          row['XGBoost Error (IDR)'] = errXgb;
          row['XGBoost APE (%)'] = actualVal ? (errXgb / actualVal) * 100 : 0;
        }

        if (resultLSTM?.evaluasi?.[horizon]) {
          const predLstm = resultLSTM.evaluasi[horizon].evaluasi_detail.predicted[index];
          const errLstm = Math.abs(actualVal - predLstm);
          row['LSTM Prediction (IDR)'] = predLstm;
          row['LSTM Error (IDR)'] = errLstm;
          row['LSTM APE (%)'] = actualVal ? (errLstm / actualVal) * 100 : 0;
        }

        if (resultHybrid?.evaluasi?.[horizon]) {
          const predHyb = resultHybrid.evaluasi[horizon].evaluasi_detail.predicted[index];
          const errHyb = Math.abs(actualVal - predHyb);
          row['Hybrid Prediction (IDR)'] = predHyb;
          row['Hybrid Error Difference (IDR)'] = errHyb;
          row['Hybrid APE (%)'] = actualVal ? (errHyb / actualVal) * 100 : 0;
        }

        testingData.push(row);
      });

      appendSheetWithAutoWidth(wb, testingData, sheetName);
    });

    XLSX.writeFile(wb, "Bitcoin_Analysis_Data.xlsx");
  };

  const formatIDR = (number) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(number);
  };

  const chartOptionsIDR = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    elements: { line: { tension: 0.25 } },
    scales: {
      x: {
        ticks: { color: theme.textMuted },
        grid: { color: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(226, 232, 240, 0.7)' }
      },
      y: {
        ticks: {
          color: theme.textMuted,
          callback: (value) => new Intl.NumberFormat('en-US', {
            notation: 'compact',
            compactDisplay: 'short'
          }).format(value)
        },
        grid: { color: isDarkMode ? 'rgba(255, 255, 255, 0.07)' : 'rgba(226, 232, 240, 0.85)' }
      }
    },
    plugins: {
      decimation: { enabled: true, algorithm: 'lttb' },
      zoom: {
        pan: { enabled: false, mode: 'x' },
        zoom: { wheel: { enabled: false }, pinch: { enabled: false }, mode: 'x' }
      },
      legend: { labels: { color: theme.textMuted } },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
        titleColor: '#f8fafc',
        bodyColor: '#dbeafe',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        callbacks: {
          label: function (context) {
            let label = context.dataset.label || '';
            if (label) { label += ': '; }
            if (context.parsed.y !== null) { label += formatIDR(context.parsed.y); }
            return label;
          }
        }
      }
    }
  };

  // Bar chart options for RMSE/MAE, displayed in IDR
  const barChartOptionsPrice = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { top: 34, bottom: 15 }
    },
    scales: {
      x: {
        ticks: { color: theme.textMuted, padding: 8, font: { size: 12 } },
        grid: { display: false }
      },
      y: {
        beginAtZero: true,
        ticks: {
          color: theme.textMuted,
          callback: function (value) {
            return 'IDR ' + new Intl.NumberFormat('en-US', { notation: 'compact', compactDisplay: 'short' }).format(value);
          }
        },
        title: { display: true, text: 'Error Value (IDR)', color: theme.textMuted },
        grid: { color: isDarkMode ? 'rgba(255, 255, 255, 0.07)' : 'rgba(226, 232, 240, 0.85)' }
      }
    },
    plugins: {
      legend: { labels: { color: theme.textMuted } },
      barValueLabels: {
        color: theme.textMain,
        formatter: (value) => new Intl.NumberFormat('en-US', {
          notation: 'compact',
          compactDisplay: 'short',
          maximumFractionDigits: 2
        }).format(value)
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
        titleColor: '#f8fafc',
        bodyColor: '#dbeafe',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        callbacks: {
          label: function (context) {
            return context.dataset.label + ': IDR ' + new Intl.NumberFormat('en-US').format(context.raw);
          }
        }
      }
    }
  };

  // Bar chart options for MAPE, displayed as a percentage (capped at 10%)
  const barChartOptionsMAPE = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { top: 34, bottom: 15 }
    },
    scales: {
      x: {
        ticks: { color: theme.textMuted, padding: 8, font: { size: 12 } },
        grid: { display: false }
      },
      y: {
        min: 0,
        max: 10,
        beginAtZero: true,
        ticks: {
          color: theme.textMuted,
          stepSize: 2,
          callback: function (value) {
            return value + '%';
          }
        },
        title: { display: true, text: 'Error Percentage (%)', color: theme.textMuted },
        grid: { color: isDarkMode ? 'rgba(255, 255, 255, 0.07)' : 'rgba(226, 232, 240, 0.85)' }
      }
    },
    plugins: {
      legend: { labels: { color: theme.textMuted } },
      barValueLabels: {
        color: theme.textMain,
        formatter: (value) => `${new Intl.NumberFormat('en-US', {
          maximumFractionDigits: 2
        }).format(value)}%`
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
        titleColor: '#f8fafc',
        bodyColor: '#dbeafe',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        callbacks: {
          label: function (context) {
            return context.dataset.label + ': ' + new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(context.raw) + '%';
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
        { label: 'Actual Historical Price', data: dataHistorical, borderColor: COLORS.actual, backgroundColor: COLORS.actualBg, borderWidth: 2, pointBackgroundColor: COLORS.actual, pointRadius: 3.5, fill: true },
        { label: 'Forecast Projection', data: dataPredicted, borderColor: colorConfig, borderDash: [5, 5], borderWidth: 2, pointBackgroundColor: colorConfig, pointRadius: 5 }
      ]
    };
  };

  const getBarChartData = (metric) => {
    return {
      labels: ['1 Day Ahead', '3 Days Ahead', '7 Days Ahead'],
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

    const labels = Array.from({ length: actualData.length }, (_, i) => `Test Data ${i + 1}`);
    const datasets = [{ label: 'Actual Price (Testing Set)', data: actualData, borderColor: COLORS.actual, backgroundColor: COLORS.actualBg, borderWidth: 2, fill: true, pointRadius: 0, pointHoverRadius: 5 }];

    if (hasDetailXGB) datasets.push({ label: 'XGBoost Prediction', data: resultXGB.evaluasi[horizon].evaluasi_detail.predicted, borderColor: COLORS.xgb, borderDash: [4, 4], borderWidth: 2, pointRadius: 0, pointHoverRadius: 5 });
    if (hasDetailLSTM) datasets.push({ label: 'LSTM Prediction', data: resultLSTM.evaluasi[horizon].evaluasi_detail.predicted, borderColor: COLORS.lstm, borderDash: [4, 4], borderWidth: 2, pointRadius: 0, pointHoverRadius: 5 });
    if (hasDetailHybrid) datasets.push({ label: 'Hybrid Prediction (LSTM+XGB)', data: resultHybrid.evaluasi[horizon].evaluasi_detail.predicted, borderColor: COLORS.hybrid, borderDash: [4, 4], borderWidth: 2, pointRadius: 0, pointHoverRadius: 5 });

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

    const formatNumber = (num) => num === Infinity ? '-' : new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(num);

    return (
      <div className="card-hover insight-summary" style={{ display: 'flex', gap: '20px', marginBottom: '30px' }}>
        <div className="glass-inner-card" style={{ flex: 1, padding: '20px', backgroundColor: theme.innerCard, borderLeft: `5px solid ${winnerColor}`, borderRadius: '10px', borderTop: `1px solid ${theme.border}`, borderRight: `1px solid ${theme.border}`, borderBottom: `1px solid ${theme.border}`, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h4 style={{ margin: '0 0 5px 0', color: theme.textMuted, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.03em' }}>Accuracy Champion (Best Model)</h4>
          <strong style={{ fontSize: '26px', color: winnerColor }}>{winner}</strong>
        </div>
        <div className="glass-inner-card" style={{ flex: 1, padding: '20px', backgroundColor: theme.innerCard, borderRadius: '10px', border: `1px solid ${theme.border}` }}>
          <h4 style={{ margin: '0 0 10px 0', color: theme.textMain, fontSize: '14px' }}>Average Error (MAPE)</h4>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ color: COLORS.xgb, fontWeight: '600' }}>XGBoost:</span>
            <span className="summary-value" style={{ color: theme.textMain, fontWeight: 'bold' }}>{resultXGB ? `${formatNumber(mapeXGB)}%` : 'Not trained yet'}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ color: COLORS.lstm, fontWeight: '600' }}>LSTM:</span>
            <span className="summary-value" style={{ color: theme.textMain, fontWeight: 'bold' }}>{resultLSTM ? `${formatNumber(mapeLSTM)}%` : 'Not trained yet'}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: COLORS.hybrid, fontWeight: '600' }}>Hybrid:</span>
            <span className="summary-value" style={{ color: theme.textMain, fontWeight: 'bold' }}>{resultHybrid ? `${formatNumber(mapeHybrid)}%` : 'Not trained yet'}</span>
          </div>
        </div>
      </div>
    );
  }, [
    resultXGB,
    resultLSTM,
    resultHybrid,
    COLORS.hybrid,
    COLORS.lstm,
    COLORS.xgb,
    theme.border,
    theme.innerCard,
    theme.textMain,
    theme.textMuted
  ]);

  const getTabBackground = (tabName, lightColorBg, darkSubtleBg) => {
    if (activeTab !== tabName) return theme.tabInactiveBg;
    return isDarkMode ? darkSubtleBg : lightColorBg;
  };

  const getTabBorder = (tabName, lightBorderColor, darkBorderColor) => {
    if (activeTab !== tabName) return `1px solid ${theme.border}`;
    return isDarkMode ? `1px solid ${darkBorderColor}` : `1px solid ${lightBorderColor}`;
  };

  const getTabShadow = (tabName) => {
    if (activeTab !== tabName) {
      return isDarkMode ? '0 2px 8px rgba(0, 0, 0, 0.2)' : '0 2px 6px rgba(15, 23, 42, 0.02)';
    }
    return isDarkMode
      ? '0 4px 14px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08)'
      : '0 4px 12px rgba(15, 23, 42, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.9)';
  };

  const renderTabContent = (algo, result, loading, error, colorBtn, title, description) => {
    let runBtnBackground = colorBtn;
    if (loading) {
      runBtnBackground = isDarkMode ? '#1e293b' : '#334155';
    } else if (algo === 'xgboost') {
      runBtnBackground = COLORS.xgbBtnGradient;
    } else if (algo === 'lstm') {
      runBtnBackground = COLORS.lstmBtnGradient;
    } else if (algo === 'hybrid') {
      runBtnBackground = COLORS.hybridBtnGradient;
    }

    const runBtnBorder = loading
      ? `1px solid ${theme.border}`
      : '1px solid rgba(255, 255, 255, 0.16)';

    const runBtnShadow = loading
      ? 'none'
      : (isDarkMode ? '0 4px 14px rgba(0, 0, 0, 0.35)' : `0 4px 12px ${colorBtn}28`);

    return (
      <div className="tab-content" style={{ padding: '30px', backgroundColor: theme.card, border: `1px solid ${theme.border}`, borderRadius: '16px', boxShadow: isDarkMode ? '0 8px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06)' : '0 4px 20px rgba(15, 23, 42, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.8)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 style={{ color: theme.textMain, margin: 0, display: 'flex', alignItems: 'center', gap: '8px', fontSize: '19px' }}>
            {algo === 'xgboost' && <Target size={24} color={colorBtn} />}
            {algo === 'lstm' && <BrainCircuit size={24} color={colorBtn} />}
            {algo === 'hybrid' && <Bot size={24} color={colorBtn} />}
            {title} Prediction Engine
          </h3>
          <span style={{ fontSize: '12px', padding: '6px 14px', backgroundColor: theme.badgeBg, border: theme.badgeBorder, borderRadius: '20px', color: theme.badgeText, fontWeight: '600' }}>{description}</span>
        </div>

        {error && (
          <div style={{ backgroundColor: isDarkMode ? 'rgba(220, 38, 38, 0.15)' : '#fef2f2', color: isDarkMode ? '#fca5a5' : '#991b1b', padding: '14px', borderRadius: '10px', marginBottom: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontWeight: '600', border: isDarkMode ? '1px solid rgba(248, 113, 113, 0.3)' : '1px solid #fecaca' }}>
            <AlertTriangle size={20} /> {error}
          </div>
        )}

        <button onClick={() => handlePredict(algo)} disabled={loading} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px', width: '100%', padding: '15px', background: runBtnBackground, color: '#ffffff', border: runBtnBorder, borderRadius: '10px', fontWeight: 'bold', fontSize: '15px', cursor: loading ? 'wait' : 'pointer', transition: 'all 0.25s', marginBottom: '25px', boxShadow: runBtnShadow }}>
          {loading ? (
            <>
              <span style={{ width: '18px', height: '18px', border: '2.5px solid rgba(255,255,255,0.25)', borderTop: `2.5px solid ${colorBtn}`, borderRadius: '50%', display: 'inline-block', animation: 'spin 0.9s linear infinite' }} />
              Processing Analytical Model...
            </>
          ) : (
            <>
              <TrendingUp size={19} /> Run {title} Prediction
            </>
          )}
        </button>

        {result && (
          <div className="card-hover glass-inner-card" style={{ border: `1px solid ${theme.border}`, padding: '25px', borderRadius: '14px', backgroundColor: theme.innerCard }}>
            <h4 style={{ color: theme.textMain, marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '16px' }}>
              <LineChart size={20} color={colorBtn} /> Future Price Projection (Blind Forecasting)
            </h4>
            <div className="forecast-cards" style={{ display: 'flex', gap: '16px', marginBottom: '25px' }}>
              {[1, 3, 7].map(hari => (
                <div key={hari} className="glass-inner-card" style={{ flex: 1, backgroundColor: theme.innerCard, padding: '18px 20px', borderRadius: '12px', textAlign: 'center', border: `1px solid ${theme.border}`, position: 'relative' }}>
                  <CalendarCheck size={17} color={theme.textMuted} style={{ position: 'absolute', top: '16px', left: '16px', opacity: 0.7 }} />
                  <div style={{ fontSize: '12px', color: theme.textMuted, fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.03em' }}>{hari} Day(s) Ahead</div>
                  <div style={{ fontSize: '21px', color: colorBtn, fontWeight: '800' }}>{formatIDR(result.hasil_prediksi[hari])}</div>
                </div>
              ))}
            </div>
            <div className="glass-inner-card" style={{ height: '400px', width: '100%', padding: '15px', backgroundColor: theme.innerCard, border: `1px solid ${theme.border}`, borderRadius: '12px' }}>
              <Line data={getChartData(result, colorBtn)} options={chartOptionsIDR} />
            </div>
            <p style={{ textAlign: 'center', fontSize: '12px', color: theme.textMuted, marginTop: '15px' }}>
              *The chart connects the historical data of the last month with future projections. <b>Use scroll/pinch to Zoom, and drag to Pan.</b>
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderEvaluasiTab = () => {
    const availableResult = resultXGB || resultLSTM || resultHybrid;

    return (
      <div className="tab-content evaluation-content" style={{ padding: '30px', backgroundColor: theme.card, border: `1px solid ${theme.border}`, borderRadius: '16px', boxShadow: isDarkMode ? '0 8px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06)' : '0 4px 20px rgba(15, 23, 42, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.8)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
          <h3 style={{ color: theme.textMain, margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={24} color={COLORS.actual} /> Model Evaluation & Comparison (Out-of-Sample Testing)
          </h3>
          {availableResult && (
            <button onClick={handleExportExcel} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '10px 18px', background: '#10b981', color: '#ffffff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontSize: '14px', fontWeight: '600', transition: '0.2s' }}>
              <FileSpreadsheet size={18} /> Export to Excel
            </button>
          )}
        </div>

        {!availableResult ? (
          <div className="glass-inner-card" style={{ textAlign: 'center', padding: '60px 20px', color: theme.textMuted, backgroundColor: theme.uploadBoxBg, borderRadius: '12px', border: `1.5px dashed ${theme.uploadBorder}` }}>
            <Inbox size={48} color={theme.textMuted} style={{ margin: '0 auto 15px auto', opacity: 0.6 }} />
            <h3 style={{ margin: '0 0 10px 0', color: theme.textMain }}>No Evaluation Data Yet</h3>
            <p style={{ margin: 0, fontSize: '15px' }}>Please run a prediction on one of the algorithm tabs first to view the comprehensive analysis.</p>
          </div>
        ) : (
          <div>
            {insightSummary}
            <h4 style={{ color: theme.textMain, borderBottom: `1px solid ${theme.border}`, paddingBottom: '10px', marginBottom: '20px' }}>1. Research Dataset Proportion (Explicit Split)</h4>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginBottom: '40px' }}>
              <div className="card-hover glass-inner-card" style={{ flex: 1, backgroundColor: theme.innerCard, padding: '20px', borderRadius: '12px', border: `1px solid ${theme.border}`, textAlign: 'center' }}>
                <span style={{ fontSize: '12px', color: theme.textMuted, display: 'block', fontWeight: '600', textTransform: 'uppercase', marginBottom: '6px', letterSpacing: '0.03em' }}>Training Data (Learning)</span>
                <strong style={{ fontSize: '26px', color: COLORS.actual }}>{availableResult.info_data.data_training_mentah} Rows</strong>
                <div style={{ fontSize: '12px', color: theme.textMuted, marginTop: '5px' }}>({availableResult.info_data.sekuens_training} Fit Sequences)</div>
              </div>
              <div className="card-hover glass-inner-card" style={{ flex: 1, backgroundColor: theme.innerCard, padding: '20px', borderRadius: '12px', border: `1px solid ${theme.border}`, textAlign: 'center' }}>
                <span style={{ fontSize: '12px', color: theme.textMuted, display: 'block', fontWeight: '600', textTransform: 'uppercase', marginBottom: '6px', letterSpacing: '0.03em' }}>Testing Data (Evaluation)</span>
                <strong style={{ fontSize: '26px', color: COLORS.xgb }}>{availableResult.info_data.data_testing_mentah} Rows</strong>
                <div style={{ fontSize: '12px', color: theme.textMuted, marginTop: '5px' }}>({availableResult.info_data.sekuens_testing} Test Sequences)</div>
              </div>
            </div>

            <h4 style={{ color: theme.textMain, borderBottom: `1px solid ${theme.border}`, paddingBottom: '10px', marginBottom: '20px' }}>2. Prediction vs. Actual Visualization (Testing Data)</h4>
            <div className="card-hover glass-inner-card" style={{ marginBottom: '40px', padding: '25px', border: `1px solid ${theme.border}`, borderRadius: '14px', backgroundColor: theme.innerCard }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <span style={{ fontSize: '15px', color: theme.textMain, fontWeight: '600' }}>Target Horizon Review:</span>
                <select className="horizon-select" value={evalHorizon} onChange={(e) => setEvalHorizon(e.target.value)} style={{ padding: '9px 18px', borderRadius: '8px', border: `1px solid ${theme.border}`, color: theme.textMain, fontWeight: '600', outline: 'none', cursor: 'pointer', backgroundColor: theme.uploadBoxBg }}>
                  <option value="1" style={{ color: theme.textMain, backgroundColor: isDarkMode ? '#111827' : '#ffffff' }}>1 Day Ahead</option>
                  <option value="3" style={{ color: theme.textMain, backgroundColor: isDarkMode ? '#111827' : '#ffffff' }}>3 Days Ahead</option>
                  <option value="7" style={{ color: theme.textMain, backgroundColor: isDarkMode ? '#111827' : '#ffffff' }}>7 Days Ahead</option>
                </select>
              </div>
              {getLineComparasiData(evalHorizon) ? (
                <div style={{ height: '380px', width: '100%' }}><Line data={getLineComparasiData(evalHorizon)} options={chartOptionsIDR} /></div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: theme.textMuted, fontStyle: 'italic', backgroundColor: theme.innerCard, borderRadius: '8px', border: `1px dashed ${theme.border}` }}>Waiting for detailed evaluation data extraction.</div>
              )}
            </div>

            <h4 style={{ color: theme.textMain, borderBottom: `1px solid ${theme.border}`, paddingBottom: '10px', marginBottom: '20px' }}>3. Error Metrics Recapitulation</h4>
            <div className="metric-grid" style={{ display: 'grid', gap: '24px' }}>
              <div className="card-hover metric-card glass-inner-card" style={{ height: '500px', backgroundColor: theme.innerCard, padding: '20px', borderRadius: '14px', border: `1px solid ${theme.border}` }}>
                <h4 style={{ textAlign: 'center', margin: '0 0 15px 0', color: theme.textMain }}>RMSE (Root Mean Squared Error)</h4>
                <div className="metric-chart"><Bar data={getBarChartData('RMSE')} options={barChartOptionsPrice} plugins={[barValueLabelsPlugin]} /></div>
              </div>
              <div className="card-hover metric-card glass-inner-card" style={{ height: '500px', backgroundColor: theme.innerCard, padding: '20px', borderRadius: '14px', border: `1px solid ${theme.border}` }}>
                <h4 style={{ textAlign: 'center', margin: '0 0 15px 0', color: theme.textMain }}>MAE (Mean Absolute Error)</h4>
                <div className="metric-chart"><Bar data={getBarChartData('MAE')} options={barChartOptionsPrice} plugins={[barValueLabelsPlugin]} /></div>
              </div>
              <div className="card-hover metric-card metric-card-wide glass-inner-card" style={{ height: '500px', backgroundColor: theme.innerCard, padding: '20px', borderRadius: '14px', border: `1px solid ${theme.border}` }}>
                <h4 style={{ textAlign: 'center', margin: '0 0 15px 0', color: theme.textMain }}>MAPE (Mean Absolute Percentage Error)</h4>
                <div className="metric-chart"><Bar data={getBarChartData('MAPE')} options={barChartOptionsMAPE} plugins={[barValueLabelsPlugin]} /></div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const isTabDisabled = loadingXGB || loadingLSTM || loadingHybrid;

  return (
    <div className={`app-shell ${isDarkMode ? 'theme-dark' : 'theme-light'}`} style={{ fontFamily: '"Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', backgroundColor: theme.bg, minHeight: '100vh', padding: '32px 20px' }}>
      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      `}</style>

      <div className="dashboard-container" style={{ maxWidth: '1150px', margin: '0 auto' }}>

        <div className="card-hover dashboard-header" style={{ backgroundColor: theme.card, padding: '32px 35px', borderRadius: '16px', marginBottom: '20px', position: 'relative' }}>
          <div style={{ position: 'absolute', top: '22px', right: '25px', display: 'flex', gap: '10px' }}>
            <button onClick={() => setIsDarkMode(!isDarkMode)} style={{ padding: '8px 14px', backgroundColor: theme.toggleBg, color: theme.toggleText, border: `1px solid ${theme.toggleBorder}`, borderRadius: '8px', cursor: 'pointer', fontWeight: '600', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', transition: '0.2s' }}>
              {isDarkMode ? <Moon size={15} /> : <Sun size={15} />}
              {isDarkMode ? 'Dark Mode' : 'Light Mode'}
            </button>
            <button onClick={handleReset} style={{ padding: '8px 14px', backgroundColor: '#e63946', color: '#ffffff', border: '1px solid #d62828', borderRadius: '8px', cursor: 'pointer', fontWeight: '600', fontSize: '13px', transition: '0.2s', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <RotateCcw size={15} /> Reset Data
            </button>
          </div>

          <div className="brand-lockup">
            <img
              className="brand-logo"
              src="/Binary Bitcoin Techno Medallion.png"
              alt="Binary Bitcoin Techno Medallion"
            />
            <h2 style={{ textAlign: 'center', color: theme.textMain, margin: 0, fontSize: '27px', fontWeight: '800' }}>
              Bitcoin Time-Series Analysis Dashboard
            </h2>
          </div>
          <p style={{ textAlign: 'center', color: theme.textMuted, fontSize: '15px', margin: '0 0 22px 0', fontWeight: '500' }}>Sapar Hidayat. S (2201020003)</p>

          <div className="upload-grid" style={{ display: 'flex', gap: '20px', transition: '0.3s', opacity: isTabDisabled ? 0.6 : 1, marginTop: '24px' }}>
            <div className="upload-card" style={{ flex: 1, backgroundColor: theme.uploadBoxBg, padding: '20px', borderRadius: '12px', border: `1.5px dashed ${theme.uploadBorder}` }}>
              <label style={{ fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: theme.textMain, fontSize: '15px' }}>
                <History size={19} color={theme.iconBlue} /> 1. Upload Historical CSV (Training Data)
              </label>
              <p style={{ margin: '0 0 12px 0', fontSize: '12px', color: theme.textMuted }}>Bitcoin Data Jan 1, 2020 - Dec 31, 2025.</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', backgroundColor: theme.inputBg, border: `1px solid ${theme.inputBorder}`, borderRadius: '8px', padding: '6px 12px' }}>
                <Upload size={18} color={theme.textMuted} />
                <input id="input-train" type="file" accept=".csv" onChange={handleFileTrainChange} disabled={isTabDisabled} style={{ flex: 1, padding: '4px', backgroundColor: 'transparent', color: theme.textMuted, border: 'none', cursor: isTabDisabled ? 'not-allowed' : 'pointer', fontSize: '13px' }} />
              </div>
            </div>

            <div className="upload-card" style={{ flex: 1, backgroundColor: theme.uploadBoxBg, padding: '20px', borderRadius: '12px', border: `1.5px dashed ${theme.uploadBorder}` }}>
              <label style={{ fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: theme.textMain, fontSize: '15px' }}>
                <CalendarCheck size={19} color={theme.iconBlue} /> 2. Upload Actual CSV (Testing Data)
              </label>
              <p style={{ margin: '0 0 12px 0', fontSize: '12px', color: theme.textMuted }}>Bitcoin Data Jan 1, 2026 - Present.</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', backgroundColor: theme.inputBg, border: `1px solid ${theme.inputBorder}`, borderRadius: '8px', padding: '6px 12px' }}>
                <Upload size={18} color={theme.textMuted} />
                <input id="input-test" type="file" accept=".csv" onChange={handleFileTestChange} disabled={isTabDisabled} style={{ flex: 1, padding: '4px', backgroundColor: 'transparent', color: theme.textMuted, border: 'none', cursor: isTabDisabled ? 'not-allowed' : 'pointer', fontSize: '13px' }} />
              </div>
            </div>
          </div>
        </div>

        <div className="tab-navigation" style={{ display: 'flex', opacity: isTabDisabled ? 0.6 : 1, pointerEvents: isTabDisabled ? 'none' : 'auto', gap: '12px', marginBottom: '16px', zIndex: 1, position: 'relative' }}>
          <button onClick={() => setActiveTab('xgboost')} style={{ flex: 1, padding: '14px 16px', background: getTabBackground('xgboost', 'linear-gradient(90deg, #ffe5e7 0%, #fff5f5 100%)', 'rgba(251, 113, 133, 0.12)'), color: activeTab === 'xgboost' ? COLORS.xgb : theme.textMain, border: getTabBorder('xgboost', '#f8b4b9', 'rgba(251, 113, 133, 0.35)'), borderRadius: '10px', cursor: 'pointer', fontWeight: '700', fontSize: '14px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', boxShadow: getTabShadow('xgboost') }}>
            <Target size={19} color={activeTab === 'xgboost' ? COLORS.xgb : theme.textMain} /> XGBoost Prediction
          </button>
          <button onClick={() => setActiveTab('lstm')} style={{ flex: 1, padding: '14px 16px', background: getTabBackground('lstm', 'linear-gradient(90deg, #dcfce7 0%, #f0fdf4 100%)', 'rgba(52, 211, 153, 0.12)'), color: activeTab === 'lstm' ? COLORS.lstm : theme.textMain, border: getTabBorder('lstm', '#86efac', 'rgba(52, 211, 153, 0.35)'), borderRadius: '10px', cursor: 'pointer', fontWeight: '700', fontSize: '14px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', boxShadow: getTabShadow('lstm') }}>
            <BrainCircuit size={19} color={activeTab === 'lstm' ? COLORS.lstm : theme.textMain} /> LSTM Prediction
          </button>
          <button onClick={() => setActiveTab('hybrid')} style={{ flex: 1, padding: '14px 16px', background: getTabBackground('hybrid', 'linear-gradient(90deg, #f3e8ff 0%, #faf5ff 100%)', 'rgba(167, 139, 250, 0.12)'), color: activeTab === 'hybrid' ? COLORS.hybrid : theme.textMain, border: getTabBorder('hybrid', '#d8b4fe', 'rgba(167, 139, 250, 0.35)'), borderRadius: '10px', cursor: 'pointer', fontWeight: '700', fontSize: '14px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', boxShadow: getTabShadow('hybrid') }}>
            <Bot size={19} color={activeTab === 'hybrid' ? COLORS.hybrid : theme.textMain} /> Hybrid Prediction
          </button>
          <button onClick={() => setActiveTab('evaluasi')} style={{ flex: 1, padding: '14px 16px', background: getTabBackground('evaluasi', 'linear-gradient(90deg, #e0f2fe 0%, #f0f9ff 100%)', 'rgba(56, 189, 248, 0.12)'), color: activeTab === 'evaluasi' ? COLORS.actual : theme.textMain, border: getTabBorder('evaluasi', '#93c5fd', 'rgba(56, 189, 248, 0.35)'), borderRadius: '10px', cursor: 'pointer', fontWeight: '700', fontSize: '14px', transition: '0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', boxShadow: getTabShadow('evaluasi') }}>
            <BarChart3 size={19} color={activeTab === 'evaluasi' ? COLORS.actual : theme.textMain} /> Model Comparison
          </button>
        </div>

        <div style={{ zIndex: 0, position: 'relative' }}>
          {activeTab === 'xgboost' && renderTabContent('xgboost', resultXGB, loadingXGB, errorXGB, COLORS.xgb, 'XGBoost', 'Boosting Tree (Tuning)')}
          {activeTab === 'lstm' && renderTabContent('lstm', resultLSTM, loadingLSTM, errorLSTM, COLORS.lstm, 'LSTM', 'Neural Network (30 Epochs)')}
          {activeTab === 'hybrid' && renderTabContent('hybrid', resultHybrid, loadingHybrid, errorHybrid, COLORS.hybrid, 'Hybrid (LSTM ➔ XGB)', 'Stacking Sequential')}
          {activeTab === 'evaluasi' && renderEvaluasiTab()}
        </div>

      </div>
    </div>
  );
}

export default App;
