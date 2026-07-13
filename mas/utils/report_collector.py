import os
import json
from datetime import datetime
import numpy as np

class ReportCollector:
    _instance = None
    _lock = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ReportCollector, cls).__new__(cls, *args, **kwargs)
            cls._instance._init()
        return cls._instance

    def _init(self):
        import threading
        self._lock = threading.Lock()
        self.selection_data = {}  # animal_id -> list of dicts: {depth_filename, label, suitable, prob}
        self.prediction_data = {}  # animal_id -> list of dicts: {depth_filename, predicted_weight}
        self.final_predictions = {}  # animal_id -> float
        self.mode = None
        self.fps = None

    def reset(self):
        with self._lock:
            self.selection_data.clear()
            self.prediction_data.clear()
            self.final_predictions.clear()
            self.mode = None
            self.fps = None

    def record_selection(self, animal_id, depth_filename, label, suitable, prob):
        with self._lock:
            self.selection_data.setdefault(animal_id, []).append({
                "depth_filename": depth_filename,
                "label": label,
                "suitable": suitable,
                "prob": prob
            })

    def record_prediction(self, animal_id, depth_filename, predicted_weight):
        with self._lock:
            self.prediction_data.setdefault(animal_id, []).append({
                "depth_filename": depth_filename,
                "predicted_weight": predicted_weight
            })

    def record_final_prediction(self, animal_id, final_weight):
        with self._lock:
            self.final_predictions[animal_id] = final_weight

    def generate_report(self, reports_dir, mode, fps, capture_mode=None):
        with self._lock:
            report_path = os.path.join(reports_dir, "report.md")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Load real weights from data/exp1/animal-tags/<tag>/weight.json
            real_weights = {}
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            tags_dir = os.path.join(project_root, 'data', 'exp1', 'animal-tags')
            for tag in self.selection_data.keys():
                w_path = os.path.join(tags_dir, tag, 'weight.json')
                if os.path.exists(w_path):
                    try:
                        with open(w_path, 'r', encoding='utf-8') as wf:
                            real_weights[tag] = json.load(wf).get("weight")
                    except Exception:
                        pass

            lines = []
            lines.append(f"# Relatório de Execução - {os.path.basename(reports_dir)}")
            lines.append("")
            lines.append(f"- **Modo de Inferencia:** {mode}")
            if capture_mode == "native-timestamps":
                lines.append(
                    "- **Fonte temporal:** timestamps originais de "
                    "simulation_index.json"
                )
            else:
                lines.append(f"- **FPS Simulado:** {fps if fps is not None else 'N/A'}")
            lines.append(f"- **Data/Hora de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # --- Executive Summary Table ---
            lines.append("## Sumário Executivo")
            lines.append("")
            lines.append("| Animal (Tag) | Peso Real (kg) | Peso Predito (kg) | Erro Absoluto (kg) | Erro Relativo (%) | Total Frames | Suitable (Pred) | Suitable (GT) | Precision | Recall | F1-Score |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            
            for tag in sorted(self.selection_data.keys()):
                real_w = real_weights.get(tag)
                pred_w = self.final_predictions.get(tag)
                
                # Selection stats
                frames = self.selection_data.get(tag, [])
                total_frames = len(frames)
                suitable_pred = sum(1 for f in frames if f["suitable"])
                suitable_gt = sum(1 for f in frames if f["label"] == "suited")
                
                tp = sum(1 for f in frames if f["suitable"] and f["label"] == "suited")
                fp = sum(1 for f in frames if f["suitable"] and f["label"] != "suited")
                fn = sum(1 for f in frames if not f["suitable"] and f["label"] == "suited")
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Errors
                if real_w is not None and pred_w is not None:
                    err_abs = abs(pred_w - real_w)
                    err_rel = (err_abs / real_w) * 100 if real_w > 0 else 0.0
                    err_abs_str = f"{err_abs:.4f}"
                    err_rel_str = f"{err_rel:.2f}%"
                    real_w_str = f"{real_w:.4f}"
                    pred_w_str = f"{pred_w:.4f}"
                else:
                    err_abs_str = "-"
                    err_rel_str = "-"
                    real_w_str = f"{real_w:.4f}" if real_w is not None else "-"
                    pred_w_str = f"{pred_w:.4f}" if pred_w is not None else "-"
                
                lines.append(
                    f"| {tag} | {real_w_str} | {pred_w_str} | {err_abs_str} | {err_rel_str} | "
                    f"{total_frames} | {suitable_pred} | {suitable_gt} | "
                    f"{precision:.2f} | {recall:.2f} | {f1:.2f} |"
                )
            
            lines.append("")
            
            # --- Detailed Animal Sections ---
            lines.append("## Detalhamento por Animal")
            lines.append("")
            
            for tag in sorted(self.selection_data.keys()):
                lines.append(f"### Animal {tag}")
                lines.append("")
                
                # Weight summary
                real_w = real_weights.get(tag)
                pred_w = self.final_predictions.get(tag)
                lines.append("#### Comparação de Peso Final")
                if real_w is not None and pred_w is not None:
                    err_abs = abs(pred_w - real_w)
                    err_rel = (err_abs / real_w) * 100 if real_w > 0 else 0.0
                    lines.append(f"- **Peso Real:** {real_w:.4f} kg")
                    lines.append(f"- **Peso Predito:** {pred_w:.4f} kg")
                    lines.append(f"- **Erro Absoluto:** {err_abs:.4f} kg")
                    lines.append(f"- **Erro Relativo:** {err_rel:.2f}%")
                else:
                    real_w_str = f"{real_w:.4f}" if real_w is not None else "Não encontrado"
                    pred_w_str = f"{pred_w:.4f}" if pred_w is not None else "Não calculado"
                    lines.append(f"- **Peso Real:** {real_w_str}")
                    lines.append(f"- **Peso Predito:** {pred_w_str}")
                lines.append("")
                
                # Selection details
                frames = self.selection_data.get(tag, [])
                tp = sum(1 for f in frames if f["suitable"] and f["label"] == "suited")
                fp = sum(1 for f in frames if f["suitable"] and f["label"] != "suited")
                fn = sum(1 for f in frames if not f["suitable"] and f["label"] == "suited")
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                lines.append("#### Avaliação do Seletor (Suited / Not Suited)")
                lines.append(f"- **Verdadeiros Positivos (TP):** {tp} (Frames suited classificados como suited)")
                lines.append(f"- **Falsos Positivos (FP):** {fp} (Frames não-suited classificados como suited)")
                lines.append(f"- **Falsos Negativos (FN):** {fn} (Frames suited descartados)")
                lines.append(f"- **Precisão:** {precision:.4f}")
                lines.append(f"- **Recall:** {recall:.4f}")
                lines.append(f"- **F1-Score:** {f1:.4f}")
                lines.append("")
                
                # Filter interesting frames:
                # - suitable=True (TP or FP)
                # - suitable=False & label='suited' (FN)
                interesting_frames = [
                    f for f in frames if f["suitable"] or (not f["suitable"] and f["label"] == "suited")
                ]
                
                lines.append("##### Frames de Interesse (TP, FP ou FN)")
                if interesting_frames:
                    lines.append("| Frame (Arquivo) | Ground Truth (Real) | Decisão do Seletor | Classificação | Confiança (Score) |")
                    lines.append("|---|---|---|---|---|")
                    for f in interesting_frames:
                        gt = f["label"] if f["label"] else "desconhecido"
                        dec = "SUITABLE" if f["suitable"] else "DISCARDED"
                        
                        # Classification label
                        if f["suitable"] and f["label"] == "suited":
                            classification = "TP (True Positive)"
                        elif f["suitable"] and f["label"] != "suited":
                            classification = "FP (False Positive)"
                        elif not f["suitable"] and f["label"] == "suited":
                            classification = "FN (False Negative)"
                        else:
                            classification = "TN (True Negative) - N/A"
                            
                        lines.append(f"| {f['depth_filename']} | {gt} | {dec} | {classification} | {f['prob']:.4f} |")
                else:
                    lines.append("*Nenhum frame de interesse (nenhum TP, FP ou FN).*")
                lines.append("")
                
                # Single weight inferences per frame
                if mode == "single":
                    preds = self.prediction_data.get(tag, [])
                    lines.append("#### Inferências de Peso por Frame")
                    if preds:
                        lines.append("| Frame (Arquivo) | Peso Predito (kg) | Peso Real (kg) | Erro Absoluto (kg) | Erro Relativo (%) |")
                        lines.append("|---|---|---|---|---|")
                        for p in preds:
                            pred_val = p["predicted_weight"]
                            if real_w is not None and pred_val is not None:
                                err_abs = abs(pred_val - real_w)
                                err_rel = (err_abs / real_w) * 100 if real_w > 0 else 0.0
                                err_abs_str = f"{err_abs:.4f}"
                                err_rel_str = f"{err_rel:.2f}%"
                            else:
                                err_abs_str = "-"
                                err_rel_str = "-"
                            
                            pred_val_str = f"{pred_val:.4f}" if pred_val is not None else "-"
                            real_w_str = f"{real_w:.4f}" if real_w is not None else "-"
                            
                            lines.append(f"| {p['depth_filename']} | {pred_val_str} | {real_w_str} | {err_abs_str} | {err_rel_str} |")
                    else:
                        lines.append("*Nenhuma inferência realizada (nenhum frame adequado).*")
                    lines.append("")
                
                lines.append("---")
                lines.append("")

            with open(report_path, 'w', encoding='utf-8') as rf:
                rf.write("\n".join(lines))
