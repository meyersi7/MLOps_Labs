from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.vision import SingleDatasetCheck, VisionData
from deepchecks.vision.context import Context
import plotly.express as px
import torch
import torch.nn.functional as F
import numpy as np

from typing import Any


class FGSMAttackCheck(SingleDatasetCheck):
    """A check to test if the model is robust to FGSM attacks."""

    def __init__(
        self, device: str = "cpu", epsilon: float = 1e-4, model=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.epsilon = epsilon
        self.model = model # Speichert das Modell für den Angriff

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        # Initialisiert die Zähler im Cache
        self.cache = {"failing_samples": 0, "total_samples": 0}

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        model = self.model
        model.to(self.device)
        model.eval() # Modell in den Evaluationsmodus versetzen

        # Wir iterieren durch die Bilder im Batch
        batch_data = zip(batch.original_images, batch.original_labels)
        for sample in batch_data:
            x_raw, y_raw = sample
            
            # Daten vorbereiten und Gradienten-Tracking aktivieren
            x = torch.tensor(x_raw, dtype=torch.float32).to(self.device).permute(2, 0, 1).reshape(1, 3, 224, 224)
            x.requires_grad = True
            y = torch.tensor(y_raw, dtype=torch.long).to(self.device).reshape(1)

            # Vorhersage für das Originalbild
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            # Gradienten berechnen
            model.zero_grad()
            loss.backward()

            # FGSM Angriff durchführen
            x_perturbed = self.fgsm_attack(x, self.epsilon)

            # Vorhersage für das manipulierte Bild
            output_perturbed = model(x_perturbed)
            y_tilde = output_perturbed.argmax(dim=1)

            # Prüfen, ob die KI getäuscht wurde
            if y_tilde != y:
                self.cache["failing_samples"] += 1
            self.cache["total_samples"] += 1

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        # Ergebnisse aus dem Cache abrufen
        failing = self.cache.get("failing_samples", 0)
        total = self.cache.get("total_samples", 0)
        
        # Ratio berechnen (Erfolgsquote des Angriffs)
        ratio = failing / total if total > 0 else 0
        result = {"ratio": ratio}

        # Pie Chart erstellen
        sizes = [ratio, 1 - ratio]
        labels = ["Failing (Fooled)", "Not failing (Robust)"]
        fig = px.pie(values=sizes, names=labels, title="Ratio of failing samples under FGSM Attack")

        display = [fig]
        return CheckResult(result, display=display)

    @staticmethod
    def fgsm_attack(x, epsilon):
        """Implementiert den FGSM Angriff: x_tilde = x + epsilon * sign(grad(L))."""
        # Erstellt das Rauschen basierend auf dem Vorzeichen des Gradienten
        x_tilde = x + epsilon * x.grad.data.sign()
        # Werte auf [0, 1] begrenzen, damit es ein gültiges Bild bleibt
        x_tilde = torch.clamp(x_tilde, 0, 1)
        return x_tilde